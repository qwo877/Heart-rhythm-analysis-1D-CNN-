"""訓練逐拍分類模型（部署用的 multi-scale CNN），並與只看形態的 single-scale 對照。

同一份資料、同樣的訓練設定，只差有沒有 RR context 分支：
  single_scale：只看單拍形態（ECGCNN）
  multi_scale ：單拍形態 + 4 維 RR 特徵（MultiScaleECGNet，部署版本）
跑多個 seed 並報 mean±std，避免單一 seed 的 run-to-run 雜訊誤導結論。
eval_end2end.py 會把這裡 3 個 seed 的 multi_scale 權重打包成 models/holter_beat.pt。

    python train_beat.py                 # 用 config 的 ablation.seeds
    python train_beat.py --seeds 42 43 44
輸出 runs/<ts>_train_beat/（每個 seed 的 best_model.pt 與 test_report.txt、ablation_table.txt、ablation_results.json）。
"""
import argparse
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import torch

from src.dataset.mitdb import prepare_beat_data
from src.dataset.torch_ds import BeatDataset
from src.engine import make_loader, plot_history, train_and_evaluate
from src.metrics import aami_report, format_report, macro_f1_score
from src.model import build_model
from src.preprocess.symbol_mapping import CLASS_NAMES, num_classes_from_map
from src.utils import load_config, make_output_dir, resolve_device, set_seed, snapshot_config

METRIC_KEYS = ["accuracy", "macro_f1", "N_f1", "S_f1", "V_f1", "F_f1"]


def build_loaders(data, cfg, multiscale, seed):
    t = cfg["train"]
    (Xtr, ytr, rtr), (Xva, yva, rva), (Xte, yte, rte) = data

    if multiscale:
        rr_mean, rr_std = rtr.mean(0), rtr.std(0)
        rr_tr, rr_va, rr_te = rtr, rva, rte
    else:
        rr_tr = rr_va = rr_te = None
        rr_mean = rr_std = None

    train_ds = BeatDataset(Xtr, ytr, rr=rr_tr, rr_mean=rr_mean, rr_std=rr_std,
                           augment=t["augment"], normalize=t["normalize"])
    val_ds = BeatDataset(Xva, yva, rr=rr_va, rr_mean=rr_mean, rr_std=rr_std,
                         augment=False, normalize=t["normalize"])
    test_ds = BeatDataset(Xte, yte, rr=rr_te, rr_mean=rr_mean, rr_std=rr_std,
                          augment=False, normalize=t["normalize"])

    train_loader = make_loader(train_ds, t["batch_size"], shuffle=True,
                               num_workers=t["num_workers"], seed=seed, drop_last=True)
    val_loader = make_loader(val_ds, t["batch_size"], num_workers=t["num_workers"], seed=seed)
    test_loader = make_loader(test_ds, t["batch_size"], num_workers=t["num_workers"], seed=seed)
    return train_loader, val_loader, test_loader


def run_variant(name, model_name, data, cfg, device, parent_dir, num_classes, seed):
    # 每個 (seed, variant) 從該 seed 起跑，確保資料順序一致、公平比較
    set_seed(seed, cfg.get("deterministic", True))
    out_dir = os.path.join(parent_dir, f"seed{seed}", name)
    os.makedirs(out_dir, exist_ok=True)

    ytr = data[0][1]
    multiscale = model_name == "multiscale"
    train_loader, val_loader, test_loader = build_loaders(data, cfg, multiscale, seed)

    t = cfg["train"]
    class_weights = None
    if t["use_class_weights"]:
        counts = np.bincount(ytr, minlength=num_classes).astype(np.float64)
        class_weights = torch.tensor(counts.sum() / (counts + 1e-8),
                                     dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    model = build_model(model_name, num_classes,
                        dropout=cfg["model"].get("dropout", 0.3)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=t["lr"])
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n===== seed {seed} | variant: {name} ({model_name}, {n_params:,} params) =====")
    history, report, te_acc, te_loss = train_and_evaluate(
        model, train_loader, val_loader, test_loader, criterion, optimizer,
        epochs=t["epochs"], device=device, output_dir=out_dir,
        select_fn=lambda yt, yp: macro_f1_score(yt, yp, num_classes),
        report_fn=lambda yt, yp, g: aami_report(yt, yp, CLASS_NAMES),
        metric_name="macro_f1",
    )
    plot_history(history, os.path.join(out_dir, "train_history.png"), metric_name="macro-F1")

    with open(os.path.join(out_dir, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"seed={seed}  variant={name}  model={model_name}  params={n_params}\n")
        f.write(f"overall accuracy: {te_acc:.4f}\n\n{format_report(report)}\n")

    f1 = {r["class"]: r["f1"] for r in report["per_class"]}
    return {
        "seed": seed, "variant": name, "model": model_name, "params": n_params,
        "accuracy": te_acc, "macro_f1": report["macro_f1"],
        "N_f1": f1["N"], "S_f1": f1["S"], "V_f1": f1["V"], "F_f1": f1["F"],
    }


def aggregate(rows, variant):
    #把某 variant 在多個 seed 的 metric 聚合成 mean/std dict
    sub = [r for r in rows if r["variant"] == variant]
    out = {"variant": variant, "params": sub[0]["params"], "n_seeds": len(sub)}
    for k in METRIC_KEYS:
        vals = np.array([r[k] for r in sub], dtype=np.float64)
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std(ddof=0))
    return out


def main():
    ap = argparse.ArgumentParser(description="訓練逐拍模型：single vs multi-scale（多 seed）")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="覆寫 config 的 ablation.seeds")
    args, _ = ap.parse_known_args()

    cfg = load_config(args.config)
    seeds = args.seeds or cfg.get("ablation", {}).get("seeds", [cfg["seed"]])
    device = resolve_device(cfg.get("device", "auto"))
    parent_dir = make_output_dir(cfg["output"]["root"], tag="train_beat")
    snapshot_config(args.config, parent_dir)
    print(f"Device: {device} | seeds: {seeds} | 輸出目錄: {parent_dir}")

    num_classes = num_classes_from_map()
    print("\n載入 mitdb（含 RR 特徵）…")
    set_seed(seeds[0], cfg.get("deterministic", True))
    data = prepare_beat_data(cfg, verbose=True)

    rows = []
    for seed in seeds:
        rows.append(run_variant("single_scale", "ecgcnn", data, cfg, device, parent_dir, num_classes, seed))
        rows.append(run_variant("multi_scale", "multiscale", data, cfg, device, parent_dir, num_classes, seed))

    ss = aggregate(rows, "single_scale")
    ms = aggregate(rows, "multi_scale")

    # ── mean±std 比較表 ──
    def fmt(agg, key):
        return f"{agg[f'{key}_mean']:.4f}±{agg[f'{key}_std']:.4f}"

    header = f"{'variant':<14}{'params':>9}  {'acc':>15}{'macroF1':>16}{'S_F1':>16}{'V_F1':>16}"
    lines = [f"single vs multi-scale（n={len(seeds)}，seeds={seeds}）— DS2 inter-patient", "", header, "-" * len(header)]
    for agg in (ss, ms):
        lines.append(
            f"{agg['variant']:<14}{agg['params']:>9,}  "
            f"{fmt(agg, 'accuracy'):>15}{fmt(agg, 'macro_f1'):>16}{fmt(agg, 'S_f1'):>16}{fmt(agg, 'V_f1'):>16}"
        )
    lines.append("-" * len(header))
    # 配對 delta（同 seed 的 multi 減 single，再報跨 seed 的 mean±std）
    paired = {}
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r["seed"], {})[r["variant"]] = r
    for k in METRIC_KEYS:
        deltas = np.array([by_seed[s]["multi_scale"][k] - by_seed[s]["single_scale"][k] for s in seeds])
        paired[k] = (float(deltas.mean()), float(deltas.std(ddof=0)))
    lines.append(
        f"{'Δ multi−single':<14}{'':>9}  "
        f"{paired['accuracy'][0]:>+9.4f}±{paired['accuracy'][1]:.4f}"
        f"{paired['macro_f1'][0]:>+9.4f}±{paired['macro_f1'][1]:.4f}"
        f"{paired['S_f1'][0]:>+9.4f}±{paired['S_f1'][1]:.4f}"
        f"{paired['V_f1'][0]:>+9.4f}±{paired['V_f1'][1]:.4f}"
    )

    s_mean, s_std = paired["S_f1"]
    macro_mean = paired["macro_f1"][0]
    robust = s_mean > 2 * s_std and s_mean > 0  # 提升幅度 > 2σ 視為穩健
    if robust:
        verdict = f"multi-scale 穩健勝出（S 類 F1 Δ={s_mean:+.4f} > 2σ={2*s_std:.4f}）"
    elif macro_mean > 0:
        verdict = f"multi-scale 勝出（macro-F1 Δ={macro_mean:+.4f}；S 類 F1 Δ 未超過 2σ）"
    else:
        verdict = "multi-scale 未勝過 single-scale"
    table = "\n".join(lines)
    print("\n===== single-scale vs multi-scale（多 seed，DS2 inter-patient）=====")
    print(table)
    print(f"\n判定：{verdict}")

    with open(os.path.join(parent_dir, "ablation_table.txt"), "w", encoding="utf-8") as f:
        f.write(table + "\n\n" + verdict + "\n")
    with open(os.path.join(parent_dir, "ablation_results.json"), "w", encoding="utf-8") as f:
        json.dump({"seeds": seeds, "runs": rows,
                   "single_scale": ss, "multi_scale": ms,
                   "paired_delta": paired, "verdict": verdict},
                  f, ensure_ascii=False, indent=2)
    print(f"\n表格已存至 {parent_dir}")


if __name__ == "__main__":
    main()
