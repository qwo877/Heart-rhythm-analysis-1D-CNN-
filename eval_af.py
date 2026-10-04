#AF 偵測（README 可行性判準 F5–F7）

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
import wfdb

from src.dataset.cache import load_af_cached
from src.dataset.torch_ds import RhythmDataset
from src.engine import make_loader, plot_history, train_and_evaluate
from src.holter import (
    af_windows,
    classify_af_windows,
    load_af_bundle,
    load_beat_bundle,
    rhythm_burden,
    rhythm_segments,
    rr_level_af,
    save_bundle,
)
from src.metrics import af_f1, af_report
from src.model import build_model
from src.preprocess.rhythm import label_samples_by_rhythm
from src.preprocess.symbol_mapping import build_symbol_map
from src.preprocess.wfdb_io import read_annotation
from src.utils import load_config, make_output_dir, resolve_device, set_seed, snapshot_config

N_FOLDS = 5
AF = frozenset({"AFIB"})
MITDB_PACED = {"102", "104", "107", "217"}


def build_af_loaders(train, val, test, cfg):
    #afdb RR 窗 → DataLoader（訓練集加 RR jitter；部署版本不用 HRV）
    at = cfg["af_train"]

    def ds(d, augment):
        return RhythmDataset(d["X"], d["y"], augment=augment, normalize=at["normalize"])

    kw = dict(num_workers=at["num_workers"], seed=cfg["seed"])
    return (make_loader(ds(train, at["augment"]), at["batch_size"], shuffle=True, drop_last=True, **kw),
            make_loader(ds(val, False), at["batch_size"], **kw),
            make_loader(ds(test, False), at["batch_size"], **kw))


def window_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    tp = int(((y == 1) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    se = tp / (tp + fn) if tp + fn else float("nan")
    sp = tn / (tn + fp) if tn + fp else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
    return {"se": se, "sp": sp, "ppv": ppv, "f1": f1, "tp": tp, "fn": fn, "fp": fp, "tn": tn}


def tier(full, cond):
    return "可行" if full else ("有條件可行" if cond else "不可行")


# afdb 病人層級交叉驗證
@torch.no_grad()
def predict_probs(model, X, device, batch_size=4096):
    model.eval()
    x = torch.from_numpy(np.asarray(X, dtype=np.float32)).unsqueeze(1)
    out = [torch.softmax(model(x[i:i + batch_size].to(device)), 1)[:, 1].cpu().numpy()
           for i in range(0, len(x), batch_size)]
    return np.concatenate(out)


def afdb_cv(cfg, device, out_dir):
    tr, va, te = load_af_cached(cfg, verbose=True)
    D = {k: np.concatenate([tr[k], va[k], te[k]]) for k in ("X", "y", "hrv", "groups")}
    records = sorted(set(D["groups"].tolist()))
    perm = np.random.default_rng(cfg["seed"]).permutation(records)
    folds = [sorted(f.tolist()) for f in np.array_split(perm, N_FOLDS)]
    at = cfg["af_train"]
    assert not at.get("use_hrv", False), "部署版本不用 HRV late fusion"

    prob_all = np.zeros(len(D["y"]), np.float32)
    fold_rows, state_dicts = [], []
    for k in range(N_FOLDS):
        test_r, val_r = folds[k], folds[(k + 1) % N_FOLDS]
        train_r = [r for r in records if r not in set(test_r) | set(val_r)]
        pick = lambda rs: {kk: D[kk][np.isin(D["groups"], rs)] for kk in D}  # noqa: E731
        dtr, dva, dte = pick(train_r), pick(val_r), pick(test_r)
        print(f"\n===== fold {k + 1}/{N_FOLDS} | test {test_r} | val {val_r} | train {len(train_r)} 筆 =====")

        set_seed(cfg["seed"], cfg.get("deterministic", True))
        loaders = build_af_loaders(dtr, dva, dte, cfg)
        counts = np.bincount(dtr["y"], minlength=2).astype(np.float64)
        w = torch.tensor(counts.sum() / (counts + 1e-8), dtype=torch.float32, device=device) \
            if at["use_class_weights"] else None
        model = build_model("rhythmcnn", 2, dropout=at.get("dropout", 0.3)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=at["lr"])
        fdir = os.path.join(out_dir, f"fold{k + 1}")
        os.makedirs(fdir, exist_ok=True)
        hist, rep, _, _ = train_and_evaluate(
            model, *loaders, torch.nn.CrossEntropyLoss(weight=w), opt, epochs=at["epochs"], device=device,
            output_dir=fdir, select_fn=af_f1, report_fn=af_report, metric_name="af_f1",
            test_groups=dte["groups"])
        plot_history(hist, os.path.join(fdir, "train_history.png"), metric_name="AF-F1")
        # train_and_evaluate 已載回 val 最佳權重
        state_dicts.append({kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()})
        m = np.isin(D["groups"], test_r)
        prob_all[m] = predict_probs(model, D["X"][m], device)
        wm = window_metrics(D["y"][m], prob_all[m] >= 0.5)
        fold_rows.append({"fold": k + 1, "test": test_r, "val": val_r, **wm})
        print(f"  fold {k + 1}: Se={wm['se']:.4f} Sp={wm['sp']:.4f} F1={wm['f1']:.4f}")

    pred_all = (prob_all >= 0.5).astype(np.int64)
    pooled = window_metrics(D["y"], pred_all)
    # 每筆 AF 負擔（afdb 快取只有窗 → 以窗時長加權：窗內 RR 總和）
    dur = D["X"].sum(1)
    rec_rows = []
    for r in records:
        m = D["groups"] == r
        tb = float((dur[m] * D["y"][m]).sum() / dur[m].sum())
        pb = float((dur[m] * pred_all[m]).sum() / dur[m].sum())
        rec_rows.append({"record": r, "true_af_burden": tb, "sys_af_burden": pb, "abs_err_pp": 100 * abs(pb - tb),
                         **{f"win_{k}": v for k, v in window_metrics(D["y"][m], pred_all[m]).items()
                            if k in ("se", "sp")}})
    return {"folds": folds, "fold_rows": fold_rows, "pooled": pooled, "records": rec_rows,
            "state_dicts": state_dicts}


# 3–4. MIT-BIH 外部測試
def mitdb_external(cfg, af_bundle, device, rpeaks_path):
    data_dir = cfg["data"]["data_dir"]
    sym_map = build_symbol_map()
    auto_peaks = dict(np.load(rpeaks_path))
    records = sorted(os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith(".hea"))
    records = [r for r in records if r not in MITDB_PACED]
    W, S_, TH, CLIP = af_bundle["window"], af_bundle["step"], af_bundle["af_threshold"], tuple(af_bundle["rr_clip"])

    ys = {"auto": [], "annotated": []}
    ps = {"auto": [], "annotated": []}
    rows = []
    for rec in records:
        if rec not in auto_peaks:
            print(f"   {rec} 沒有自動 R 峰快取，跳過"); continue
        hdr = wfdb.rdheader(os.path.join(data_dir, rec))
        fs, n_samples = hdr.fs, hdr.sig_len
        ann = read_annotation(rec, "atr", data_dir=data_dir)
        segs = rhythm_segments(ann)
        ref_beats = np.asarray([s for s, sym in zip(ann.sample, ann.symbol) if sym in sym_map], np.int64)
        row = {"record": rec}
        for mode, peaks in (("auto", auto_peaks[rec]), ("annotated", ref_beats)):
            X, starts, rr = af_windows(peaks, fs, W, S_, CLIP)
            rr_true = label_samples_by_rhythm(peaks[1:], segs, af_rhythms=AF)
            y = np.array([int(rr_true[s:s + W].mean() >= TH) for s in starts], np.int64)
            prob, _ = classify_af_windows(X, af_bundle, device=device)
            ys[mode].append(y); ps[mode].append((prob >= 0.5).astype(np.int64))
            if mode == "auto":
                _, rr_af = rr_level_af(len(rr), starts, W, prob, TH)
                row["sys_af_burden"] = float(rr[rr_af == 1].sum() / rr.sum())
        _, tb = rhythm_burden(segs, n_samples, fs, "AFIB")
        row["true_af_burden"] = float(tb)
        row["abs_err_pp"] = 100 * abs(row["sys_af_burden"] - tb)
        rows.append(row)
        flag = "  ← AF" if tb > 0 else ("  ← 假 AF" if row["sys_af_burden"] > 0.05 else "")
        print(f"  MIT-BIH {rec}: AF 負擔 參考 {tb:6.1%}  系統 {row['sys_af_burden']:6.1%}{flag}")
        sys.stdout.flush()
    res = {m: window_metrics(np.concatenate(ys[m]), np.concatenate(ps[m])) for m in ys}
    return res, rows


def make_figure(cv, ext_rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, VERM, GRAY = "#0072B2", "#D55E00", "#9AA0A6"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for a, rows, title, col in ((ax[0], cv["records"], "A. afdb, patient-level 5-fold CV (23 records)", BLUE),
                                (ax[1], ext_rows, "B. MIT-BIH external test (44 records,\nautomatic R-peaks)", VERM)):
        tb = np.array([r["true_af_burden"] for r in rows]) * 100
        sb = np.array([r["sys_af_burden"] for r in rows]) * 100
        a.plot([0, 100], [0, 100], color="black", lw=1)
        a.fill_between([0, 100], [-5, 95], [5, 105], color=GRAY, alpha=0.2, lw=0, label="±5 pp")
        a.scatter(tb, sb, s=30, color=col, edgecolor="black", lw=0.5, zorder=3)
        for r, x, y in zip(rows, tb, sb):
            if abs(y - x) > 10:
                a.annotate(r["record"], (x, y), textcoords="offset points", xytext=(4, 3), fontsize=7.5)
        a.set_xlim(-3, 103); a.set_ylim(-3, 103)
        a.set_xlabel("reference AF burden (%)"); a.set_ylabel("system AF burden (%)")
        a.set_title(title, fontsize=10.5); a.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Offline Holter feasibility: AF burden per record", fontsize=12.5, y=1.02, fontweight="bold")
    fig.tight_layout()
    png = os.path.join(out_dir, "af_figure.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "af_figure.pdf"), bbox_inches="tight")
    plt.close(fig)
    return png


def main():
    ap = argparse.ArgumentParser(description="AF 可行性評估（F5–F7）")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--beat-bundle", default=os.path.join("models", "holter_beat.pt"),
                    help="讀其中的 qrs 設定，決定用哪份自動 R 峰快取")
    ap.add_argument("--af-bundle", default=os.path.join("models", "holter_af.pt"))
    args, _ = ap.parse_known_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.get("device", "auto"))
    out_dir = make_output_dir(cfg["output"]["root"], tag="eval_af")
    snapshot_config(args.config, out_dir)
    print(f"Device: {device} | 輸出目錄: {out_dir}")

    print("\n[1] afdb 病人層級 5-fold 交叉驗證")
    cv = afdb_cv(cfg, device, out_dir)
    a = cfg["afdb"]
    save_bundle(args.af_bundle, "af", cv["state_dicts"], window=a.get("window", 32), step=a.get("step", 16),
                af_threshold=a.get("af_threshold", 0.5), rr_clip=tuple(a.get("rr_clip") or (0.2, 2.0)),
                folds=cv["folds"])
    print(f"\n[2] AF bundle 已存 {args.af_bundle}（{N_FOLDS} 個 fold 模型）")

    print("\n[3] MIT-BIH 外部測試")
    q = load_beat_bundle(args.beat_bundle)["qrs"]
    rpeaks = os.path.join("cache", f"rpeaks_{q['method']}_{'corr' if q['correct'] else 'raw'}.npz")
    af_bundle = load_af_bundle(args.af_bundle, device=device)
    ext, ext_rows = mitdb_external(cfg, af_bundle, device, rpeaks)

    P = cv["pooled"]
    fm = np.array([[r["se"], r["sp"], r["f1"]] for r in cv["fold_rows"]])
    cv_err = np.array([r["abs_err_pp"] for r in cv["records"]])
    ext_err = np.array([r["abs_err_pp"] for r in ext_rows])
    ext_err_af = np.array([r["abs_err_pp"] for r in ext_rows if r["true_af_burden"] > 0])
    false_af = [r for r in ext_rows if r["true_af_burden"] == 0 and r["sys_af_burden"] > 0.05]
    E = ext["auto"]
    lines = [
        "AF 偵測（窗層級；AF 比例 ≥ 0.5 標為 AF）", "",
        f"afdb 5-fold（pooled，23 筆）：Se={P['se']:.4f} Sp={P['sp']:.4f} PPV={P['ppv']:.4f} F1={P['f1']:.4f}",
        f"  各 fold mean±std：Se={fm[:, 0].mean():.4f}±{fm[:, 0].std():.4f}  Sp={fm[:, 1].mean():.4f}±{fm[:, 1].std():.4f}"
        f"  F1={fm[:, 2].mean():.4f}±{fm[:, 2].std():.4f}",
    ] + [f"  fold{r['fold']} test={r['test']}: Se={r['se']:.4f} Sp={r['sp']:.4f} F1={r['f1']:.4f}" for r in cv["fold_rows"]] + [
        "",
        f"MIT-BIH 外部（44 筆）自動 R 峰：Se={E['se']:.4f} Sp={E['sp']:.4f} PPV={E['ppv']:.4f} F1={E['f1']:.4f}",
        f"MIT-BIH 外部（44 筆）標註 R 峰：Se={ext['annotated']['se']:.4f} Sp={ext['annotated']['sp']:.4f} "
        f"PPV={ext['annotated']['ppv']:.4f} F1={ext['annotated']['f1']:.4f}",
        "",
        f"每筆 AF 負擔絕對誤差（百分點）：afdb 中位數 {np.median(cv_err):.2f}（最大 {cv_err.max():.1f}）｜"
        f"MIT-BIH 中位數 {np.median(ext_err):.2f}；只看含 AF 的 {len(ext_err_af)} 筆：中位數 {np.median(ext_err_af):.2f}",
        f"MIT-BIH 無 AF 卻被判 AF 負擔 > 5% 的紀錄：{[(r['record'], round(100 * r['sys_af_burden'], 1)) for r in false_af]}",
    ]
    f7_full = np.median(cv_err) <= 5 and np.median(ext_err_af) <= 5
    f7_cond = np.median(cv_err) <= 10 and np.median(ext_err_af) <= 10
    verdicts = [
        {"id": "F5", "item": "AF 窗層級（afdb 5-fold）", "value": f"Se={P['se']:.1%} Sp={P['sp']:.1%}",
         "verdict": tier(P["se"] >= .95 and P["sp"] >= .95, P["se"] >= .90 and P["sp"] >= .90)},
        {"id": "F6", "item": "AF 外部（MIT-BIH，自動 R 峰）", "value": f"Se={E['se']:.1%} Sp={E['sp']:.1%}",
         "verdict": tier(E["se"] >= .90 and E["sp"] >= .90, E["se"] >= .80 and E["sp"] >= .80)},
        {"id": "F7", "item": "每筆 AF 負擔",
         "value": f"afdb 中位數 {np.median(cv_err):.1f}、MIT-BIH（含 AF 者）中位數 {np.median(ext_err_af):.1f} 個百分點",
         "verdict": tier(f7_full, f7_cond)},
    ]
    lines += ["", "可行性判定（F7 取兩個資料集中較差者；MIT-BIH 只計含 AF 的紀錄，避免 37 筆無 AF 紀錄把中位數壓成 0）："]
    lines += [f"  {v['id']} {v['item']:<22} {v['value']:<46} → {v['verdict']}" for v in verdicts]
    text = "\n".join(lines)
    print("\n" + text)

    with open(os.path.join(out_dir, "af_table.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(os.path.join(out_dir, "af_results.json"), "w", encoding="utf-8") as f:
        json.dump({"cv": {k: v for k, v in cv.items() if k != "state_dicts"}, "external": ext,
                   "external_records": ext_rows, "verdicts": verdicts}, f, ensure_ascii=False, indent=2, default=float)
    png = make_figure(cv, ext_rows, out_dir)
    print(f"\n圖：{png}\n輸出目錄：{out_dir}")


if __name__ == "__main__":
    main()
