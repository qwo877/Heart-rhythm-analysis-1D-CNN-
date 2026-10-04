"""逐拍端到端評估（README 可行性判準 F1–F4）
  1. DS1 上比較 R 峰偵測器（GQRS / XQRS × 是否校正峰位）。只用 DS1 挑選：
     先以偵測 F 值選方法，再以「與標註位置的中位數偏移」決定是否校正
  2. 打包部署 bundle（models/holter_beat.pt）：train_beat.py 訓練的 multi-scale 模型（3 個 seed 權重）
     + 訓練集 RR 標準化統計 + 選定的偵測器。不重新訓練
  3. DS2：理想 R 峰（醫師標註；重現訓練時的測試數字當對照）vs 自動 R 峰（真實條件）
  4. DS2 每筆紀錄的 PVC / PAC 負擔：系統 vs 參考
"""
import argparse
import json
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import torch

from src.dataset.cache import load_beats_cached
from src.holter import (
    CLS_S,
    CLS_V,
    MODEL_FS,
    beat_confusion,
    classify_beats,
    end2end_metrics,
    load_beat_bundle,
    save_bundle,
)
from src.preprocess.qrs import QRS_METHODS, detection_stats, refine_peaks, run_detector
from src.preprocess.symbol_mapping import CLASS_NAMES, build_symbol_map
from src.preprocess.wfdb_io import read_annotation, read_signal
from src.splits.aami_split import make_mitdb_split
from src.utils import load_config, make_output_dir, resolve_device, set_seed, snapshot_config

SRC_RUN = os.path.join("runs", "20260708_221912_train_beat")   # train_beat.py 的輸出（3 個 seed）
SEEDS = [42, 43, 44]
TOL_S = 0.150                     # EC57 比對容差
BURDEN_BINS = (1.0, 10.0)         # PVC 負擔分級（%）：<1、1–10、≥10
KEYS = [f"seed{s}" for s in SEEDS] + ["ensemble"]


def load_mitdb_record(rec, cfg, sym_map):
    #讀 MLII 訊號與 AAMI beat 標註（只留五類 beat，非 beat 標記略過，與訓練期相同
    d = cfg["data"]
    sig, fs = read_signal(rec, data_dir=d["data_dir"], channel=d.get("channel", "MLII"))
    ann = read_annotation(rec, "atr", data_dir=d["data_dir"])
    ref_s, ref_y = [], []
    for s, sym in zip(ann.sample, ann.symbol):
        c = sym_map.get(sym)
        if c is not None:
            ref_s.append(int(s))
            ref_y.append(c)
    return sig, fs, np.asarray(ref_s, np.int64), np.asarray(ref_y, np.int64)


def cfg_key(method, correct):
    return f"{method}_{'corr' if correct else 'raw'}"


#偵測器選擇（只看 DS1）
def select_detector(ds1, cfg, sym_map, peaks_store):
    agg = {cfg_key(m, c): {"tp": 0, "fn": 0, "fp": 0, "off": [], "sec": 0.0}
           for m in QRS_METHODS for c in (True, False)}
    for rec in ds1:
        sig, fs, ref_s, _ = load_mitdb_record(rec, cfg, sym_map)
        tol = int(TOL_S * fs)
        line = [f"  DS1 {rec}:"]
        for m in QRS_METHODS:
            t0 = time.perf_counter()
            raw = run_detector(sig, fs, m)
            dt = time.perf_counter() - t0
            for c in (True, False):
                t1 = time.perf_counter()
                pk = refine_peaks(sig, fs, raw, correct=c)
                k = cfg_key(m, c)
                st = detection_stats(ref_s, pk, tol)
                a = agg[k]
                a["tp"] += st["tp"]; a["fn"] += st["fn"]; a["fp"] += st["fp"]
                a["off"].append(st["offsets"])
                a["sec"] += dt + (time.perf_counter() - t1)
                peaks_store.setdefault(k, {})[rec] = pk
                if c:
                    line.append(f"{m} Se={st['se']:.4f} +P={st['ppv']:.4f}")
        print("  ".join(line))
        sys.stdout.flush()

    summary = {}
    for k, a in agg.items():
        se = a["tp"] / (a["tp"] + a["fn"])
        ppv = a["tp"] / (a["tp"] + a["fp"])
        off = np.abs(np.concatenate(a["off"])) / MODEL_FS * 1000.0  # ms
        summary[k] = {"se": se, "ppv": ppv, "f": 2 * se * ppv / (se + ppv),
                      "tp": a["tp"], "fn": a["fn"], "fp": a["fp"],
                      "median_abs_offset_ms": float(np.median(off)),
                      "p95_abs_offset_ms": float(np.percentile(off, 95)), "seconds": a["sec"]}
    best_m = max(QRS_METHODS, key=lambda m: max(summary[cfg_key(m, c)]["f"] for c in (True, False)))
    best_c = min((True, False), key=lambda c: summary[cfg_key(best_m, c)]["median_abs_offset_ms"])
    return best_m, best_c, summary


def format_detector_table(summary, best):
    lines = ["R 峰偵測器選擇（只用 DS1，22 筆；±150 ms 比對）", "",
             f"{'config':<11}{'Se':>9}{'+P':>9}{'F':>9}{'FN':>7}{'FP':>7}{'|off| med':>11}{'p95':>8}{'sec':>8}",
             "-" * 79]
    for k, s in summary.items():
        mark = "  ← 選定" if k == best else ""
        lines.append(f"{k:<11}{s['se']:>9.4f}{s['ppv']:>9.4f}{s['f']:>9.4f}{s['fn']:>7d}{s['fp']:>7d}"
                     f"{s['median_abs_offset_ms']:>9.1f}ms{s['p95_abs_offset_ms']:>6.1f}ms{s['seconds']:>8.1f}{mark}")
    lines += ["", "規則：先以 F 值選方法，再以與標註位置的中位數 |偏移| 決定是否校正（模型的單拍窗以標註位置為中心訓練）"]
    return "\n".join(lines)


# 2. 部署 bundle
def build_beat_bundle(cfg, src_run, method, correct, path):
    tr, _, _ = load_beats_cached(cfg, verbose=True)
    rr_mean, rr_std = tr["rr"].mean(0), tr["rr"].std(0)   # 與 train_beat.build_loaders 相同
    sds = []
    for s in SEEDS:
        ck = torch.load(os.path.join(src_run, f"seed{s}", "multi_scale", "best_model.pt"),
                        map_location="cpu", weights_only=False)
        sds.append(ck["model_state_dict"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_bundle(path, "beat", sds, seeds=SEEDS, source_run=src_run,
                rr_mean=rr_mean, rr_std=rr_std, window=cfg["data"]["window"],
                rr_local_window=cfg["model"].get("rr_local_window", 10), rr_clip=(0.2, 2.0),
                class_names=CLASS_NAMES, fs=MODEL_FS, qrs={"method": method, "correct": correct})
    print(f"[bundle] 已存 {path}（{len(sds)} 個 seed 權重 + RR 統計 + 偵測器 {cfg_key(method, correct)}）")



# DS2 評估
def burden_category(pct):
    lo, hi = BURDEN_BINS
    return "<1%" if pct < lo else ("1-10%" if pct < hi else ">=10%")


def evaluate_ds2(ds2, cfg, sym_map, bundle, device, method, correct, peaks_store):
    C = len(CLASS_NAMES)
    ideal = {k: np.zeros((C, C), np.int64) for k in KEYS}
    auto = {k: {"cm": np.zeros((C, C), np.int64), "missed": np.zeros(C, np.int64),
                "false": np.zeros(C, np.int64)} for k in KEYS}
    det_tot = {"tp": 0, "fn": 0, "fp": 0}
    rows = []
    k_cfg = cfg_key(method, correct)
    for rec in ds2:
        sig, fs, ref_s, ref_y = load_mitdb_record(rec, cfg, sym_map)
        tol = int(TOL_S * fs)

        # 理想 R 峰：與訓練期（train_beat.py）完全相同的取樣方式（丟邊界窗、batch 64）
        out = classify_beats(sig, fs, ref_s, bundle, device=device, pad_edges=False, batch_size=64)
        y = ref_y[out["valid"]]
        for i, k in enumerate(KEYS[:-1]):
            np.add.at(ideal[k], (y, out["per_model_pred"][i]), 1)
        np.add.at(ideal["ensemble"], (y, out["pred"]), 1)

        # 自動 R 峰：真實條件（每個偵測到的拍都要判，邊界窗補值）
        pk = peaks_store.get(k_cfg, {}).get(rec)
        if pk is None:
            pk = refine_peaks(sig, fs, run_detector(sig, fs, method), correct=correct)
            peaks_store.setdefault(k_cfg, {})[rec] = pk
        st = detection_stats(ref_s, pk, tol)
        for kk in ("tp", "fn", "fp"):
            det_tot[kk] += st[kk]
        out2 = classify_beats(sig, fs, pk, bundle, device=device, pad_edges=True)
        preds = list(out2["per_model_pred"]) + [out2["pred"]]
        for k, p in zip(KEYS, preds):
            cm, mi, fa = beat_confusion(ref_s, ref_y, pk, p, tol)
            auto[k]["cm"] += cm; auto[k]["missed"] += mi; auto[k]["false"] += fa

        n_ref, n_sys = len(ref_y), len(pk)
        row = {"record": rec, "ref_beats": n_ref, "det_beats": n_sys,
               "det_se": st["se"], "det_ppv": st["ppv"], "det_fn": st["fn"], "det_fp": st["fp"]}
        for name, c in (("V", CLS_V), ("S", CLS_S)):
            rc, sc = int((ref_y == c).sum()), int((out2["pred"] == c).sum())
            rb, sb = 100.0 * rc / n_ref, 100.0 * sc / max(n_sys, 1)
            row.update({f"ref_{name}": rc, f"sys_{name}": sc, f"ref_{name}_burden": rb,
                        f"sys_{name}_burden": sb, f"{name}_abs_err_pp": abs(sb - rb)})
        row["ref_V_cat"] = burden_category(row["ref_V_burden"])
        row["sys_V_cat"] = burden_category(row["sys_V_burden"])
        rows.append(row)
        print(f"  DS2 {rec}: 偵測 Se={st['se']:.4f} +P={st['ppv']:.4f} | "
              f"V {row['ref_V']}→{row['sys_V']}  S {row['ref_S']}→{row['sys_S']}")
        sys.stdout.flush()
    return ideal, auto, det_tot, rows


def metrics_table(ideal, auto):
    #每個 key 的 N/S/V Se/+P/F1 與 macro-F1（5 類，與既有表同定義）
    res = {"ideal": {}, "auto": {}}
    for k in KEYS:
        res["ideal"][k] = end2end_metrics(ideal[k])
        res["auto"][k] = end2end_metrics(auto[k]["cm"], auto[k]["missed"], auto[k]["false"])
    return res


def _get(rows, cls, field):
    return next(r[field] for r in rows if r["class"] == cls)


def seed_stats(res_mode, cls, field):
    vals = np.array([_get(res_mode[k], cls, field) for k in KEYS[:-1]], dtype=np.float64)
    return float(vals.mean()), float(vals.std(ddof=0))   # ddof=0 與全專案既有表一致


def format_end2end_table(res):
    lines = ["DS2 逐拍端到端（inter-patient，22 筆；multi-scale 3 seeds）", "",
             "ideal = 醫師標註的 R 峰（訓練時的條件）；auto = 自動偵測的 R 峰（真實條件，漏偵測算漏判、誤偵測算進 +P 分母）", ""]
    hdr = f"{'mode':<7}{'model':<13}" + "".join(f"{c + '-' + f:>16}" for c in ("N", "S", "V") for f in ("Se", "+P", "F1"))
    lines += [hdr, "-" * len(hdr)]
    for mode in ("ideal", "auto"):
        cells = []
        for c in ("N", "S", "V"):
            for f in ("se", "ppv", "f1"):
                m, s = seed_stats(res[mode], c, f)
                cells.append(f"{m:.3f}±{s:.3f}")
        lines.append(f"{mode:<7}{'3-seed mean':<13}" + "".join(f"{x:>16}" for x in cells))
        cells = [f"{_get(res[mode]['ensemble'], c, f):.3f}" for c in ("N", "S", "V") for f in ("se", "ppv", "f1")]
        lines.append(f"{mode:<7}{'ensemble':<13}" + "".join(f"{x:>16}" for x in cells))
    lines.append("-" * len(hdr))
    return "\n".join(lines)


def tier(full, cond):
    return "可行" if full else ("有條件可行" if cond else "不可行")


def judge(res, det_tot, rows):
    se_d = det_tot["tp"] / (det_tot["tp"] + det_tot["fn"])
    ppv_d = det_tot["tp"] / (det_tot["tp"] + det_tot["fp"])
    ens = res["auto"]["ensemble"]
    v_se, v_ppv = _get(ens, "V", "se"), _get(ens, "V", "ppv")
    s_se, s_ppv = _get(ens, "S", "se"), _get(ens, "S", "ppv")
    err = np.array([r["V_abs_err_pp"] for r in rows])
    agree = float(np.mean([r["ref_V_cat"] == r["sys_V_cat"] for r in rows]))
    med = float(np.median(err))
    return [
        {"id": "F1", "item": "R 峰偵測（DS2）", "value": f"Se={se_d:.2%} +P={ppv_d:.2%}",
         "verdict": tier(se_d >= .99 and ppv_d >= .99, se_d >= .97 and ppv_d >= .97)},
        {"id": "F2", "item": "V（PVC）逐拍，端到端", "value": f"Se={v_se:.1%} +P={v_ppv:.1%}",
         "verdict": tier(v_se >= .90 and v_ppv >= .80, v_se >= .80 and v_ppv >= .60)},
        {"id": "F3", "item": "S（PAC）逐拍，端到端", "value": f"Se={s_se:.1%} +P={s_ppv:.1%}",
         "verdict": tier(s_se >= .80 and s_ppv >= .60, s_se >= .60 and s_ppv >= .40)},
        {"id": "F4", "item": "每筆 PVC 負擔（DS2）",
         "value": f"中位數絕對誤差={med:.2f} 個百分點，分級一致 {agree:.0%}",
         "verdict": tier(med <= 1.0 and agree >= .90, med <= 3.0 and agree >= .75)},
    ]


def make_figure(res, rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    GRAY, BLUE, GREEN, VERM = "#9AA0A6", "#0072B2", "#009E73", "#D55E00"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))

    # A. 理想 vs 自動 R 峰（ensemble）
    groups = [("V", "se"), ("V", "ppv"), ("S", "se"), ("S", "ppv")]
    labels = ["V Se", "V +P", "S Se", "S +P"]
    ideal = [_get(res["ideal"]["ensemble"], c, f) for c, f in groups]
    auto = [_get(res["auto"]["ensemble"], c, f) for c, f in groups]
    x = np.arange(len(groups))
    ax[0].bar(x - 0.19, ideal, 0.38, color=GRAY, edgecolor="black", lw=0.6, label="annotated R-peaks")
    ax[0].bar(x + 0.19, auto, 0.38, color=BLUE, edgecolor="black", lw=0.6, label="automatic R-peaks")
    for xi, (a, b) in enumerate(zip(ideal, auto)):
        ax[0].text(xi - 0.19, a + 0.015, f"{a:.2f}", ha="center", fontsize=8)
        ax[0].text(xi + 0.19, b + 0.015, f"{b:.2f}", ha="center", fontsize=8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels); ax[0].set_ylim(0, 1.08)
    ax[0].set_title("A. Beat classification on DS2 (ensemble)\nannotated vs automatic R-peaks", fontsize=10.5)
    ax[0].legend(frameon=False, fontsize=8.5, loc="upper right")

    # B/C. 每筆負擔
    for a_, name, title in ((ax[1], "V", "B. PVC burden per record (DS2)"),
                            (ax[2], "S", "C. PAC (SVEB) burden per record (DS2)")):
        rb = np.array([r[f"ref_{name}_burden"] for r in rows])
        sb = np.array([r[f"sys_{name}_burden"] for r in rows])
        lim = max(rb.max(), sb.max()) * 1.15 + 0.5
        a_.plot([0, lim], [0, lim], color="black", lw=1)
        if name == "V":
            for b in BURDEN_BINS:
                a_.axvline(b, color=GRAY, ls=":", lw=1); a_.axhline(b, color=GRAY, ls=":", lw=1)
        a_.scatter(rb, sb, s=28, color=GREEN if name == "V" else VERM, edgecolor="black", lw=0.5, zorder=3)
        for r, xv, yv in zip(rows, rb, sb):
            if abs(yv - xv) > 2.0:
                a_.annotate(r["record"], (xv, yv), textcoords="offset points", xytext=(4, 3), fontsize=7.5)
        a_.set_xscale("symlog", linthresh=1.0); a_.set_yscale("symlog", linthresh=1.0)
        a_.set_xlim(0, lim); a_.set_ylim(0, lim)
        a_.set_xlabel(f"reference {name} burden (%)"); a_.set_ylabel(f"system {name} burden (%)")
        a_.set_title(title, fontsize=10.5)

    fig.suptitle("Offline Holter feasibility: end-to-end beat analysis with automatic R-peak detection",
                 fontsize=12.5, y=1.02, fontweight="bold")
    fig.tight_layout()
    png = os.path.join(out_dir, "end2end_figure.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "end2end_figure.pdf"), bbox_inches="tight")
    plt.close(fig)
    return png


def replication_check(res, src_run):
    """理想 R 峰模式應重現 train_beat.py 測試報表的 per-seed 數字（驗證管線實作正確）。"""
    with open(os.path.join(src_run, "ablation_results.json"), encoding="utf-8") as f:
        src = json.load(f)
    lines = ["重現檢查（理想 R 峰 vs 訓練時的 test 報表）："]
    ok = True
    for r in src["runs"]:
        if r["variant"] != "multi_scale":
            continue
        k = f"seed{r['seed']}"
        mine = {c: _get(res["ideal"][k], c, "f1") for c in ("S", "V")}
        diff = max(abs(mine["S"] - r["S_f1"]), abs(mine["V"] - r["V_f1"]))
        ok &= diff < 5e-4
        lines.append(f"  {k}: S-F1 {mine['S']:.4f} vs {r['S_f1']:.4f} | V-F1 {mine['V']:.4f} vs {r['V_f1']:.4f}")
    lines.append("  → 一致" if ok else "  → 有差異（檢查前處理是否與訓練期相同）")
    return "\n".join(lines), ok


def save_peaks(peaks_store, cache_dir="./cache"):
    os.makedirs(cache_dir, exist_ok=True)
    for k, d in peaks_store.items():
        path = os.path.join(cache_dir, f"rpeaks_{k}.npz")
        old = dict(np.load(path)) if os.path.exists(path) else {}
        old.update({rec: v for rec, v in d.items()})
        np.savez(path, **old)


def main():
    ap = argparse.ArgumentParser(description="逐拍端到端評估（F1–F4）")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--src-run", default=SRC_RUN, help="train_beat.py 的輸出目錄（3 個 seed 的 multi-scale 權重）")
    ap.add_argument("--bundle", default=os.path.join("models", "holter_beat.pt"))
    args, _ = ap.parse_known_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"], cfg.get("deterministic", True))
    device = resolve_device(cfg.get("device", "auto"))
    out_dir = make_output_dir(cfg["output"]["root"], tag="eval_end2end")
    snapshot_config(args.config, out_dir)
    print(f"Device: {device} | 輸出目錄: {out_dir}\n")

    sym_map = build_symbol_map()
    tr_rec, va_rec, ds2 = make_mitdb_split(cfg["data"])
    ds1 = tr_rec + va_rec
    peaks_store = {}

    print("[1] R 峰偵測器選擇（DS1）")
    method, correct, det_summary = select_detector(ds1, cfg, sym_map, peaks_store)
    det_text = format_detector_table(det_summary, cfg_key(method, correct))
    print("\n" + det_text + "\n")

    print("[2] 打包部署 bundle")
    build_beat_bundle(cfg, args.src_run, method, correct, args.bundle)
    bundle = load_beat_bundle(args.bundle, device=device)

    print("\n[3] DS2 評估（理想 vs 自動 R 峰）")
    ideal, auto, det_tot, rows = evaluate_ds2(ds2, cfg, sym_map, bundle, device, method, correct, peaks_store)
    save_peaks(peaks_store)

    res = metrics_table(ideal, auto)
    table = format_end2end_table(res)
    rep_text, rep_ok = replication_check(res, args.src_run)
    verdicts = judge(res, det_tot, rows)
    ver_text = "\n".join(["可行性判定（README 可行性判準，判定用部署版本 = ensemble）："] +
                         [f"  {v['id']} {v['item']:<18} {v['value']:<42} → {v['verdict']}" for v in verdicts])
    print("\n" + table + "\n\n" + rep_text + "\n\n" + ver_text)

    with open(os.path.join(out_dir, "detector_selection.txt"), "w", encoding="utf-8") as f:
        f.write(det_text + "\n")
    with open(os.path.join(out_dir, "end2end_table.txt"), "w", encoding="utf-8") as f:
        f.write(table + "\n\n" + rep_text + "\n\n" + ver_text + "\n")
    import csv
    with open(os.path.join(out_dir, "per_record.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(out_dir, "end2end_results.json"), "w", encoding="utf-8") as f:
        json.dump({"detector": {"method": method, "correct": correct, "ds1": det_summary},
                   "ds2_detection": det_tot, "metrics": res, "replication_ok": rep_ok,
                   "verdicts": verdicts, "per_record": rows,
                   "confusion": {"ideal": {k: v.tolist() for k, v in ideal.items()},
                                 "auto": {k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in auto.items()}}},
                  f, ensure_ascii=False, indent=2, default=float)
    png = make_figure(res, rows, out_dir)
    print(f"\n圖：{png}\n輸出目錄：{out_dir}")


if __name__ == "__main__":
    main()
