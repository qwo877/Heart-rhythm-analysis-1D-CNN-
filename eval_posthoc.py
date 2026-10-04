# 分析
import argparse
import glob
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
from scipy.stats import spearmanr

from eval_end2end import TOL_S, load_mitdb_record
from src.holter import (
    CLS_S,
    CLS_V,
    af_windows,
    beat_confusion,
    classify_af_windows,
    classify_beats,
    end2end_metrics,
    load_af_bundle,
    load_beat_bundle,
    rr_level_af,
    suppress_s_in_af,
)
from src.preprocess.symbol_mapping import build_symbol_map
from src.splits.aami_split import make_mitdb_split
from src.utils import load_config, make_output_dir, resolve_device


def latest(pattern):
    runs = sorted(glob.glob(os.path.join("runs", pattern)))
    assert runs, f"找不到 runs/{pattern}，先跑對應的評估腳本"
    return runs[-1]


def main():
    ap = argparse.ArgumentParser(description="分析")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--beat-bundle", default=os.path.join("models", "holter_beat.pt"))
    ap.add_argument("--af-bundle", default=os.path.join("models", "holter_af.pt"))
    args, _ = ap.parse_known_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.get("device", "auto"))
    out_dir = make_output_dir(cfg["output"]["root"], tag="eval_posthoc")
    beat_b = load_beat_bundle(args.beat_bundle, device=device)
    af_b = load_af_bundle(args.af_bundle, device=device)
    q = beat_b["qrs"]
    peaks_all = dict(np.load(os.path.join("cache", f"rpeaks_{q['method']}_{'corr' if q['correct'] else 'raw'}.npz")))
    sym_map = build_symbol_map()
    _, _, ds2 = make_mitdb_split(cfg["data"])

    # 1. AF 期間不計 PAC
    C = 5
    acc = {k: {"cm": np.zeros((C, C), np.int64), "missed": np.zeros(C, np.int64), "false": np.zeros(C, np.int64)}
           for k in ("raw", "rule")}
    pac_rows = []
    for rec in ds2:
        sig, fs, ref_s, ref_y = load_mitdb_record(rec, cfg, sym_map)
        pk = peaks_all[rec]
        pred = classify_beats(sig, fs, pk, beat_b, device=device, pad_edges=True)["pred"]
        X, starts, rr = af_windows(pk, fs, af_b["window"], af_b["step"], tuple(af_b["rr_clip"]))
        prob, _ = classify_af_windows(X, af_b, device=device)
        _, rr_af = rr_level_af(len(rr), starts, af_b["window"], prob, af_b["af_threshold"])
        pred_rule = suppress_s_in_af(pred, rr_af)
        for k, p in (("raw", pred), ("rule", pred_rule)):
            cm, mi, fa = beat_confusion(ref_s, ref_y, pk, p, int(TOL_S * fs))
            acc[k]["cm"] += cm; acc[k]["missed"] += mi; acc[k]["false"] += fa
        rb = 100 * (ref_y == CLS_S).mean()
        pac_rows.append({"record": rec, "ref_S": int((ref_y == CLS_S).sum()), "raw_S": int((pred == CLS_S).sum()),
                         "rule_S": int((pred_rule == CLS_S).sum()), "ref_burden": rb,
                         "raw_err_pp": abs(100 * (pred == CLS_S).mean() - rb),
                         "rule_err_pp": abs(100 * (pred_rule == CLS_S).mean() - rb)})
    m = {k: {r["class"]: r for r in end2end_metrics(a["cm"], a["missed"], a["false"])} for k, a in acc.items()}

    # 2. 假 AF vs 異位搏動負擔（MIT-BIH 無 AF 紀錄）
    with open(os.path.join(latest("*_eval_af"), "af_results.json"), encoding="utf-8") as f:
        ext = json.load(f)["external_records"]
    fa_rows = []
    for r in ext:
        if r["true_af_burden"] > 0:
            continue
        _, _, _, ref_y = load_mitdb_record(r["record"], cfg, sym_map)
        fa_rows.append({"record": r["record"], "false_af_pct": 100 * r["sys_af_burden"],
                        "ectopy_pct": 100 * float(np.isin(ref_y, [CLS_V, CLS_S]).mean())})
    rho, pval = spearmanr([r["ectopy_pct"] for r in fa_rows], [r["false_af_pct"] for r in fa_rows])
    hi = [r for r in fa_rows if r["false_af_pct"] > 5]

    lines = [
        "分析（不在事先訂定的判準內；看過結果後才做，只作為改良方向）", "",
        "1. AF 期間不計 PAC（DS2，自動 R 峰，ensemble）",
        f"   S：Se {m['raw']['S']['se']:.3f} → {m['rule']['S']['se']:.3f}；+P {m['raw']['S']['ppv']:.3f} → {m['rule']['S']['ppv']:.3f}；"
        f"F1 {m['raw']['S']['f1']:.3f} → {m['rule']['S']['f1']:.3f}",
        f"   V（不受影響，對照）：Se {m['raw']['V']['se']:.3f} → {m['rule']['V']['se']:.3f}；+P {m['raw']['V']['ppv']:.3f} → {m['rule']['V']['ppv']:.3f}",
        f"   每筆 PAC 負擔中位數絕對誤差：{np.median([r['raw_err_pp'] for r in pac_rows]):.2f} → "
        f"{np.median([r['rule_err_pp'] for r in pac_rows]):.2f} 個百分點；最大 {max(r['raw_err_pp'] for r in pac_rows):.1f} → "
        f"{max(r['rule_err_pp'] for r in pac_rows):.1f}",
    ] + [f"     {r['record']}: 參考 {r['ref_S']:>5}  原始 {r['raw_S']:>5}  規則後 {r['rule_S']:>5}"
         for r in pac_rows if abs(r["raw_S"] - r["rule_S"]) > 0] + [
        "",
        f"2. MIT-BIH 無 AF 紀錄（{len(fa_rows)} 筆）：假 AF 負擔 vs 參考異位搏動（V+S）負擔 Spearman ρ={rho:.2f}（p={pval:.1e}）",
    ] + [f"     {r['record']}: 假 AF {r['false_af_pct']:5.1f}%  異位搏動 {r['ectopy_pct']:5.1f}%"
         for r in sorted(hi, key=lambda r: -r["false_af_pct"])]
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(out_dir, "posthoc.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(os.path.join(out_dir, "posthoc.json"), "w", encoding="utf-8") as f:
        json.dump({"pac_rule": {"metrics": m, "records": pac_rows},
                   "false_af_vs_ectopy": {"rho": rho, "p": pval, "records": fa_rows}},
                  f, ensure_ascii=False, indent=2, default=float)
    print(f"\n輸出目錄：{out_dir}")


if __name__ == "__main__":
    main()
