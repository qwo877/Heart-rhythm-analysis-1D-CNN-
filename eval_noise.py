#雜訊魯棒性（README 可行性判準 F9）

import argparse
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import wfdb

from eval_end2end import TOL_S, load_mitdb_record
from src.holter import CLS_N, beat_confusion, classify_beats, end2end_metrics, load_beat_bundle
from src.preprocess.qrs import detect_r_peaks, detection_stats
from src.preprocess.symbol_mapping import CLASS_NAMES, build_symbol_map
from src.splits.aami_split import make_mitdb_split
from src.utils import load_config, make_output_dir, resolve_device, set_seed

NOISES = ["em", "ma", "bw"]
SNRS = [24, 18, 12, 6, 0]


def load_noise(data_dir):
    if not all(os.path.exists(os.path.join(data_dir, f"{n}.dat")) for n in NOISES):
        os.makedirs(data_dir, exist_ok=True)
        print(f"下載 NSTDB 雜訊紀錄 {NOISES} 到 {data_dir} …")
        wfdb.dl_database("nstdb", data_dir, records=NOISES)
    out = {}
    for n in NOISES:
        rec = wfdb.rdrecord(os.path.join(data_dir, n))
        x = rec.p_signal[:, 0].astype(np.float64)
        out[n] = x - x.mean()
    return out


def qrs_signal_power(sig, ref_s, ref_y, fs):
    #nst 式的訊號功率：正常拍 QRS 峰對峰振幅中位數的平方 / 8
    h = int(0.05 * fs)
    pp = [np.ptp(sig[s - h:s + h]) for s, y in zip(ref_s, ref_y) if y == CLS_N and s - h >= 0 and s + h < len(sig)]
    return float(np.median(pp)) ** 2 / 8.0


def add_noise(sig, noise, s_power, snr_db):
    n = np.resize(noise, len(sig))            # 長度不足時循環
    scale = np.sqrt(s_power / (n.var() * 10 ** (snr_db / 10.0)))
    return (sig + scale * n).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="雜訊魯棒性（F9）")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--beat-bundle", default=os.path.join("models", "holter_beat.pt"))
    ap.add_argument("--nstdb-dir", default="./nstdb_data")
    args, _ = ap.parse_known_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"], cfg.get("deterministic", True))
    device = resolve_device(cfg.get("device", "auto"))
    out_dir = make_output_dir(cfg["output"]["root"], tag="eval_noise")
    bundle = load_beat_bundle(args.beat_bundle, device=device)
    q = bundle["qrs"]
    noise = load_noise(args.nstdb_dir)
    _, _, ds2 = make_mitdb_split(cfg["data"])
    sym_map = build_symbol_map()
    C = len(CLASS_NAMES)

    conds = [("clean", None)] + [(n, s) for n in NOISES for s in SNRS]
    acc = {c: {"cm": np.zeros((C, C), np.int64), "missed": np.zeros(C, np.int64), "false": np.zeros(C, np.int64),
               "tp": 0, "fn": 0, "fp": 0} for c in conds}
    for rec in ds2:
        sig, fs, ref_s, ref_y = load_mitdb_record(rec, cfg, sym_map)
        tol = int(TOL_S * fs)
        sp = qrs_signal_power(sig, ref_s, ref_y, fs)
        for c in conds:
            x = sig if c[0] == "clean" else add_noise(sig, noise[c[0]], sp, c[1])
            pk = detect_r_peaks(x, fs, method=q["method"], correct=q["correct"])
            st = detection_stats(ref_s, pk, tol)
            out = classify_beats(x, fs, pk, bundle, device=device, pad_edges=True)
            cm, mi, fa = beat_confusion(ref_s, ref_y, pk, out["pred"], tol)
            a = acc[c]
            a["cm"] += cm; a["missed"] += mi; a["false"] += fa
            a["tp"] += st["tp"]; a["fn"] += st["fn"]; a["fp"] += st["fp"]
        print(f"  {rec} 完成（{len(conds)} 個條件）")
        sys.stdout.flush()

    rows = []
    for c in conds:
        a = acc[c]
        m = {r["class"]: r for r in end2end_metrics(a["cm"], a["missed"], a["false"])}
        rows.append({"noise": c[0], "snr_db": c[1],
                     "det_se": a["tp"] / (a["tp"] + a["fn"]), "det_ppv": a["tp"] / (a["tp"] + a["fp"]),
                     "V_se": m["V"]["se"], "V_ppv": m["V"]["ppv"], "S_se": m["S"]["se"], "S_ppv": m["S"]["ppv"]})

    def ok_cond(r):
        return r["det_se"] >= .97 and r["det_ppv"] >= .97 and r["V_se"] >= .80 and r["V_ppv"] >= .60

    def at(noise_name, snr):
        return next(r for r in rows if r["noise"] == noise_name and r["snr_db"] == snr)

    full = all(ok_cond(at(n, 12)) for n in ("em", "ma"))
    cond = all(ok_cond(at(n, 18)) for n in ("em", "ma"))
    verdict = "可行" if full else ("有條件可行" if cond else "不可行")

    hdr = f"{'noise':<7}{'SNR':>6}{'R Se':>9}{'R +P':>9}{'V Se':>9}{'V +P':>9}{'S Se':>9}{'S +P':>9}"
    lines = ["雜訊魯棒性（DS2 22 筆，NSTDB 雜訊全程疊加；自動 R 峰 + ensemble）", "", hdr, "-" * len(hdr)]
    for r in rows:
        snr = "—" if r["snr_db"] is None else f"{r['snr_db']}dB"
        lines.append(f"{r['noise']:<7}{snr:>6}{r['det_se']:>9.3f}{r['det_ppv']:>9.3f}{r['V_se']:>9.3f}"
                     f"{r['V_ppv']:>9.3f}{r['S_se']:>9.3f}{r['S_ppv']:>9.3f}")
    lines += ["", f"F9 判定（em、ma 在 12 dB 維持 F1/F2 有條件可行 → 可行；18 dB 才維持 → 有條件可行）：{verdict}"]
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(out_dir, "noise_table.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(os.path.join(out_dir, "noise_results.json"), "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "verdict": verdict}, f, ensure_ascii=False, indent=2, default=float)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"em": "#D55E00", "ma": "#0072B2", "bw": "#009E73"}
    clean = rows[0]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
    for a, key, title in ((ax[0], "det_ppv", "R-peak +P"), (ax[1], "V_se", "PVC (V) Se"), (ax[2], "V_ppv", "PVC (V) +P")):
        for n in NOISES:
            ys = [at(n, s)[key] for s in SNRS]
            a.plot(SNRS, ys, "-o", color=colors[n], ms=4, label=n)
        a.axhline(clean[key], color="black", ls="--", lw=1, label="clean")
        a.set_title(title, fontsize=10.5); a.set_xlabel("SNR (dB)"); a.invert_xaxis(); a.set_ylim(0, 1.02)
    ax[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle("Noise stress test (NSTDB em / ma / bw added to DS2)", fontsize=12, y=1.03, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "noise_figure.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "noise_figure.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"\n輸出目錄：{out_dir}")


if __name__ == "__main__":
    main()
