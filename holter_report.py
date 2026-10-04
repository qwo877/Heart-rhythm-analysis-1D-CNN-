"""應用展示 — Holter 輔助判讀報告（PC 端）  (AI生成，非正式醫療，需醫師複核)

輸入一筆單導程 ECG，跑完整部署管線（自動 R 峰 → 逐拍 N/S/V → AF → 摘要），
產出一份自帶圖表的 HTML 報告（離線可開、可列印）與 JSON 摘要。

    python holter_report.py --record 203                 # MIT-BIH 本機紀錄（有標註時附「系統 vs 醫師標註」對照）
    python holter_report.py --wfdb path/to/record        # 任何 WFDB 紀錄（不含副檔名）
    python holter_report.py --csv ecg.csv --fs 250       # 單欄 CSV（mV）
輸出 reports/<name>_holter_report.html 與 reports/<name>_summary.json。
"""
import argparse
import base64
import datetime
import html
import io
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import wfdb  # noqa: E402

from src.holter import (  # noqa: E402
    CLS_S,
    CLS_V,
    analyze_record,
    load_af_bundle,
    load_beat_bundle,
    rhythm_burden,
    rhythm_segments,
    runs_of,
)
from src.preprocess.qrs import match_beats  # noqa: E402
from src.preprocess.symbol_mapping import CLASS_NAMES, build_symbol_map  # noqa: E402

COL = {"N": "#9AA0A6", "S": "#E69F00", "V": "#D55E00", "F": "#CC79A7", "Q": "#CC79A7"}
AF_COL, BLUE = "#0072B2", "#0072B2"


def hms(sec):
    sec = int(round(sec))
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# 輸入
def load_input(args, data_dir):
    """回傳 (signal, fs, name, reference or None)。reference 含 beats/labels/segments（在原始 fs）。"""
    if args.csv:
        sig = np.loadtxt(args.csv, delimiter=",", dtype=np.float64).ravel()
        return sig, float(args.fs), os.path.splitext(os.path.basename(args.csv))[0], None
    path = args.wfdb or os.path.join(data_dir, str(args.record))
    rec = wfdb.rdrecord(path)
    ch = rec.sig_name.index("MLII") if "MLII" in rec.sig_name else 0
    sig = rec.p_signal[:, ch].astype(np.float64)
    name = os.path.basename(path)
    ref = None
    if os.path.exists(path + ".atr"):
        ann = wfdb.rdann(path, "atr")
        sym_map = build_symbol_map()
        pairs = [(int(s), sym_map[sym]) for s, sym in zip(ann.sample, ann.symbol) if sym in sym_map]
        ref = {"beats": np.array([p[0] for p in pairs], np.int64),
               "labels": np.array([p[1] for p in pairs], np.int64),
               "segments": rhythm_segments(ann)}
    return sig, float(rec.fs), name, ref


def reference_summary(ref, fs, n_samples, res):
    """醫師標註的對照數字（在原始 fs 下算），與系統輸出並列。"""
    y = ref["labels"]
    n = len(y)
    out = {"n_beats": n, "V": int((y == CLS_V).sum()), "S": int((y == CLS_S).sum()),
           "pvc_burden": float((y == CLS_V).mean()) if n else 0.0,
           "pac_burden": float((y == CLS_S).mean()) if n else 0.0,
           "v_runs": len(runs_of(y, CLS_V, 3))}
    _, af = rhythm_burden(ref["segments"], n_samples, fs, "AFIB")
    out["af_burden"] = af
    # 逐拍對照：把參考標註換到系統的 360 Hz 座標後比對（±150 ms）
    scale = res["fs"] / fs
    rb = np.round(ref["beats"] * scale).astype(np.int64)
    r2d, _ = match_beats(rb, res["peaks"], int(0.15 * res["fs"]))
    out["ref_label_of_peak"] = np.full(len(res["peaks"]), -1, np.int64)
    m = r2d >= 0
    out["ref_label_of_peak"][r2d[m]] = y[m]
    out["ref_beats_model_fs"] = rb
    return out


# 圖
def plot_trend(res):
    fs, peaks, pred = res["fs"], res["peaks"], res["beat_pred"]
    dur = len(res["signal"]) / fs
    bin_s = 60.0 if dur <= 4 * 3600 else 300.0
    nb = int(np.ceil(dur / bin_s))
    t = peaks / fs
    rr = np.diff(peaks) / fs
    hr = np.full(nb, np.nan)
    v = np.zeros(nb); s = np.zeros(nb)
    for b in range(nb):
        m = (t[1:] >= b * bin_s) & (t[1:] < (b + 1) * bin_s) & (rr > 0.2) & (rr < 3.0)
        if m.sum() >= 5:
            hr[b] = 60.0 / rr[m].mean()
        mb = (t >= b * bin_s) & (t < (b + 1) * bin_s)
        v[b] = (pred[mb] == CLS_V).sum(); s[b] = (pred[mb] == CLS_S).sum()
    x = (np.arange(nb) + 0.5) * bin_s / 60.0
    fig, ax = plt.subplots(2, 1, figsize=(11, 4.2), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    ax[0].plot(x, hr, color=BLUE, lw=1.5)
    ax[0].set_ylabel("HR (bpm)")
    ax[0].grid(alpha=0.3)
    w = bin_s / 60.0 * 0.4
    ax[1].bar(x - w / 2, v, w, color=COL["V"], label="PVC (V)")
    ax[1].bar(x + w / 2, s, w, color=COL["S"], label="PAC (S)")
    ax[1].set_ylabel(f"beats / {int(bin_s / 60)} min")
    ax[1].set_xlabel("time (min)")
    ax[1].legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_rhythm(res, ref, fs_in):
    fs = res["fs"]
    dur_min = len(res["signal"]) / fs / 60
    fig, ax = plt.subplots(figsize=(11, 2.3))
    if ref is not None:
        segs = ref["segments"]
        for i, (s0, r) in enumerate(segs):
            e0 = segs[i + 1][0] if i + 1 < len(segs) else len(res["signal"]) / fs * fs_in
            if r == "AFIB":
                ax.axvspan(s0 / fs_in / 60, e0 / fs_in / 60, ymin=0.0, ymax=0.12, color="black", alpha=0.55, lw=0)
    w = res["af_windows"]
    if w is not None and len(w["starts"]):
        peaks = res["peaks"]
        W = w["window"]
        centers = np.array([(peaks[s] + peaks[min(s + W, len(peaks) - 1)]) / 2 for s in w["starts"]]) / fs / 60
        ax.fill_between(centers, 0, w["prob"], color=AF_COL, alpha=0.35, lw=0, step="mid")
        ax.plot(centers, w["prob"], color=AF_COL, lw=0.8, drawstyle="steps-mid")
    for s0, e0, _d in res["summary"].get("af_episodes", []):
        ax.axvspan(s0 / fs / 60, e0 / fs / 60, ymin=0.9, ymax=1.0, color=AF_COL, alpha=0.9, lw=0)
    ax.axhline(0.5, color="black", ls=":", lw=0.8)
    ax.set_xlim(0, dur_min); ax.set_ylim(0, 1.05)
    ax.set_ylabel("AF prob."); ax.set_xlabel("time (min)")
    ax.spines[["top", "right"]].set_visible(False)
    note = "top bar = system AF episode (≥30 s)" + ("; bottom bar = reference AF (annotation)" if ref is not None else "")
    ax.set_title(note, fontsize=8.5, loc="left", color="#555")
    fig.tight_layout()
    return fig_to_b64(fig)


def pick_strips(res, max_strips=4):
    """挑要看的 10 秒片段：最長 V 連發 / 第一個 PVC、第一個 PAC、第一段 AF 起點、開頭。"""
    fs, peaks, pred = res["fs"], res["peaks"], res["beat_pred"]
    cands = []
    v_runs = runs_of(pred, CLS_V, 3)
    if v_runs:
        i, L = max(v_runs, key=lambda r: r[1])
        cands.append((peaks[i] / fs, f"Longest V run ({L} beats)"))
    elif (pred == CLS_V).any():
        cands.append((peaks[np.argmax(pred == CLS_V)] / fs, "First PVC"))
    if (pred == CLS_S).any():
        cands.append((peaks[np.argmax(pred == CLS_S)] / fs, "First PAC (unverified class)"))
    eps = res["summary"].get("af_episodes", [])
    if eps:
        cands.append((eps[0][0] / fs, "AF episode onset"))
    cands.append((5.0, "Recording start"))
    seen, out = [], []
    for t, title in cands:
        if all(abs(t - s) > 10 for s in seen):
            seen.append(t); out.append((t, title))
    return out[:max_strips]


def plot_strip(res, t_center, title, ref_info):
    fs, sig, peaks, pred = res["fs"], res["signal"], res["peaks"], res["beat_pred"]
    t0 = max(0.0, t_center - 5.0)
    t1 = min(len(sig) / fs, t0 + 10.0)
    a, b = int(t0 * fs), int(t1 * fs)
    tt = np.arange(a, b) / fs
    fig, ax = plt.subplots(figsize=(11, 1.9))
    ax.plot(tt, sig[a:b], color="black", lw=0.8)
    lo, hi = np.percentile(sig[a:b], [0.5, 99.5])
    pad = (hi - lo) * 0.25
    ax.set_ylim(lo - pad * 1.6, hi + pad * 1.4)
    for i in np.where((peaks >= a) & (peaks < b))[0]:
        name = CLASS_NAMES[pred[i]]
        ax.text(peaks[i] / fs, hi + pad * 0.55, name, ha="center", fontsize=8,
                color=COL[name], fontweight="bold" if name != "N" else "normal")
        if ref_info is not None and ref_info["ref_label_of_peak"][i] >= 0:
            rn = CLASS_NAMES[ref_info["ref_label_of_peak"][i]]
            ax.text(peaks[i] / fs, lo - pad * 1.25, rn, ha="center", fontsize=7, color="#777")
    if ref_info is not None:  # 系統漏掉的參考拍
        rb = ref_info["ref_beats_model_fs"]
        for rbx in rb[(rb >= a) & (rb < b)]:
            if not np.any(np.abs(peaks - rbx) <= int(0.15 * fs)):
                ax.text(rbx / fs, hi + pad * 0.55, "×", ha="center", fontsize=9, color="#D55E00")
    ax.set_xlim(t0, t1)
    ax.set_yticks([])
    ax.set_xlabel("time (s)", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    label = f"{title} — top: system label" + ("; bottom (gray): reference label; × = missed beat" if ref_info is not None else "")
    ax.set_title(label, fontsize=8.5, loc="left", color="#333")
    fig.tight_layout()
    return fig_to_b64(fig)


# HTML
CSS = """
:root{
--ink:#1d2330;
--mut:#5b6474;
--line:#e3e6ec;
--bg:#fafbfc;
--card:#fff;
--acc:#0072B2;
--warn:#b35900
}
*{
box-sizing:border-box
}
body{
margin:0;
padding:24px 16px;
background:var(--bg);
color:var(--ink);
font-family:"Microsoft JhengHei","PingFang TC","Noto Sans TC",system-ui,sans-serif;
font-size:14px;
line-height:1.55
}
.wrap{
max-width:1080px;
margin:0 auto
}
h1{font-size:22px;margin:0 0 4px}
h2{
font-size:16px;
margin:28px 0 10px;
border-left:4px solid var(--acc);
padding-left:8px
}
.meta{color:var(--mut);font-size:13px}
.tag{
display:inline-block;
background:#fff3e0;
color:var(--warn);
border:1px solid #f3d3a8;border-radius:4px;
padding:1px 8px;
font-size:12px;
margin-left:6px;
vertical-align:middle
}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-top:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.card .k{color:var(--mut);font-size:12px}.card .v{font-size:22px;font-weight:700;margin-top:2px}
.card .s{color:var(--mut);font-size:12px}.card.warn .v{color:var(--warn)}
table{
border-collapse:collapse;
width:100%;
background:var(--card);
border:1px solid var(--line);
border-radius:8px
}
th,td{
padding:6px 10px;
border-bottom:1px solid var(--line);
text-align:right;
font-variant-numeric:tabular-nums
}
th:first-child,td:first-child{text-align:left}
th{background:#f3f5f8;font-weight:600;font-size:13px}
.fig{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px;margin-bottom:10px}
.fig img{width:100%;display:block}.scroll{overflow-x:auto}
.note{color:var(--mut);font-size:12.5px}.foot{margin-top:28px;padding:12px 14px;border:1px solid #f3d3a8;
background:#fffaf2;border-radius:8px;font-size:12.5px;color:#6b4a1f}
@media print{body{background:#fff;padding:0}.fig,.card,table{break-inside:avoid}}
"""


def card(k, v, s="", warn=False):
    return (f'<div class="card{" warn" if warn else ""}"><div class="k">{html.escape(k)}</div>'
            f'<div class="v">{v}</div><div class="s">{s}</div></div>')


def build_html(name, res, fs_in, ref, refs, figs, strips, bundles):
    S = res["summary"]
    c = S["counts"]
    eps = S.get("af_episodes", [])
    af_b = S.get("af_burden")
    cards = [
        card("分析長度", hms(S["duration_s"]), f"{S['n_beats']:,} 拍"),
        card("心率（bpm）", f"{S['hr_mean']:.0f}", f"每分鐘平均最低 {S['hr_min_1min']:.0f}／最高 {S['hr_max_1min']:.0f}"),
        card("PVC（V）", f"{c['V']:,}", f"負擔 {S['pvc_burden']:.2%}；成對 {S['v_couplets']}、連發 {S['v_runs']}（最長 {S['v_run_longest']} 拍）"),
        card("PAC（S）", f"{c['S']:,}", f"負擔 {S['pac_burden']:.2%}；"
             + ("AF 期間不計；" if S.get("pac_rule") else "") + "僅供參考", warn=True),
        card("AF 負擔", "—" if af_b is None else f"{af_b:.1%}",
             "—" if af_b is None else f"{len(eps)} 次發作（≥30 秒），共 {hms(S.get('af_time_s', 0))}"),
        card("其他（F／Q）", f"{c['F'] + c['Q']:,}", "樣本少，未納入判讀"),
    ]
    comp = ""
    if refs is not None:
        def row(k, a, b):
            return f"<tr><td>{k}</td><td>{a}</td><td>{b}</td></tr>"
        comp = ("<h2>系統 vs 醫師標註（本紀錄有參考標註）</h2><div class='scroll'><table><tr><th>項目</th><th>系統</th><th>醫師標註</th></tr>"
                + row("心跳數", f"{S['n_beats']:,}", f"{refs['n_beats']:,}")
                + row("PVC 數（負擔）", f"{c['V']:,}（{S['pvc_burden']:.2%}）", f"{refs['V']:,}（{refs['pvc_burden']:.2%}）")
                + row("PAC 數（負擔）", f"{c['S']:,}（{S['pac_burden']:.2%}）", f"{refs['S']:,}（{refs['pac_burden']:.2%}）")
                + row("V 連發（≥3 拍）", f"{S['v_runs']}", f"{refs['v_runs']}")
                + row("AF 負擔", "—" if af_b is None else f"{af_b:.1%}", f"{refs['af_burden']:.1%}")
                + "</table></div>")
    ep_rows = "".join(f"<tr><td>{i + 1}</td><td>{hms(s / res['fs'])}</td><td>{hms(e / res['fs'])}</td><td>{hms(d)}</td></tr>"
                      for i, (s, e, d) in enumerate(eps[:50]))
    ep_tbl = (f"<div class='scroll'><table><tr><th>#</th><th>開始</th><th>結束</th><th>長度</th></tr>{ep_rows}</table></div>"
              if eps else "<p class='note'>未偵測到 ≥ 30 秒的 AF 發作。</p>")
    t = res["timing"]
    strip_html = "".join(f"<div class='fig'><img alt='ECG strip' src='data:image/png;base64,{b}'></div>" for b in strips)
    return f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Holter 輔助判讀報告 — {html.escape(name)}</title>
<style>{CSS}</style></head><body><div class="wrap">
<h1>Holter 輔助判讀報告 <span class="tag">研究原型・需醫師複核</span></h1>
<div class="meta">紀錄 {html.escape(name)}｜原始取樣 {fs_in:g} Hz｜分析於 {datetime.datetime.now():%Y-%m-%d %H:%M}｜
CPU 分析耗時 {t['total']:.1f} 秒｜R 峰偵測 {bundles['qrs']}｜逐拍模型 multi-scale CNN × {bundles['n_beat']}｜AF 模型 RhythmCNN × {bundles['n_af']}</div>
<div class="cards">{''.join(cards)}</div>
{comp}
<h2>心率與異位搏動趨勢</h2><div class="fig"><img alt="trend" src="data:image/png;base64,{figs['trend']}"></div>
<h2>節律時間軸（AF）</h2><div class="fig"><img alt="rhythm" src="data:image/png;base64,{figs['rhythm']}"></div>
{ep_tbl}
<h2>事件心電圖片段（每段 10 秒）</h2>{strip_html}
<div class="foot"><b>使用限制。</b>本報告由研究原型自動產生，不是醫療器材，結果必須由醫師複核。
可行性評估（MIT-BIH inter-patient，見專案 README）顯示：R 峰偵測與 PVC 計數可作為輔助（PVC 需人工刪除部分誤標），
<b>PAC（S）未達可用標準，只能當提示</b>；AF 期間心房沒有規律電活動，PAC 不計入。F／Q 類樣本太少，未納入判讀。
AF 偵測只看 RR 間期，<b>頻繁的早期收縮（二聯律、成串 PVC）可能被誤判成 AF</b>；訊號品質差的片段（電極移動、肌電）可能產生假的心跳與 PVC。</div>
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser(description="Holter 輔助判讀報告（研究原型）")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--record", help="MIT-BIH 本機紀錄名（例：203）")
    src.add_argument("--wfdb", help="WFDB 紀錄路徑（不含副檔名）")
    src.add_argument("--csv", help="單欄 CSV（mV）")
    ap.add_argument("--fs", type=float, help="--csv 的取樣率")
    ap.add_argument("--data-dir", default="./mitdb_data")
    ap.add_argument("--beat-bundle", default=os.path.join("models", "holter_beat.pt"))
    ap.add_argument("--af-bundle", default=os.path.join("models", "holter_af.pt"))
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    if args.csv and not args.fs:
        ap.error("--csv 需要 --fs")

    device = torch.device("cpu")
    beat_b = load_beat_bundle(args.beat_bundle, device=device)
    af_b = load_af_bundle(args.af_bundle, device=device) if os.path.exists(args.af_bundle) else None
    sig, fs_in, name, ref = load_input(args, args.data_dir)

    res = analyze_record(sig, fs_in, beat_b, af_b, device=device)
    refs = reference_summary(ref, fs_in, len(sig), res) if ref is not None else None
    figs = {"trend": plot_trend(res), "rhythm": plot_rhythm(res, ref, fs_in)}
    strips = [plot_strip(res, t, title, refs) for t, title in pick_strips(res)]
    q = beat_b["qrs"]
    bundles = {"qrs": f"{q['method'].upper()}{'（峰位校正）' if q['correct'] else ''}",
               "n_beat": len(beat_b["models"]), "n_af": len(af_b["models"]) if af_b else 0}
    page = build_html(name, res, fs_in, ref, refs, figs, strips, bundles)

    os.makedirs(args.out, exist_ok=True)
    out_html = os.path.join(args.out, f"{name}_holter_report.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(page)
    S = dict(res["summary"])
    S["af_episodes"] = [{"start_s": s / res["fs"], "end_s": e / res["fs"], "duration_s": d}
                        for s, e, d in S.get("af_episodes", [])]
    summary = {"record": name, "fs_input": fs_in, "timing_s": res["timing"], "summary": S}
    if refs is not None:
        summary["reference"] = {k: v for k, v in refs.items() if not isinstance(v, np.ndarray)}
    with open(os.path.join(args.out, f"{name}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    print(f"報告：{out_html}（分析 {res['timing']['total']:.1f} 秒）")


if __name__ == "__main__":
    main()
