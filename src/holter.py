"""離線 Holter 輔助判讀管線（應用可行性驗證的核心）。

    原始單導程 ECG ─► 自動 R 峰偵測 ─► 單拍窗 + RR 特徵 ─► multi-scale CNN（seed ensemble）逐拍 N/S/V/F/Q
                                  └─► 32 拍 RR 窗 ─► RhythmCNN（fold ensemble）─► AF 時段
    ─► 整筆紀錄摘要：心跳數、心率、PVC / PAC 數與負擔、連發、AF 發作與負擔

與 Phase 0–2 的關鍵差別：不再使用醫師標註的 R 峰，一切由原始訊號自動算出。
模型權重與前處理參數打包成 bundle（models/*.pt），評估與報告讀同一份 → 「評估的就是部署的」。
"""
import time

import numpy as np
import torch
from scipy.signal import resample_poly

from .model import MultiScaleECGNet, RhythmCNN
from .preprocess.qrs import detect_r_peaks, match_beats
from .preprocess.rhythm import beat_rr_features, label_samples_by_rhythm
from .preprocess.symbol_mapping import CLASS_NAMES

MODEL_FS = 360            # 逐拍模型以 mitdb 360 Hz 的 256 點窗訓練；其他取樣率先重取樣
CLS_N, CLS_S, CLS_V = 0, 1, 2
AF_MIN_EPISODE_S = 30.0   # AF 發作的最短長度（README 可行性判準）


# ---------------------------------------------------------------------------
# bundle：部署用的權重 + 前處理參數
# ---------------------------------------------------------------------------
def save_bundle(path, kind, state_dicts, **meta):
    torch.save({"kind": kind, "state_dicts": state_dicts, **meta}, path)


def load_beat_bundle(path, device="cpu"):
    b = torch.load(path, map_location=device, weights_only=False)
    assert b["kind"] == "beat", f"{path} 不是 beat bundle"
    models = []
    for sd in b["state_dicts"]:
        m = MultiScaleECGNet(num_classes=len(b["class_names"]), n_rr_features=len(b["rr_mean"]))
        m.load_state_dict(sd)
        models.append(m.to(device).eval())
    b["models"] = models
    return b


def load_af_bundle(path, device="cpu"):
    b = torch.load(path, map_location=device, weights_only=False)
    assert b["kind"] == "af", f"{path} 不是 AF bundle"
    models = []
    for sd in b["state_dicts"]:
        m = RhythmCNN(num_classes=2)
        m.load_state_dict(sd)
        models.append(m.to(device).eval())
    b["models"] = models
    return b


# ---------------------------------------------------------------------------
# 逐拍分類
# ---------------------------------------------------------------------------
def to_model_fs(signal, fs):
    """重取樣到 MODEL_FS（360 Hz）。回傳 (signal float32, MODEL_FS)。"""
    if int(round(fs)) == MODEL_FS:
        return np.asarray(signal, dtype=np.float32), MODEL_FS
    g = np.gcd(int(round(fs)), MODEL_FS)
    out = resample_poly(np.asarray(signal, dtype=np.float64), MODEL_FS // g, int(round(fs)) // g)
    return out.astype(np.float32), MODEL_FS


def beat_windows(signal, peaks, window, pad_edges=True):
    """以 R 峰為中心切單拍窗。pad_edges=False 時丟掉窗超出邊界的拍（與訓練期 load_beats 相同）。

    回傳 (X (n_valid, window) float32, valid mask (n_peaks,))。
    """
    half = window // 2
    peaks = np.asarray(peaks, dtype=np.int64)
    n = len(signal)
    if pad_edges:
        padded = np.pad(np.asarray(signal, dtype=np.float32), (half, window - half), mode="edge")
        idx = peaks[:, None] + np.arange(window)[None, :]          # padded 座標：start = peak - half + half
        return padded[idx], np.ones(len(peaks), dtype=bool)
    valid = (peaks - half >= 0) & (peaks - half + window <= n)
    starts = peaks[valid] - half
    idx = starts[:, None] + np.arange(window)[None, :]
    return np.asarray(signal, dtype=np.float32)[idx], valid


@torch.no_grad()
def classify_beats(signal, fs, peaks, bundle, device="cpu", pad_edges=True, batch_size=4096):
    """對給定 R 峰逐拍分類。signal 必須已是 MODEL_FS。

    RR 特徵用「全部」peaks 算（與訓練期相同：先算整筆 RR，再丟邊界窗）。
    回傳 dict：pred (n_valid,)、probs (n_valid, C)、per_model_pred (M, n_valid)、valid (n_peaks,)。
    """
    assert int(round(fs)) == MODEL_FS, "先用 to_model_fs 重取樣"
    rr = beat_rr_features(peaks, fs, local_window=bundle["rr_local_window"], rr_clip=tuple(bundle["rr_clip"]))
    X, valid = beat_windows(signal, peaks, bundle["window"], pad_edges=pad_edges)
    rr = rr[valid]
    C = len(bundle["class_names"])
    if len(X) == 0:
        return {"pred": np.zeros(0, np.int64), "probs": np.zeros((0, C), np.float32),
                "per_model_pred": np.zeros((len(bundle["models"]), 0), np.int64), "valid": valid}
    # 與 BeatDataset(normalize=True) 相同：每拍 z-score；RR 用訓練集統計量標準化
    Xn = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
    rrn = (rr - bundle["rr_mean"]) / (bundle["rr_std"] + 1e-8)
    xm = torch.from_numpy(Xn.astype(np.float32)).unsqueeze(1)
    xr = torch.from_numpy(rrn.astype(np.float32))
    probs_sum = np.zeros((len(X), C), dtype=np.float64)
    per_model = []
    for m in bundle["models"]:
        preds, probs = [], []
        for i in range(0, len(X), batch_size):
            out = m(xm[i:i + batch_size].to(device), xr[i:i + batch_size].to(device))
            p = torch.softmax(out, dim=1).cpu().numpy()
            probs.append(p)
            preds.append(out.argmax(1).cpu().numpy())
        probs = np.concatenate(probs)
        probs_sum += probs
        per_model.append(np.concatenate(preds))
    probs_mean = (probs_sum / len(bundle["models"])).astype(np.float32)
    return {"pred": probs_mean.argmax(1), "probs": probs_mean,
            "per_model_pred": np.stack(per_model), "valid": valid}


def beat_confusion(ref_samples, ref_labels, det_samples, det_pred, tol, num_classes=len(CLASS_NAMES)):
    """端到端逐拍混淆（評估定義見 README「可行性判準」）。

    cm[t, p]：配對成功、真 t 被判為 p 的拍數；missed[t]：沒被偵測到的真 t 拍；
    false_pred[p]：沒有對應真拍（誤偵測）卻被判為 p 的拍數。
    """
    ref_to_det, det_to_ref = match_beats(ref_samples, det_samples, tol)
    ref_labels = np.asarray(ref_labels, dtype=np.int64)
    det_pred = np.asarray(det_pred, dtype=np.int64)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    m = ref_to_det >= 0
    np.add.at(cm, (ref_labels[m], det_pred[ref_to_det[m]]), 1)
    missed = np.bincount(ref_labels[~m], minlength=num_classes)
    false_pred = np.bincount(det_pred[det_to_ref < 0], minlength=num_classes)
    return cm, missed, false_pred


def end2end_metrics(cm, missed=None, false_pred=None, class_names=CLASS_NAMES):
    """由 beat_confusion 的三個計數算每類 Se / +P / F1（漏偵測算漏判、誤偵測算進 +P 分母）。"""
    C = len(class_names)
    missed = np.zeros(C, np.int64) if missed is None else missed
    false_pred = np.zeros(C, np.int64) if false_pred is None else false_pred
    rows = []
    for c, name in enumerate(class_names):
        tp = cm[c, c]
        n_true = cm[c].sum() + missed[c]
        n_pred = cm[:, c].sum() + false_pred[c]
        se = tp / n_true if n_true else float("nan")
        ppv = tp / n_pred if n_pred else float("nan")
        f1 = 2 * se * ppv / (se + ppv) if n_true and n_pred and (se + ppv) > 0 else 0.0
        rows.append({"class": name, "se": float(se), "ppv": float(ppv), "f1": float(f1),
                     "n_true": int(n_true), "n_pred": int(n_pred)})
    return rows


# ---------------------------------------------------------------------------
# AF
# ---------------------------------------------------------------------------
def rhythm_segments(ann):
    """rhythm annotation（symbol '+'、aux_note '(AFIB' 等）→ [(start_sample, rhythm), ...]。

    mitdb 的 .atr 混有其他帶 aux_note 的註記，只取 '+' 開頭為 '(' 者；afdb 全部都是這種。
    """
    segs = []
    aux = ann.aux_note if ann.aux_note is not None else [""] * len(ann.sample)
    for s, sym, a in zip(ann.sample, ann.symbol, aux):
        if sym == "+" and a and a.startswith("("):
            r = a.lstrip("(").strip().strip("\x00")
            if r:
                segs.append((int(s), r))
    segs.sort(key=lambda x: x[0])
    return segs


def rhythm_burden(segments, n_samples, fs, rhythm="AFIB"):
    """參考端 AF 負擔（依節律標註的時間計，與 R 峰偵測無關）。回傳 (秒數, 比例)。"""
    total = 0.0
    for i, (s, r) in enumerate(segments):
        e = segments[i + 1][0] if i + 1 < len(segments) else n_samples
        if r == rhythm:
            total += (e - s) / fs
    return total, total / (n_samples / fs)


def af_windows(peaks, fs, window=32, step=16, rr_clip=(0.2, 2.0)):
    """R 峰 → RR（秒）→ 滑動窗。回傳 (X (n_win, window), starts (n_win,), rr 未裁切 (n_rr,))。

    窗 k 涵蓋 RR 索引 [starts[k], starts[k] + window)，與 afdb 訓練期的切法相同。
    """
    rr = np.diff(np.asarray(peaks, dtype=np.float64)) / float(fs)
    rrc = np.clip(rr, rr_clip[0], rr_clip[1]) if rr_clip is not None else rr
    if len(rr) < window:
        return np.zeros((0, window), np.float32), np.zeros(0, np.int64), rr
    starts = np.arange(0, len(rr) - window + 1, step, dtype=np.int64)
    X = np.stack([rrc[s:s + window] for s in starts]).astype(np.float32)
    return X, starts, rr


@torch.no_grad()
def classify_af_windows(X, bundle, device="cpu", batch_size=4096):
    """RR 窗 → AF 機率（fold ensemble 平均）與各模型的機率 (M, n_win)。"""
    if len(X) == 0:
        return np.zeros(0, np.float32), np.zeros((len(bundle["models"]), 0), np.float32)
    x = torch.from_numpy(np.asarray(X, dtype=np.float32)).unsqueeze(1)
    per_model = []
    for m in bundle["models"]:
        ps = []
        for i in range(0, len(x), batch_size):
            ps.append(torch.softmax(m(x[i:i + batch_size].to(device)), dim=1)[:, 1].cpu().numpy())
        per_model.append(np.concatenate(ps))
    per_model = np.stack(per_model).astype(np.float32)
    return per_model.mean(0), per_model


def rr_level_af(n_rr, starts, window, win_prob, threshold=0.5):
    """窗機率 → 每個 RR 的 AF 機率（覆蓋它的窗取平均）與標籤。尾端沒被覆蓋的 RR 取最後一窗。"""
    acc = np.zeros(n_rr, np.float64)
    cnt = np.zeros(n_rr, np.float64)
    for s, p in zip(starts, win_prob):
        acc[s:s + window] += p
        cnt[s:s + window] += 1
    if len(starts):
        last = int(starts[-1]) + window
        acc[last:] = win_prob[-1]
        cnt[last:] = 1
    prob = np.divide(acc, cnt, out=np.zeros_like(acc), where=cnt > 0)
    return prob, (prob >= threshold).astype(np.int64)


def af_episodes(peaks, fs, rr_af, min_dur_s=AF_MIN_EPISODE_S):
    """每個 RR 的 AF 標籤 → 連續 AF 段 [(start_sample, end_sample, 秒數)]，只留 ≥ min_dur_s。"""
    eps = []
    peaks = np.asarray(peaks)
    i, n = 0, len(rr_af)
    while i < n:
        if rr_af[i] == 1:
            j = i
            while j + 1 < n and rr_af[j + 1] == 1:
                j += 1
            s, e = int(peaks[i]), int(peaks[j + 1])
            dur = (e - s) / fs
            if dur >= min_dur_s:
                eps.append((s, e, dur))
            i = j + 1
        else:
            i += 1
    return eps


def suppress_s_in_af(beat_pred, rr_af):
    """AF 期間心房沒有規律電活動，PAC（S）無從定義 → AF 段內被判為 S 的拍改回 N。

    第 i 拍以「結束於它的 RR 區間」rr_af[i-1] 判定是否在 AF 內（第 0 拍用 rr_af[0]）。V 不受影響。
    這條規則是看到 DS2 結果後才加的（事後改良），可行性判定仍以協定鎖定的原始系統為準。
    """
    pred = np.asarray(beat_pred).copy()
    if rr_af is None or len(rr_af) == 0:
        return pred
    in_af = np.concatenate([[rr_af[0]], rr_af]).astype(bool)[:len(pred)]
    pred[in_af & (pred == CLS_S)] = CLS_N
    return pred


# ---------------------------------------------------------------------------
# 整筆紀錄摘要
# ---------------------------------------------------------------------------
def runs_of(labels, cls, min_len):
    """連續 cls 的段落（長度 ≥ min_len）→ [(start_idx, length)]。"""
    out, i, n = [], 0, len(labels)
    while i < n:
        if labels[i] == cls:
            j = i
            while j + 1 < n and labels[j + 1] == cls:
                j += 1
            if j - i + 1 >= min_len:
                out.append((i, j - i + 1))
            i = j + 1
        else:
            i += 1
    return out


def summarize(peaks, fs, beat_pred, n_samples, rr_af=None):
    """Holter 摘要：心率、各類拍數與負擔、V/S 連發、AF 負擔與發作。"""
    peaks = np.asarray(peaks)
    beat_pred = np.asarray(beat_pred)
    n = len(peaks)
    dur_s = n_samples / fs
    rr = np.diff(peaks) / fs
    good = (rr > 0.2) & (rr < 3.0)
    hr_bins = []
    if n > 1:
        t = peaks[1:] / fs
        for b in range(int(np.ceil(dur_s / 60.0))):
            m = good & (t >= b * 60) & (t < (b + 1) * 60)
            if m.sum() >= 10:
                hr_bins.append(60.0 / rr[m].mean())
    counts = {name: int((beat_pred == c).sum()) for c, name in enumerate(CLASS_NAMES)}
    v_runs = runs_of(beat_pred, CLS_V, 3)
    s_runs = runs_of(beat_pred, CLS_S, 3)
    out = {
        "duration_s": dur_s,
        "n_beats": int(n),
        "hr_mean": float(60.0 / rr[good].mean()) if good.any() else float("nan"),
        "hr_min_1min": float(min(hr_bins)) if hr_bins else float("nan"),
        "hr_max_1min": float(max(hr_bins)) if hr_bins else float("nan"),
        "counts": counts,
        "pvc_burden": counts["V"] / n if n else 0.0,
        "pac_burden": counts["S"] / n if n else 0.0,
        "v_couplets": len(runs_of(beat_pred, CLS_V, 2)) - len(v_runs),
        "v_runs": len(v_runs),
        "v_run_longest": max((L for _, L in v_runs), default=0),
        "s_runs": len(s_runs),
        "hourly": _hourly(peaks, fs, beat_pred, dur_s),
    }
    if rr_af is not None and len(rr):
        af_time = float(rr[rr_af == 1].sum())
        out["af_burden"] = af_time / float(rr.sum())
        out["af_time_s"] = af_time
        out["af_episodes"] = af_episodes(peaks, fs, rr_af)
    return out


def _hourly(peaks, fs, beat_pred, dur_s):
    """每小時（紀錄短於 2 小時則每 5 分鐘）的心跳數、PVC、PAC 數，給報告趨勢圖。"""
    bin_s = 3600.0 if dur_s >= 7200 else 300.0
    t = np.asarray(peaks) / fs
    rows = []
    for b in range(int(np.ceil(dur_s / bin_s))):
        m = (t >= b * bin_s) & (t < (b + 1) * bin_s)
        rows.append({"start_s": b * bin_s, "beats": int(m.sum()),
                     "V": int((beat_pred[m] == CLS_V).sum()), "S": int((beat_pred[m] == CLS_S).sum())})
    return {"bin_s": bin_s, "rows": rows}


def analyze_record(signal, fs, beat_bundle, af_bundle=None, device="cpu", pac_rule=True):
    """一筆原始 ECG 的完整分析（部署版本）。回傳 dict（含各步驟耗時，供 F8 速度評估）。

    pac_rule=True：AF 段內不計 PAC（suppress_s_in_af）；beat_pred_raw 保留規則前的逐拍結果。
    """
    t = {}
    t0 = time.perf_counter()
    sig, fs_m = to_model_fs(signal, fs)
    t["resample"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    q = beat_bundle["qrs"]
    peaks = detect_r_peaks(sig, fs_m, method=q["method"], correct=q["correct"])
    t["detect"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    beats = classify_beats(sig, fs_m, peaks, beat_bundle, device=device, pad_edges=True)
    t["beat"] = time.perf_counter() - t0

    rr_af = win = None
    t0 = time.perf_counter()
    if af_bundle is not None and len(peaks) > 1:
        X, starts, rr = af_windows(peaks, fs_m, af_bundle["window"], af_bundle["step"], tuple(af_bundle["rr_clip"]))
        win_prob, _ = classify_af_windows(X, af_bundle, device=device)
        _, rr_af = rr_level_af(len(rr), starts, af_bundle["window"], win_prob, af_bundle["af_threshold"])
        win = {"starts": starts, "prob": win_prob, "window": af_bundle["window"]}
    t["af"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    beat_pred = suppress_s_in_af(beats["pred"], rr_af) if pac_rule and rr_af is not None else beats["pred"]
    summary = summarize(peaks, fs_m, beat_pred, len(sig), rr_af=rr_af)
    summary["pac_rule"] = bool(pac_rule and rr_af is not None)
    t["summary"] = time.perf_counter() - t0
    t["total"] = sum(t.values())
    return {"signal": sig, "fs": fs_m, "peaks": peaks, "beat_pred": beat_pred, "beat_pred_raw": beats["pred"],
            "beat_probs": beats["probs"], "rr_af": rr_af, "af_windows": win,
            "summary": summary, "timing": t}
