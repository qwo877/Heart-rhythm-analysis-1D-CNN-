"""自動 R 峰偵測與逐拍比對（離線 Holter 管線的第一步）。

真實的 Holter 紀錄沒有醫師標註的 R 峰位置；Phase 0–2 的單拍窗與 RR 特徵都取自 `.atr` 標註，
等於假設「R 峰完美」。這裡改成從原始訊號自動偵測，後面的分類才代表真實使用條件。

- detect_r_peaks：wfdb 內建的 GQRS / XQRS；可選 correct_peaks 把位置對齊到 QRS 主峰
  （接近 MIT-BIH 標註的 fiducial point，模型的單拍窗是以它為中心訓練的）。
- match_beats：±tol 容差的一對一比對（ANSI/AAMI EC57 慣例 150 ms），供偵測與分類的端到端評估。
"""
import numpy as np
from wfdb import processing

QRS_METHODS = ("gqrs", "xqrs")


def detect_r_peaks(signal, fs, method="gqrs", correct=True, **refine_kw):
    """單導程訊號 → R 峰 sample 位置（int64、遞增、無重複）。= run_detector + refine_peaks。"""
    sig = np.asarray(signal, dtype=np.float64)
    return refine_peaks(sig, fs, run_detector(sig, fs, method), correct=correct, **refine_kw)


def run_detector(signal, fs, method="gqrs"):
    """原始偵測器輸出（未校正）。"""
    sig = np.asarray(signal, dtype=np.float64)
    if method == "gqrs":
        peaks = processing.gqrs_detect(sig=sig, fs=fs)
    elif method == "xqrs":
        peaks = processing.xqrs_detect(sig=sig, fs=fs, verbose=False)
    else:
        raise ValueError(f"未知 QRS 偵測方法：{method}（可用 {QRS_METHODS}）")
    return np.asarray(peaks, dtype=np.int64)


def refine_peaks(signal, fs, peaks, correct=True, search_ms=50, smooth_ms=20, refractory_ms=200):
    """correct=True：用 wfdb.processing.correct_peaks 在 ±search_ms 內移到局部極值。
    refractory_ms：若兩個峰距離小於不應期（生理上不可能），保留振幅較大者。
    """
    sig = np.asarray(signal, dtype=np.float64)
    peaks = np.asarray(peaks, dtype=np.int64)
    if len(peaks) == 0:
        return peaks
    if correct:
        peaks = processing.correct_peaks(
            sig, peaks,
            search_radius=max(1, int(search_ms / 1000 * fs)),
            smooth_window_size=max(1, int(smooth_ms / 1000 * fs)),
            peak_dir="compare",
        )
        peaks = np.asarray(peaks, dtype=np.int64)
    peaks = np.unique(np.clip(peaks, 0, len(sig) - 1))
    return _enforce_refractory(sig, peaks, int(refractory_ms / 1000 * fs))


def _enforce_refractory(sig, peaks, min_gap):
    """相鄰峰距離 < min_gap 時只留 |振幅|（相對中位數）較大的那個。"""
    if len(peaks) < 2 or min_gap <= 0:
        return peaks
    amp = np.abs(sig[peaks] - np.median(sig))
    keep = [0]
    for i in range(1, len(peaks)):
        last = keep[-1]
        if peaks[i] - peaks[last] < min_gap:
            if amp[i] > amp[last]:
                keep[-1] = i
        else:
            keep.append(i)
    return peaks[np.asarray(keep)]


def match_beats(ref, det, tol):
    """參考 beat 與偵測 beat 的一對一比對（容差 tol samples）。

    依參考拍的時間順序，為每個參考拍挑「容差內、尚未被配對、距離最近」的偵測拍。
    回傳 (ref_to_det, det_to_ref)：長度分別為 len(ref)、len(det) 的 int64 array，未配對為 -1。
    """
    ref = np.asarray(ref, dtype=np.int64)
    det = np.asarray(det, dtype=np.int64)
    ref_to_det = np.full(len(ref), -1, dtype=np.int64)
    det_to_ref = np.full(len(det), -1, dtype=np.int64)
    j = 0
    for i, r in enumerate(ref):
        while j < len(det) and det[j] < r - tol:
            j += 1
        best, best_d = -1, tol + 1
        k = j
        while k < len(det) and det[k] <= r + tol:
            if det_to_ref[k] == -1:
                d = abs(int(det[k]) - int(r))
                if d < best_d:
                    best, best_d = k, d
            k += 1
        if best >= 0:
            ref_to_det[i] = best
            det_to_ref[best] = i
    return ref_to_det, det_to_ref


def detection_stats(ref, det, tol):
    """R 峰偵測的 TP / FN / FP、Se、+P 與配對拍的位置偏移（samples）。"""
    ref_to_det, det_to_ref = match_beats(ref, det, tol)
    tp = int((ref_to_det >= 0).sum())
    fn = int(len(ref) - tp)
    fp = int((det_to_ref < 0).sum())
    matched = ref_to_det >= 0
    offsets = np.asarray(det, dtype=np.int64)[ref_to_det[matched]] - np.asarray(ref, dtype=np.int64)[matched]
    return {
        "tp": tp, "fn": fn, "fp": fp,
        "se": tp / max(tp + fn, 1),
        "ppv": tp / max(tp + fp, 1),
        "offsets": offsets,
    }
