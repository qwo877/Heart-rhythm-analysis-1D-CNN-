"""ECG 訊號前處理濾波（做成可開關的元件）。

三個獨立可開關的步驟，套在「整筆 record 的原始訊號」上、切窗之前：
  - bandpass  ~0.5–40Hz：同時去 baseline wander（<0.5Hz 漂移）與高頻肌電雜訊。
  - notch     60Hz（美國電網；mitdb 為美國錄製）：去電源線干擾。
  - baseline  單獨的 highpass，只去漂移不動高頻（想保留 QRS 高頻細節時用，與 bandpass 二選一）。

一律用 filtfilt（零相位），避免濾波造成 R 峰位置偏移而影響 morphology 對齊。
所有函式對「找不到 scipy / 訊號過短」都安全退回原訊號，不讓前處理拖垮整條 pipeline。
"""
import numpy as np

try:
    from scipy.signal import butter, filtfilt, iirnotch
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - scipy 缺席時前處理整段跳過
    _HAS_SCIPY = False


def _safe_filtfilt(b, a, x):
    """filtfilt 需要訊號長度 > 3*max(len(a),len(b))；太短就原樣回傳。"""
    padlen = 3 * max(len(a), len(b))
    if len(x) <= padlen:
        return x
    return filtfilt(b, a, x).astype(np.float32)


def bandpass(signal, fs, low=0.5, high=40.0, order=4):
    """Butterworth 帶通（含去 baseline wander）。"""
    if not _HAS_SCIPY:
        return signal
    nyq = 0.5 * fs
    high = min(high, nyq * 0.99)  # 擋掉 high ≥ Nyquist 造成的設計失敗
    if low <= 0 or high <= low:
        return signal
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return _safe_filtfilt(b, a, signal)


def highpass(signal, fs, cutoff=0.5, order=4):
    """只去 baseline wander（<cutoff 的漂移），保留高頻。"""
    if not _HAS_SCIPY:
        return signal
    nyq = 0.5 * fs
    if cutoff <= 0 or cutoff >= nyq:
        return signal
    b, a = butter(order, cutoff / nyq, btype="high")
    return _safe_filtfilt(b, a, signal)


def notch(signal, fs, freq=60.0, q=30.0):
    """去電源線干擾（美國 60Hz / 歐洲 50Hz）。"""
    if not _HAS_SCIPY:
        return signal
    nyq = 0.5 * fs
    if freq <= 0 or freq >= nyq:
        return signal
    b, a = iirnotch(freq / nyq, q)
    return _safe_filtfilt(b, a, signal)


def apply_signal_filters(signal, fs, filter_cfg=None):
    """依 config 對整筆訊號套用可開關的濾波組合（切窗前呼叫）。

    filter_cfg 例：
        {enabled: true, bandpass: true, bp_low: 0.5, bp_high: 40,
         notch: false, notch_freq: 60, baseline_highpass: false, hp_cutoff: 0.5}
    enabled=False 或 None → 原樣回傳（等同關閉前處理，維持與舊 run 可比）。
    """
    if not filter_cfg or not filter_cfg.get("enabled", False):
        return signal
    x = np.asarray(signal, dtype=np.float32)
    if filter_cfg.get("bandpass", True):
        x = bandpass(x, fs, low=filter_cfg.get("bp_low", 0.5),
                     high=filter_cfg.get("bp_high", 40.0),
                     order=filter_cfg.get("bp_order", 4))
    elif filter_cfg.get("baseline_highpass", False):
        x = highpass(x, fs, cutoff=filter_cfg.get("hp_cutoff", 0.5),
                     order=filter_cfg.get("hp_order", 4))
    if filter_cfg.get("notch", False):
        x = notch(x, fs, freq=filter_cfg.get("notch_freq", 60.0),
                  q=filter_cfg.get("notch_q", 30.0))
    return np.asarray(x, dtype=np.float32)
