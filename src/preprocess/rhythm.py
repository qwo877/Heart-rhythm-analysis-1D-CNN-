"""afdb 節律段解析 + RR 序列抽取。

依實際探查到的 afdb 結構寫（非假設）：
  - .atr：symbol 一律 '+'，節律寫在 aux_note，如 '(N' / '(AFIB' / '(AFL' / '(J'，
          每筆 sample 標示一段節律的「起點」，段內標籤不變、直到下一個起點。
  - .qrs：R 峰位置（beat），symbol 'N'，無 aux_note。
坑2：afdb 250Hz、mitdb 360Hz，RR 一律換成「秒」(sample ÷ fs) 才能跨資料集比較。
"""
import numpy as np

# afdb 節律 aux_note（去掉前導 '('）。標準 AF 偵測慣例：只有 AFIB 算 AF 陽性，
# AFL(心房撲動) / J(交界性) / N(正常) 皆為 non-AF。可由 af_rhythms 覆寫。
AF_RHYTHMS_DEFAULT = frozenset({"AFIB"})


def parse_rhythm_segments(atr_ann):
    """把 .atr 轉成 [(start_sample, rhythm_str), ...]，依 sample 排序。"""
    segs = []
    aux_notes = atr_ann.aux_note if atr_ann.aux_note is not None else []
    for samp, aux in zip(atr_ann.sample, aux_notes):
        if not aux:
            continue
        rhythm = aux.lstrip("(").strip().strip("\x00")
        if rhythm:
            segs.append((int(samp), rhythm))
    segs.sort(key=lambda s: s[0])
    return segs


def label_samples_by_rhythm(samples, segments, af_rhythms=AF_RHYTHMS_DEFAULT):
    """給一組 sample 位置，回傳每個位置的 AF 標籤(1/0)。

    以「該 sample 落在哪一段節律」判定；用 searchsorted 找 <= sample 的最後一個段起點。
    """
    if not segments:
        return np.zeros(len(samples), dtype=np.int64)
    starts = np.asarray([s for s, _ in segments])
    is_af = np.asarray([1 if r in af_rhythms else 0 for _, r in segments], dtype=np.int64)
    idx = np.searchsorted(starts, np.asarray(samples), side="right") - 1
    idx = np.clip(idx, 0, len(segments) - 1)
    return is_af[idx]


def rr_seconds(r_samples, fs):
    """R 峰 sample 位置 → RR 間期（秒）。長度 = len(r_samples) - 1。"""
    r_samples = np.asarray(r_samples, dtype=np.float64)
    return np.diff(r_samples) / float(fs)


def rhythm_distribution(atr_ann):
    """統計一筆紀錄出現過哪些節律（除錯 / 資料檢查用）。"""
    from collections import Counter

    return Counter(r for _, r in parse_rhythm_segments(atr_ann))


# 每個 RR 特徵的名稱（給多尺度 beat 分支的 context）
RR_FEATURE_NAMES = ["pre_RR", "post_RR", "RR_ratio", "local_avg_RR"]


# 手工 HRV 特徵名稱（給 rhythm 分支 late fusion）
HRV_FEATURE_NAMES = ["SDNN", "RMSSD", "pNN50"]


def hrv_features(rr_window, pnn_thresh=0.05):
    """由一段 RR 序列（秒）算三個標準 HRV 時域指標，供 rhythm 分支 late fusion。

    - SDNN ：RR 標準差（整體變異度）。
    - RMSSD：相鄰 RR 差的均方根（短期變異度，AF 特別高）。
    - pNN50：相鄰 RR 差 > 50ms 的比例（AF 因不規則而偏高）。
    RR 已是秒，故 50ms = 0.05s。回傳 shape (3,) float32。
    """
    rr = np.asarray(rr_window, dtype=np.float64)
    if len(rr) < 2:
        return np.zeros(3, dtype=np.float32)
    diff = np.diff(rr)
    sdnn = float(np.std(rr))
    rmssd = float(np.sqrt(np.mean(diff ** 2)))
    pnn50 = float(np.mean(np.abs(diff) > pnn_thresh))
    return np.array([sdnn, rmssd, pnn50], dtype=np.float32)


def beat_rr_features(r_samples, fs, local_window=10, rr_clip=(0.2, 2.0)):
    """由「一筆紀錄內」連續 beat 的 R 峰位置，算每個 beat 的 RR 特徵（秒）。

    回傳 (n_beats, 4)：pre_RR, post_RR, RR_ratio(pre/post), local_avg_RR。
    邊界 beat 的缺項用鄰近值補（不丟資料，讓 single/multi ablation 用同一組 beat）。
    RR 一律換算成秒（sample ÷ fs），跨 mitdb/afdb 才可比（坑2）。
    rr_clip：把 RR 裁到生理範圍，擋掉標註斷點造成的極端值（如 100s），否則會毀掉標準化。
    """
    r = np.asarray(r_samples, dtype=np.float64)
    n = len(r)
    if n == 0:
        return np.zeros((0, 4), dtype=np.float32)
    if n == 1:
        return np.zeros((1, 4), dtype=np.float32)

    rr = np.diff(r) / float(fs)  # 長度 n-1，rr[i] = beat i → i+1 的間期
    if rr_clip is not None:
        rr = np.clip(rr, rr_clip[0], rr_clip[1])
    pre = np.empty(n)
    post = np.empty(n)
    pre[1:] = rr
    pre[0] = rr[0]
    post[:-1] = rr
    post[-1] = rr[-1]
    ratio = pre / (post + 1e-8)

    local = np.empty(n)
    for i in range(n):
        lo = max(0, i - local_window)
        hi = min(len(rr), i + local_window)
        seg = rr[lo:hi]
        local[i] = seg.mean() if len(seg) > 0 else post[i]

    return np.stack([pre, post, ratio, local], axis=1).astype(np.float32)


def beat_rr_sequences(r_samples, fs, seq_len=32, rr_clip=(0.2, 2.0)):
    """每個 beat 的「中心化局部 RR 序列」（秒），長度固定 seq_len。

    給 Phase 1.4 joint model 的**共享 rhythm encoder** 用：格式與 afdb 的 RR 窗一致
    （皆為秒、可 clip），所以同一個 encoder 能同時吃 afdb 的 AF 窗與 mitdb 每個 beat 的
    節律 context。邊界 beat 以 edge padding 補足，長度恆為 seq_len。
    回傳 (n_beats, seq_len) float32。
    """
    r = np.asarray(r_samples, dtype=np.float64)
    n = len(r)
    if n < 2:
        return np.zeros((n, seq_len), dtype=np.float32)
    rr = np.diff(r) / float(fs)           # 長度 n-1
    if rr_clip is not None:
        rr = np.clip(rr, rr_clip[0], rr_clip[1])
    half = seq_len // 2
    padded = np.pad(rr, (half, seq_len - half), mode="edge")  # 長度 (n-1)+seq_len
    out = np.empty((n, seq_len), dtype=np.float32)
    for i in range(n):
        c = i if i < len(rr) else len(rr) - 1   # beat i 對應的 RR 中心索引
        out[i] = padded[c:c + seq_len]
    return out
