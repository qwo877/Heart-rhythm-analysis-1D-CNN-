"""mitdb 單拍載入器。除了單拍 morphology 窗，也算每個 beat 的 RR 特徵（秒），
供 Phase 1.2/1.5 的多尺度 context 分支使用。single/multi ablation 共用同一組 beat。
"""
from collections import Counter

import numpy as np

from ..preprocess.filters import apply_signal_filters
from ..preprocess.rhythm import beat_rr_features, beat_rr_sequences
from ..preprocess.symbol_mapping import CLASS_NAMES, build_symbol_map
from ..preprocess.wfdb_io import ensure_downloaded, read_annotation, read_signal
from ..splits.aami_split import make_mitdb_split


def load_beats(
    record_names,
    data_dir,
    window=256,
    channel="MLII",
    exclude_records=(102, 104, 107, 217),
    sym_map=None,
    rr_local_window=10,
    rr_clip=(0.2, 2.0),
    rhythm_seq_len=32,
    filter_cfg=None,
    verbose=True,
):
    """回傳 dict：X(N,window), y(N,), rr(N,4), rr_seq(N,rhythm_seq_len), groups(N,)。

    - RR 特徵/序列在「每筆紀錄內」用連續 beat 的 R 峰算，避免跨紀錄污染。
    - filter_cfg 若開啟，對整筆原始訊號套 bandpass/notch 後才切窗（可開關）。
    - rr_seq 是每個 beat 的中心化局部 RR 序列，格式同 afdb RR 窗，給 1.4 共享 rhythm encoder。
    """
    sym_map = sym_map or build_symbol_map()
    exclude = {str(r) for r in exclude_records}
    half = window // 2

    X, y, rr_all, rr_seq_all, groups = [], [], [], [], []
    skipped_symbols = Counter()

    for rec_name in record_names:
        rec_name = str(rec_name)
        if rec_name in exclude:
            if verbose:
                print(f"  跳過起搏紀錄 {rec_name}")
            continue

        try:
            signal, fs = read_signal(rec_name, data_dir=data_dir, channel=channel)
            ann = read_annotation(rec_name, "atr", data_dir=data_dir)
        except Exception as e:  # noqa: BLE001
            print(f"  讀取 {rec_name} 失敗：{e}")
            continue

        # 1.5 訊號前處理（可開關）：整筆濾波後再切窗，避免濾波邊界效應落在每個 beat 上
        signal = apply_signal_filters(signal, fs, filter_cfg)

        # 先收集這筆紀錄內所有 beat（R 峰 sample + 類別）
        beat_samples, beat_cls = [], []
        for samp, sym in zip(ann.sample, ann.symbol):
            if isinstance(sym, bytes):
                sym = sym.decode("utf-8", "ignore")
            cls = sym_map.get(sym)
            if cls is None:
                skipped_symbols[sym] += 1
                continue
            beat_samples.append(int(samp))
            beat_cls.append(cls)

        if not beat_samples:
            continue
        rr_feats = beat_rr_features(beat_samples, fs, local_window=rr_local_window, rr_clip=rr_clip)
        rr_seqs = beat_rr_sequences(beat_samples, fs, seq_len=rhythm_seq_len, rr_clip=rr_clip)

        n = len(signal)
        for i, (samp, cls) in enumerate(zip(beat_samples, beat_cls)):
            start = samp - half
            end = start + window
            if start < 0 or end > n:
                continue
            X.append(signal[start:end])
            y.append(cls)
            rr_all.append(rr_feats[i])
            rr_seq_all.append(rr_seqs[i])
            groups.append(rec_name)

    if not X:
        raise RuntimeError("沒有抽到任何 beat，檢查 data_dir / record list / symbol map。")

    out = {
        "X": np.stack(X).astype(np.float32),
        "y": np.asarray(y, dtype=np.int64),
        "rr": np.stack(rr_all).astype(np.float32),
        "rr_seq": np.stack(rr_seq_all).astype(np.float32),
        "groups": np.asarray(groups),
    }
    if verbose:
        dist = {CLASS_NAMES[k]: v for k, v in sorted(Counter(out["y"]).items())}
        print(f"  抽出 {len(out['X'])} beats；分布 {dist}")
        if skipped_symbols:
            print(f"  跳過非 beat 標記：{dict(skipped_symbols)}")
    return out


def load_beat_splits(cfg, verbose=True):
    """依 config 載入 train/val/test 三個「完整 dict」（含 rr_seq/groups），並驗證病人不重疊。

    beat 任務（Phase 0/1.2）與 joint 任務（Phase 1.4）共用這裡；差別只在下游取用哪些欄位。
    """
    d = cfg["data"]
    data_dir = d["data_dir"]
    ensure_downloaded("mitdb", data_dir, d.get("download_if_missing", True))

    train_records, val_records, test_records = make_mitdb_split(d)
    common = dict(
        data_dir=data_dir,
        window=d["window"],
        channel=d.get("channel", "MLII"),
        exclude_records=d.get("exclude_records", [102, 104, 107, 217]),
        rr_local_window=cfg.get("model", {}).get("rr_local_window", 10),
        rr_clip=tuple(d["rr_clip"]) if d.get("rr_clip") else (0.2, 2.0),
        rhythm_seq_len=cfg.get("afdb", {}).get("window", 32),  # 與 afdb 窗長一致 → 共享 rhythm encoder
        filter_cfg=d.get("filter"),
        verbose=verbose,
    )

    if verbose:
        print(f"[train] {len(train_records)} records: {train_records}")
    tr = load_beats(train_records, **common)
    if verbose:
        print(f"[val]   {len(val_records)} records: {val_records}")
    va = load_beats(val_records, **common)
    if verbose:
        print(f"[test]  {len(test_records)} records: {test_records}")
    te = load_beats(test_records, **common)

    assert not (set(tr["groups"]) & set(va["groups"])), "train/val 病人重疊！"
    assert not (set(tr["groups"]) & set(te["groups"])), "train/test 病人重疊！"
    assert not (set(va["groups"]) & set(te["groups"])), "val/test 病人重疊！"
    return tr, va, te


def prepare_beat_data(cfg, verbose=True):
    """依 config 產生 train/val/test，每組為 (X, y, rr)（Phase 0/1.2 用，維持舊介面）。"""
    tr, va, te = load_beat_splits(cfg, verbose=verbose)
    return (
        (tr["X"], tr["y"], tr["rr"]),
        (va["X"], va["y"], va["rr"]),
        (te["X"], te["y"], te["rr"]),
    )
