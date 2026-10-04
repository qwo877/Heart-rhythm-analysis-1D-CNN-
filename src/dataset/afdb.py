"""afdb 節律 / AF 載入器（RR 序列 → AF / non-AF）。

只需 annotation（.qrs 抓 R 峰、.atr 抓節律段），預設直接從 PhysioNet 串流，
不必下載數 GB 的 10 小時原始訊號。RR 一律以「秒」表示（坑2）。

AF 標籤流程：
  1. .qrs → R 峰 sample → RR(秒)
  2. .atr → 節律段；每個 RR 用其結束 R 峰所在的節律段標成 AF(1)/non-AF(0)
  3. 滑動窗（W 個 RR，步長 step）；窗內 AF 比例 ≥ threshold → 該窗標 AF
以「整筆 record（病人）」為單位切 train/val/test（AF 是 subject-level 的病）。
"""
import numpy as np

from ..preprocess.rhythm import (
    AF_RHYTHMS_DEFAULT,
    hrv_features,
    label_samples_by_rhythm,
    parse_rhythm_segments,
    rr_seconds,
)
from ..preprocess.wfdb_io import read_annotation, read_header_fs
from ..splits.afdb_split import make_afdb_split


def _load_record_rr(rec_name, pn_dir="afdb", data_dir=None, af_rhythms=AF_RHYTHMS_DEFAULT):
    """回傳單筆的 (rr_秒 array, per-RR AF 標籤 array, fs)。"""
    src = dict(pn_dir=pn_dir) if data_dir is None else dict(data_dir=data_dir)
    fs = read_header_fs(rec_name, **src)
    qrs = read_annotation(rec_name, "qrs", **src)
    atr = read_annotation(rec_name, "atr", **src)

    r_samples = np.asarray(qrs.sample)
    if len(r_samples) < 2:
        return np.empty(0), np.empty(0, dtype=np.int64), fs

    rr = rr_seconds(r_samples, fs)
    segments = parse_rhythm_segments(atr)
    # 每個 RR 用「結束的 R 峰」所在節律段標籤
    rr_af = label_samples_by_rhythm(r_samples[1:], segments, af_rhythms=af_rhythms)
    return rr.astype(np.float32), rr_af.astype(np.int64), fs


def load_af_windows(
    record_names,
    window=32,
    step=16,
    af_threshold=0.5,
    af_rhythms=AF_RHYTHMS_DEFAULT,
    rr_clip=(0.2, 2.0),
    pn_dir="afdb",
    data_dir=None,
    verbose=True,
):
    """把多筆 record 切成 RR 窗。回傳 dict：X(N,window), y(N,), hrv(N,3), groups(N,)。

    hrv 是每個窗的 SDNN/RMSSD/pNN50，供 rhythm 分支 late fusion。
    """
    X, y, hrv, groups = [], [], [], []
    for rec_name in record_names:
        rec_name = str(rec_name)
        try:
            rr, rr_af, _fs = _load_record_rr(
                rec_name, pn_dir=pn_dir, data_dir=data_dir, af_rhythms=af_rhythms
            )
        except Exception as e:  # noqa: BLE001
            print(f"  讀取 afdb {rec_name} 失敗：{e}")
            continue
        if len(rr) < window:
            if verbose:
                print(f"  {rec_name}: RR 太短（{len(rr)}）跳過")
            continue
        if rr_clip is not None:
            rr = np.clip(rr, rr_clip[0], rr_clip[1])

        n_af = 0
        for start in range(0, len(rr) - window + 1, step):
            win = rr[start:start + window]
            lab = int(rr_af[start:start + window].mean() >= af_threshold)
            X.append(win)
            y.append(lab)
            hrv.append(hrv_features(win))
            groups.append(rec_name)
            n_af += lab
        if verbose:
            n_win = (len(rr) - window) // step + 1
            print(f"  {rec_name}: {n_win} 窗，AF {n_af} / non-AF {n_win - n_af}")

    if not X:
        raise RuntimeError("afdb 沒有抽到任何窗，檢查 record list / window 設定。")

    return {
        "X": np.stack(X).astype(np.float32),
        "y": np.asarray(y, dtype=np.int64),
        "hrv": np.stack(hrv).astype(np.float32),
        "groups": np.asarray(groups),
    }


def prepare_af_data(cfg, verbose=True):
    """依 config 產生 afdb train / val / test 三個 dict，並驗證病人不重疊。"""
    a = cfg["afdb"]
    train_records, val_records, test_records = make_afdb_split(a)
    common = dict(
        window=a.get("window", 32),
        step=a.get("step", 16),
        af_threshold=a.get("af_threshold", 0.5),
        af_rhythms=frozenset(a.get("af_rhythms", ["AFIB"])),
        rr_clip=tuple(a["rr_clip"]) if a.get("rr_clip") else None,
        pn_dir=a.get("pn_dir", "afdb"),
        data_dir=a.get("data_dir"),
        verbose=verbose,
    )

    if verbose:
        print(f"[af-train] {len(train_records)} records: {train_records}")
    tr = load_af_windows(train_records, **common)
    if verbose:
        print(f"[af-val]   {len(val_records)} records: {val_records}")
    va = load_af_windows(val_records, **common)
    if verbose:
        print(f"[af-test]  {len(test_records)} records: {test_records}")
    te = load_af_windows(test_records, **common)

    assert not (set(tr["groups"]) & set(va["groups"])), "afdb train/val 病人重疊！"
    assert not (set(tr["groups"]) & set(te["groups"])), "afdb train/test 病人重疊！"
    assert not (set(va["groups"]) & set(te["groups"])), "afdb val/test 病人重疊！"

    if verbose:
        for name, dd in [("train", tr), ("val", va), ("test", te)]:
            yy = dd["y"]
            n_af = int(yy.sum())
            print(f"  [{name}] 窗數={len(yy)}  AF={n_af} ({n_af / len(yy):.1%})  non-AF={len(yy) - n_af}")

    return tr, va, te
