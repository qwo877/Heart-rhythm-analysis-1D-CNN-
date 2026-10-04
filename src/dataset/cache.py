"""資料載入的快取：讀 mitdb 單拍與 afdb RR 窗約需數分鐘，快取成 npz 後秒級復用。

快取檔名用「會影響資料內容的設定」的雜湊當鍵；設定一改就自動重建，不會讀到過期的資料。
    cache/beats_<key>.npz ：mitdb train / val / test 的單拍窗、標籤、RR 特徵、局部 RR 序列、病人編號
    cache/afwin_<key>.npz ：afdb train / val / test 的 RR 窗、AF 標籤、HRV、病人編號
"""
import hashlib
import json
import os

import numpy as np

from .afdb import prepare_af_data
from .mitdb import load_beat_splits


def _beats_key(cfg):
    d = cfg["data"]
    relevant = {
        "window": d["window"], "channel": d.get("channel", "MLII"),
        "exclude": d.get("exclude_records"), "ds1": d.get("ds1"), "ds2": d.get("ds2"),
        "val": d.get("val_records"), "rr_clip": d.get("rr_clip"),
        "rhythm_seq_len": cfg.get("afdb", {}).get("window", 32),
        "filter": d.get("filter"),
    }
    return hashlib.md5(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:12]


def load_beats_cached(cfg, cache_dir="./cache", use_cache=True, verbose=True):
    """load_beat_splits 的快取版本。回傳 train / val / test 三個 dict（X, y, rr, rr_seq, groups）。"""
    path = os.path.join(cache_dir, f"beats_{_beats_key(cfg)}.npz")
    if use_cache and os.path.exists(path):
        if verbose:
            print(f"[cache] 命中 {path}（跳過 mitdb 載入）")
        z = np.load(path, allow_pickle=True)

        def unpack(p):
            return {"X": z[f"{p}_X"], "y": z[f"{p}_y"], "rr": z[f"{p}_rr"],
                    "rr_seq": z[f"{p}_rrseq"], "groups": z[f"{p}_groups"]}
        return unpack("tr"), unpack("va"), unpack("te")

    tr, va, te = load_beat_splits(cfg, verbose=verbose)
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(path, **{f"{p}_{k2}": d[k1] for p, d in [("tr", tr), ("va", va), ("te", te)]
                          for k1, k2 in [("X", "X"), ("y", "y"), ("rr", "rr"),
                                         ("rr_seq", "rrseq"), ("groups", "groups")]})
        if verbose:
            print(f"[cache] 已存 {path}")
    return tr, va, te


def _af_key(cfg):
    a = cfg["afdb"]
    rel = {k: a.get(k) for k in ("window", "step", "af_threshold", "af_rhythms",
                                 "rr_clip", "exclude_records", "val_records", "test_records", "pn_dir")}
    return hashlib.md5(json.dumps(rel, sort_keys=True, default=str).encode()).hexdigest()[:12]


def load_af_cached(cfg, cache_dir="./cache", use_cache=True, verbose=True):
    """prepare_af_data 的快取版本（避免每次從 PhysioNet 串流 annotation）。回傳 train / val / test。"""
    path = os.path.join(cache_dir, f"afwin_{_af_key(cfg)}.npz")
    if use_cache and os.path.exists(path):
        if verbose:
            print(f"[cache] 命中 {path}（跳過 afdb 串流）")
        z = np.load(path, allow_pickle=True)

        def unpack(p):
            return {"X": z[f"{p}_X"], "y": z[f"{p}_y"], "hrv": z[f"{p}_hrv"], "groups": z[f"{p}_groups"]}
        return unpack("tr"), unpack("va"), unpack("te")

    tr, va, te = prepare_af_data(cfg, verbose=verbose)
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(path, **{f"{p}_{k}": d[k] for p, d in [("tr", tr), ("va", va), ("te", te)]
                          for k in ("X", "y", "hrv", "groups")})
        if verbose:
            print(f"[cache] 已存 {path}")
    return tr, va, te
