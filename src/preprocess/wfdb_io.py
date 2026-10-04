"""wfdb 讀取的共用薄層（mitdb 用本機檔案、afdb 用 PhysioNet 串流）。

兩個資料集共通的重複部分：下載、讀 record 訊號（含選導程）、讀 annotation。
支援兩種來源：
  - data_dir=... → 讀本機已下載的檔案（mitdb Phase 0 流程）
  - pn_dir=...   → 直接從 PhysioNet 串流（afdb 只需 annotation，不必下載數 GB 訊號）
"""
import os

import numpy as np
import wfdb


def ensure_downloaded(db_name, data_dir, download_if_missing=True):
    """確保本機資料夾有該資料庫的訊號檔（給需要完整訊號的 mitdb 用）。"""
    has_data = os.path.isdir(data_dir) and any(
        f.endswith(".dat") for f in os.listdir(data_dir)
    )
    if has_data:
        return
    if not download_if_missing:
        raise FileNotFoundError(
            f"找不到 {db_name} 資料於 {data_dir}，且 download_if_missing=False。"
        )
    os.makedirs(data_dir, exist_ok=True)
    print(f"下載 {db_name} 到 {data_dir} …（首次執行需數分鐘）")
    wfdb.dl_database(db_name, data_dir)
    print("下載完成。")


def _locate(rec_name, data_dir=None, pn_dir=None):
    """回傳 (record_path, kwargs) 供 wfdb 讀取，兼容本機 / 串流兩種來源。"""
    rec_name = str(rec_name)
    if pn_dir is not None:
        return rec_name, {"pn_dir": pn_dir}
    return os.path.join(data_dir, rec_name), {}


def read_signal(rec_name, data_dir=None, pn_dir=None, channel="MLII"):
    """讀單導程訊號。回傳 (signal float32 1D, fs)。找不到指定導程就退回第 0 個。"""
    path, kw = _locate(rec_name, data_dir, pn_dir)
    rec = wfdb.rdrecord(path, **kw)
    try:
        ch = rec.sig_name.index(channel)
    except (ValueError, AttributeError):
        ch = 0
    signal = np.asarray(rec.p_signal[:, ch], dtype=np.float32).flatten()
    return signal, rec.fs


def read_annotation(rec_name, ext, data_dir=None, pn_dir=None):
    """讀 annotation（ext 例：'atr'、'qrs'）。"""
    path, kw = _locate(rec_name, data_dir, pn_dir)
    return wfdb.rdann(path, ext, **kw)


def read_header_fs(rec_name, data_dir=None, pn_dir=None):
    path, kw = _locate(rec_name, data_dir, pn_dir)
    return wfdb.rdheader(path, **kw).fs
