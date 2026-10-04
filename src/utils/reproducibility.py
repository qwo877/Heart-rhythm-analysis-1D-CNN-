"""設定載入、亂數種子、輸出目錄與 config 快照（mitdb / afdb 共用）。"""
import datetime
import os
import random
import shutil

import numpy as np
import torch
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed, deterministic=True):
    """把所有亂數來源釘死，讓實驗可重現。

    只設 cudnn.deterministic 還不夠：GPU 上 cuBLAS matmul 與部分 backward kernel
    仍非確定性，會讓相同 seed 的兩次 run 從第 2 個 epoch 起發散。因此再開
    torch.use_deterministic_algorithms 並設 CUBLAS_WORKSPACE_CONFIG（必須在第一個
    CUDA 呼叫前設好；set_seed 於程式最開頭呼叫，符合此要求）。
    warn_only=True：少數沒有確定性實作的 op（如某些 pooling backward）只警告不中斷，
    換取「幾乎逐位元可重現」而非硬崩潰。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(pref="auto"):
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_output_dir(root, tag="phase0"):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(root, f"{ts}_{tag}")
    os.makedirs(out, exist_ok=True)
    return out


def snapshot_config(cfg_path, output_dir):
    """複製當次使用的 config，確保 run 與設定綁在一起。"""
    dst = os.path.join(output_dir, "config.snapshot.yaml")
    shutil.copy(cfg_path, dst)
    return dst
