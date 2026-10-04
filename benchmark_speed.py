#處理速度（README 可行性判準 F8

import argparse
import json
import os
import platform
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import torch

from src.holter import analyze_record, load_af_bundle, load_beat_bundle
from src.preprocess.wfdb_io import read_signal
from src.utils import load_config, make_output_dir


def cpu_name():
    try:
        import winreg  # Windows：registry 裡的處理器名稱比 platform.processor() 可讀
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:  # noqa: BLE001
        return platform.processor() or platform.machine()


def main():
    ap = argparse.ArgumentParser(description="處理速度（F8）")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--beat-bundle", default=os.path.join("models", "holter_beat.pt"))
    ap.add_argument("--af-bundle", default=os.path.join("models", "holter_af.pt"))
    args, _ = ap.parse_known_args()

    cfg = load_config(args.config)
    out_dir = make_output_dir(cfg["output"]["root"], tag="benchmark_speed")
    data_dir = cfg["data"]["data_dir"]
    device = torch.device("cpu")
    beat_b = load_beat_bundle(args.beat_bundle, device=device)
    af_b = load_af_bundle(args.af_bundle, device=device)

    records = sorted(os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith(".hea"))
    t0 = time.perf_counter()
    parts, fs = [], None
    for rec in records:
        sig, fs_r = read_signal(rec, data_dir=data_dir, channel=cfg["data"].get("channel", "MLII"))
        assert fs is None or fs_r == fs
        fs = fs_r
        parts.append(sig)
    signal = np.concatenate(parts)
    t_load = time.perf_counter() - t0
    hours = len(signal) / fs / 3600
    print(f"串接 {len(records)} 筆 → {hours:.2f} 小時（{len(signal):,} 點 @ {fs} Hz），讀檔 {t_load:.1f}s")

    res = analyze_record(signal, fs, beat_b, af_b, device=device)
    t = res["timing"]
    total = t["total"] + t_load
    per24 = total / hours * 24
    lines = [
        f"處理速度（CPU：{cpu_name()}；torch threads={torch.get_num_threads()}；不用 GPU）", "",
        f"輸入：MIT-BIH {len(records)} 筆串接 = {hours:.2f} 小時單導程 ECG（{len(signal):,} 點）",
        f"偵測到 {len(res['peaks']):,} 拍", "",
        f"  讀檔        {t_load:8.1f} s",
        f"  重取樣      {t['resample']:8.1f} s",
        f"  R 峰偵測    {t['detect']:8.1f} s",
        f"  逐拍分類    {t['beat']:8.1f} s   （{len(beat_b['models'])} 模型 ensemble）",
        f"  AF          {t['af']:8.1f} s   （{len(af_b['models'])} 模型 ensemble）",
        f"  摘要        {t['summary']:8.1f} s",
        f"  合計        {total:8.1f} s  = {total / 60:.1f} 分鐘（換算 24 小時：{per24 / 60:.1f} 分鐘）", "",
        "F8 判定（≤10 分鐘可行、≤60 分鐘有條件可行）：" +
        ("可行" if per24 <= 600 else ("有條件可行" if per24 <= 3600 else "不可行")),
    ]
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(out_dir, "speed.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(os.path.join(out_dir, "speed.json"), "w", encoding="utf-8") as f:
        json.dump({"cpu": cpu_name(), "threads": torch.get_num_threads(), "hours": hours,
                   "n_beats": int(len(res["peaks"])), "timing_s": {**t, "load": t_load, "total_with_load": total},
                   "minutes_per_24h": per24 / 60}, f, ensure_ascii=False, indent=2)
    print(f"\n輸出目錄：{out_dir}")


if __name__ == "__main__":
    main()
