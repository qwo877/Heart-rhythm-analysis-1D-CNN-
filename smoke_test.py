"""煙霧測試

驗證 dataset / model / engine / metrics / Holter 部署管線串得起來且不報錯（不驗準確率）：
  [beat]       單拍形態 CNN 的訓練與 AAMI 報表
  [af]         RR 窗 AF 模型的訓練與 AF 報表
  [multiscale] 單拍形態 + RR 特徵（部署的逐拍模型）
  [preprocess] 訊號濾波、HRV、局部 RR 序列
  [holter]     自動 R 峰、EC57 比對、端到端混淆、AF／PAC 規則、摘要、完整 analyze_record
    python smoke_test.py
"""
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import torch

from src.dataset.torch_ds import BeatDataset, RhythmDataset
from src.engine import make_loader, train_and_evaluate
from src.metrics import (
    af_f1,
    af_report,
    aami_report,
    format_af_report,
    format_report,
    macro_f1_score,
)
from src.model import build_model
from src.preprocess.filters import apply_signal_filters
from src.preprocess.rhythm import beat_rr_sequences, hrv_features
from src.preprocess.symbol_mapping import CLASS_NAMES, num_classes_from_map
from src.utils import set_seed


def _fake(n, window, num_classes, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, window)).astype(np.float32)
    y = rng.integers(0, num_classes, size=n).astype(np.int64)
    return X, y


def _run(ds_cls, X_y, model_name, num_classes, select_fn, report_fn, metric_name, groups=None):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = X_y
    device = torch.device("cpu")
    loaders = [
        make_loader(ds_cls(Xtr, ytr, augment=True), 32, shuffle=True, drop_last=True, seed=0),
        make_loader(ds_cls(Xva, yva), 32, seed=0),
        make_loader(ds_cls(Xte, yte), 32, seed=0),
    ]
    model = build_model(model_name, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = torch.nn.CrossEntropyLoss()
    out = tempfile.mkdtemp()
    return train_and_evaluate(
        model, *loaders, crit, opt, epochs=2, device=device, output_dir=out,
        select_fn=select_fn, report_fn=report_fn, metric_name=metric_name, test_groups=groups,
    )


def main():
    set_seed(0)

    #  beat pipeline 
    nc = num_classes_from_map()
    beat_data = (_fake(400, 128, nc, 1), _fake(120, 128, nc, 2), _fake(120, 128, nc, 3))
    _, rep, acc, _ = _run(
        BeatDataset, beat_data, "ecgcnn", nc,
        select_fn=lambda yt, yp: macro_f1_score(yt, yp, nc),
        report_fn=lambda yt, yp, g: aami_report(yt, yp, CLASS_NAMES),
        metric_name="macro_f1",
    )
    assert "macro_f1" in rep
    print(format_report(rep))
    print(f"[beat] SMOKE OK (acc={acc:.3f})\n")

    #  AF pipeline 
    af_data = (_fake(400, 32, 2, 4), _fake(120, 32, 2, 5), _fake(120, 32, 2, 6))
    groups = np.array(["rec_a"] * 60 + ["rec_b"] * 60)
    _, rep2, acc2, _ = _run(
        RhythmDataset, af_data, "rhythmcnn", 2,
        select_fn=af_f1, report_fn=af_report, metric_name="af_f1", groups=groups,
    )
    assert "af_f1" in rep2 and rep2["per_record"]
    print(format_af_report(rep2))
    print(f"\n[af] SMOKE OK (acc={acc2:.3f})")

    #  multi-scale beat pipeline (morphology + RR context) 
    def _fake_ms(n, seed):
        r = np.random.default_rng(seed)
        return (
            r.standard_normal((n, 128)).astype(np.float32),
            r.integers(0, nc, size=n).astype(np.int64),
            r.standard_normal((n, 4)).astype(np.float32),
        )

    (Xtr, ytr, rtr), (Xva, yva, rva), (Xte, yte, rte) = _fake_ms(400, 10), _fake_ms(120, 11), _fake_ms(120, 12)
    rr_mean, rr_std = rtr.mean(0), rtr.std(0)
    device = torch.device("cpu")
    loaders = [
        make_loader(BeatDataset(Xtr, ytr, rr=rtr, rr_mean=rr_mean, rr_std=rr_std, augment=True),
                    32, shuffle=True, drop_last=True, seed=0),
        make_loader(BeatDataset(Xva, yva, rr=rva, rr_mean=rr_mean, rr_std=rr_std), 32, seed=0),
        make_loader(BeatDataset(Xte, yte, rr=rte, rr_mean=rr_mean, rr_std=rr_std), 32, seed=0),
    ]
    model = build_model("multiscale", nc).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    out = tempfile.mkdtemp()
    _, rep3, acc3, _ = train_and_evaluate(
        model, *loaders, torch.nn.CrossEntropyLoss(), opt, epochs=2, device=device, output_dir=out,
        select_fn=lambda yt, yp: macro_f1_score(yt, yp, nc),
        report_fn=lambda yt, yp, g: aami_report(yt, yp, CLASS_NAMES), metric_name="macro_f1",
    )
    assert "macro_f1" in rep3
    print(f"\n[multiscale] SMOKE OK (acc={acc3:.3f})")

    test_preprocess()
    test_holter_pipeline(nc)


def test_preprocess():
    #前處理元件：訊號濾波、HRV、per-beat RR 序列、RhythmCNN+HRV
    rng = np.random.default_rng(7)
    fs = 360
    sig = rng.standard_normal(3600).astype(np.float32)
    cfg = {"enabled": True, "bandpass": True, "bp_low": 0.5, "bp_high": 40.0,
           "notch": True, "notch_freq": 60.0}
    filt = apply_signal_filters(sig, fs, cfg)
    assert filt.shape == sig.shape and np.isfinite(filt).all()
    assert np.allclose(apply_signal_filters(sig, fs, {"enabled": False}), sig)  # 關閉時原樣

    r_samples = np.cumsum(rng.integers(200, 400, size=50))  # 假 R 峰
    seqs = beat_rr_sequences(r_samples, fs, seq_len=32)
    assert seqs.shape == (50, 32) and np.isfinite(seqs).all()
    hrv = hrv_features(seqs[10])
    assert hrv.shape == (3,) and np.isfinite(hrv).all()

    # RhythmCNN + HRV late fusion 前向
    model = build_model("rhythmcnn", 2, n_hrv=3)
    out = model(torch.randn(8, 1, 32), torch.randn(8, 3))
    assert out.shape == (8, 2)
    print("\n[preprocess] SMOKE OK（filter / HRV / rr_seq / RhythmCNN+HRV）")


def _synthetic_ecg(fs=360, dur=60, rr=0.8, seed=0):
    #P-QRS-T 高斯波組成的合成 ECG（R 峰位置已知），給 Holter 管線測試用
    t = np.arange(int(dur * fs)) / fs
    beats = np.arange(0.5, dur - 0.5, rr)
    sig = np.zeros_like(t)
    for b in beats:
        sig += 1.2 * np.exp(-((t - b) / 0.012) ** 2) - 0.25 * np.exp(-((t - b - 0.03) / 0.01) ** 2)
        sig += 0.15 * np.exp(-((t - b + 0.18) / 0.03) ** 2) + 0.3 * np.exp(-((t - b - 0.3) / 0.06) ** 2)
    sig += 0.02 * np.random.default_rng(seed).standard_normal(len(t))
    return sig.astype(np.float32), np.round(beats * fs).astype(np.int64)


def test_holter_pipeline(nc):
    #離線 Holter 管線：R 峰偵測、EC57 比對、端到端混淆、AF 規則、摘要、完整 analyze_record
    from src.holter import (
        CLS_N, CLS_S, CLS_V, af_episodes, analyze_record, beat_confusion, end2end_metrics,
        rr_level_af, summarize, suppress_s_in_af,
    )
    from src.model import MultiScaleECGNet, RhythmCNN
    from src.preprocess.qrs import detect_r_peaks, detection_stats, match_beats

    # R 峰偵測（合成 ECG 上應全對）
    sig, ref = _synthetic_ecg()
    pk = detect_r_peaks(sig, 360, method="gqrs")
    st = detection_stats(ref, pk, int(0.15 * 360))
    assert st["se"] == 1.0 and st["ppv"] == 1.0, st

    # 一對一比對：已知答案
    r2d, d2r = match_beats([100, 200, 300], [102, 260, 305, 400], tol=10)
    assert r2d.tolist() == [0, -1, 2] and d2r.tolist() == [0, -1, 2, -1]

    # 端到端混淆：漏偵測算漏判、誤偵測算進 +P 分母
    cm, missed, false = beat_confusion([100, 200, 300], [CLS_N, CLS_V, CLS_V], [101, 299, 500], [CLS_N, CLS_V, CLS_V], tol=10)
    rows = {r["class"]: r for r in end2end_metrics(cm, missed, false)}
    assert missed[CLS_V] == 1 and false[CLS_V] == 1
    assert abs(rows["V"]["se"] - 0.5) < 1e-9 and abs(rows["V"]["ppv"] - 0.5) < 1e-9

    # AF 期間不計 PAC；AF 段落與摘要
    pred = np.array([CLS_N, CLS_S, CLS_S, CLS_V, CLS_V, CLS_V, CLS_N])
    rr_af = np.array([0, 1, 0, 0, 0, 0])   # 第 1 個 RR（拍 1→2）在 AF 內 → 只有第 2 拍的 S 被改掉
    assert suppress_s_in_af(pred, rr_af).tolist() == [CLS_N, CLS_S, CLS_N, CLS_V, CLS_V, CLS_V, CLS_N]
    prob, lab = rr_level_af(40, np.array([0, 16]), 32, np.array([0.9, 0.1]), 0.5)
    assert prob.shape == (40,) and lab[:16].all() and not lab[-8:].any()
    peaks = np.arange(0, 360 * 100, 288)
    assert len(af_episodes(peaks, 360, np.ones(len(peaks) - 1, np.int64))) == 1
    S = summarize(peaks[:7], 360, pred, n_samples=peaks[6] + 360)
    assert S["counts"]["V"] == 3 and S["v_runs"] == 1 and S["v_run_longest"] == 3

    # 完整部署管線（隨機權重 bundle；只驗形狀與流程，含 250 Hz 重取樣路徑）
    beat_b = {"models": [MultiScaleECGNet(nc).eval()], "rr_mean": np.zeros(4, np.float32),
              "rr_std": np.ones(4, np.float32), "window": 256, "rr_local_window": 10, "rr_clip": (0.2, 2.0),
              "class_names": CLASS_NAMES, "qrs": {"method": "gqrs", "correct": True}}
    af_b = {"models": [RhythmCNN(2).eval()], "window": 32, "step": 16, "rr_clip": (0.2, 2.0), "af_threshold": 0.5}
    for fs in (360, 250):
        s, _ = _synthetic_ecg(fs=fs)
        res = analyze_record(s, fs, beat_b, af_b)
        assert res["fs"] == 360 and len(res["beat_pred"]) == len(res["peaks"]) >= 70
        assert res["rr_af"] is not None and "af_burden" in res["summary"]
    print(f"[holter] SMOKE OK（R 峰 Se/+P=1、EC57 比對、端到端混淆、AF/PAC 規則、摘要、analyze_record "
          f"{res['timing']['total'] * 1000:.0f} ms）")


if __name__ == "__main__":
    main()
