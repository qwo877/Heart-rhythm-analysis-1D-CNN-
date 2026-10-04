"""torch Dataset 包裝。

- BeatDataset       ：mitdb 單拍（Phase 0 / 1.2 輕量 multiscale）。
- RhythmDataset     ：afdb RR 窗，可選 HRV late fusion（Phase 1.3 / 1.5）。
- BeatRhythmDataset ：joint 的 beat 側，morphology + 局部 RR 序列（Phase 1.4）。
"""
import numpy as np
import torch
from torch.utils.data import Dataset


def _standardize(vec, mean, std):
    if mean is None or std is None:
        return vec
    return (vec - mean) / (std + 1e-8)


class BeatDataset(Dataset):
    """mitdb 單拍。augment 與 normalize 可開關（Phase 0 ablation 需求）。

    多尺度：若給 rr（每 beat 的 4 維 RR 特徵）＋ rr_mean/rr_std，__getitem__ 會回傳
    ((morphology, rr_vec), y)。rr=None 則退回單尺度 (morphology, y)。
    """

    def __init__(self, X, y, rr=None, rr_mean=None, rr_std=None,
                 augment=False, normalize=True, noise_std=0.01, max_shift=5):
        self.X = X
        self.y = y
        self.rr = rr
        self.rr_mean = rr_mean
        self.rr_std = rr_std
        self.augment = augment
        self.normalize = normalize
        self.noise_std = noise_std
        self.max_shift = max_shift

    def __len__(self):
        return len(self.X)

    def _morph(self, idx):
        x = self.X[idx].copy()
        if self.augment:
            x = x + self.noise_std * np.random.randn(*x.shape).astype(np.float32)
            shift = np.random.randint(-self.max_shift, self.max_shift + 1)
            x = np.roll(x, shift)
        if self.normalize:
            x = (x - x.mean()) / (x.std() + 1e-8)
        return torch.from_numpy(x.astype(np.float32)).unsqueeze(0)

    def __getitem__(self, idx):
        morph = self._morph(idx)
        label = torch.tensor(int(self.y[idx]), dtype=torch.long)
        if self.rr is None:
            return morph, label
        rr = _standardize(self.rr[idx].astype(np.float32), self.rr_mean, self.rr_std)
        return (morph, torch.from_numpy(rr.astype(np.float32))), label


class RhythmDataset(Dataset):
    """afdb RR 窗（秒）。AF 靠 RR 變異度，預設保留原始 RR 值（normalize 預設關）。

    hrv 給定時（+ hrv_mean/std 標準化）→ 回傳 ((rr, hrv), y) 做 late fusion；
    否則 (rr, y)。augment：對 RR 加小幅 jitter，模擬 R 峰偵測誤差。
    """

    def __init__(self, X, y, hrv=None, hrv_mean=None, hrv_std=None,
                 augment=False, normalize=False, noise_std=0.01):
        self.X = X
        self.y = y
        self.hrv = hrv
        self.hrv_mean = hrv_mean
        self.hrv_std = hrv_std
        self.augment = augment
        self.normalize = normalize
        self.noise_std = noise_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        if self.augment:
            x = x + self.noise_std * np.random.randn(*x.shape).astype(np.float32)
        if self.normalize:
            x = (x - x.mean()) / (x.std() + 1e-8)
        rr = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
        label = torch.tensor(int(self.y[idx]), dtype=torch.long)
        if self.hrv is None:
            return rr, label
        h = _standardize(self.hrv[idx].astype(np.float32), self.hrv_mean, self.hrv_std)
        return (rr, torch.from_numpy(h.astype(np.float32))), label


class BeatRhythmDataset(Dataset):
    """joint 的 beat 側：morphology 窗 + 該 beat 的局部 RR 序列（+可選 HRV）+ beat 標籤。

    RR 序列保持原始秒值（不做 per-window 標準化），與 afdb 的 RR 窗同尺度，
    好讓 JointECGNet 的**共享 RhythmEncoder** 能同時吃兩邊。
    __getitem__ 回傳 (inputs_tuple, y)，inputs = (morph, rr_seq[, hrv])，
    由 joint 訓練迴圈解包餵給 model.forward_beat。
    """

    def __init__(self, X, y, rr_seq, hrv=None, hrv_mean=None, hrv_std=None,
                 augment=False, normalize=True, noise_std=0.01, max_shift=5):
        self.X = X
        self.y = y
        self.rr_seq = rr_seq
        self.hrv = hrv
        self.hrv_mean = hrv_mean
        self.hrv_std = hrv_std
        self.augment = augment
        self.normalize = normalize
        self.noise_std = noise_std
        self.max_shift = max_shift

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        if self.augment:
            x = x + self.noise_std * np.random.randn(*x.shape).astype(np.float32)
            shift = np.random.randint(-self.max_shift, self.max_shift + 1)
            x = np.roll(x, shift)
        if self.normalize:
            x = (x - x.mean()) / (x.std() + 1e-8)
        morph = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
        rr_seq = torch.from_numpy(self.rr_seq[idx].astype(np.float32)).unsqueeze(0)  # (1,L)
        label = torch.tensor(int(self.y[idx]), dtype=torch.long)
        if self.hrv is None:
            return (morph, rr_seq), label
        h = _standardize(self.hrv[idx].astype(np.float32), self.hrv_mean, self.hrv_std)
        return (morph, rr_seq, torch.from_numpy(h.astype(np.float32))), label
