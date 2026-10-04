"""1D-CNN 模型。

- ECGCNN：單拍 morphology 分支（Phase 0；也是多尺度的 local branch）。
- RhythmEncoder：RR 序列 → embedding 的**共享元件**（Phase 1.4 joint 的關鍵：同一個 encoder
  同時吃 afdb 的 AF 窗與 mitdb 每個 beat 的局部 RR context）。
- RhythmCNN：RR 序列 (+可選 HRV late fusion) → 節律分類（Phase 1.3 AF 任務）。
- MultiScaleECGNet：morphology + 4 維 RR 特徵（Phase 1.2 的輕量 late-fusion 版）。
- JointECGNet：morphology + 共享 RhythmEncoder → beat head 與 AF head 兩個任務（Phase 1.4）。
num_classes 皆為參數，不寫死。
"""
import torch
import torch.nn as nn


def _morph_features(in_channels=1):
    """單拍 morphology 特徵萃取（輸出 64 維，經 AdaptiveAvgPool → (B,64,1)）。"""
    return nn.Sequential(
        nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Conv1d(16, 32, kernel_size=5, padding=2),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Conv1d(32, 64, kernel_size=5, padding=2),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
    )


class MorphEncoder(nn.Module):
    """單拍 morphology 編碼器（= ECGCNN 的 features 部分），輸出 64 維 embedding。

    Phase 2 SSL 預訓練的**對象**：把 `_morph_features` 包成可獨立訓練/凍結的 encoder。
    因為 `self.features` 與 `ECGCNN.features` 都是同一個 `_morph_features()`（逐層相同），
    預訓練完的權重可直接 `ecgcnn.features.load_state_dict(enc.features.state_dict())`
    轉進監督分類器 —— 這是自監督預訓練實驗「backbone 不變、只換表示怎麼學」的前提。
    """

    out_dim = 64

    def __init__(self, in_channels=1):
        super().__init__()
        self.features = _morph_features(in_channels)

    def forward(self, x):
        return self.features(x).flatten(1)  # (B,64)


class ProjectionHead(nn.Module):
    """SimCLR / CLOCS 投影頭：encoder embedding → 對比空間（下游一律丟棄）。

    對比學習在投影空間算 InfoNCE，但表示品質評估（linear-probe / align-uniform）
    用**投影前**的 encoder embedding —— 標準 SimCLR 作法，投影頭只是訓練期的鷹架。
    """

    def __init__(self, in_dim=64, hidden=64, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class RhythmEncoder(nn.Module):
    """RR 序列（秒）→ 64 維 embedding。joint 的共享部分：afdb AF 窗與 mitdb 局部 RR 共用。"""

    out_dim = 64

    def __init__(self, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return self.features(x).flatten(1)  # (B,64)


class ECGCNN(nn.Module):
    def __init__(self, num_classes, in_channels=1, dropout=0.3):
        super().__init__()
        self.features = _morph_features(in_channels)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class RhythmCNN(nn.Module):
    """吃一段 RR 序列（秒），學 RR 變異度 → 節律分類（AF / non-AF）。

    n_hrv>0：把手工 HRV 特徵（SDNN/RMSSD/pNN50）在 head 前做 late fusion。
    forward 兼容單輸入 (x) 與雙輸入 (x, hrv)，配合 engine 對 tuple batch 的 model(*xb)。
    """

    def __init__(self, num_classes, in_channels=1, n_hrv=0, dropout=0.3):
        super().__init__()
        self.encoder = RhythmEncoder(in_channels)
        self.n_hrv = n_hrv
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(RhythmEncoder.out_dim + n_hrv, num_classes),
        )

    def forward(self, x, hrv=None):
        z = self.encoder(x)
        if self.n_hrv > 0:
            z = torch.cat([z, hrv], dim=1)
        return self.classifier(z)


class MultiScaleECGNet(nn.Module):
    """多尺度雙分支（輕量版）：local(morphology CNN) + context(4 維 RR 特徵 MLP) → 融合分類。

    這是 Phase 1.2 的 late-fusion 輕量版；完整的「RR 序列 encoder」在 JointECGNet。
    核心假設（來自 Phase 0）：S 類光看單拍形狀分不出，要靠 RR 節律 context。
    """

    def __init__(self, num_classes, n_rr_features=4, in_channels=1, dropout=0.3):
        super().__init__()
        self.morph = _morph_features(in_channels)  # → (B,64,1)
        self.rr = nn.Sequential(
            nn.Linear(n_rr_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64 + 32, num_classes),
        )

    def forward(self, x_morph, x_rr):
        m = self.morph(x_morph).flatten(1)  # (B,64)
        r = self.rr(x_rr)                    # (B,32)
        return self.classifier(torch.cat([m, r], dim=1))


class JointECGNet(nn.Module):
    """Phase 1.4 joint model：morphology encoder + 共享 RhythmEncoder → 兩個任務 head。

    共享點：**同一個 RhythmEncoder** 既提供 mitdb 每個 beat 的節律 context
    給 beat head，又被 afdb 的 AF 任務監督。於是「AF 有沒有幫到 beat」= 加了 AF 任務對這個
    共享 encoder 的梯度，beat 分類有沒有變好（用 λ_af=0 vs >0 做乾淨對照）。

    n_hrv>0 時，兩個 head 都可 late-fuse HRV 特徵（1.5）。
    """

    def __init__(self, num_beat_classes, num_af_classes=2, n_hrv=0, in_channels=1, dropout=0.3):
        super().__init__()
        self.morph = _morph_features(in_channels)      # beat morphology → (B,64,1)
        self.rhythm = RhythmEncoder(in_channels)       # 共享 RR encoder → (B,64)
        self.n_hrv = n_hrv
        self.beat_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64 + RhythmEncoder.out_dim + n_hrv, num_beat_classes),
        )
        self.af_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(RhythmEncoder.out_dim + n_hrv, num_af_classes),
        )

    def forward_beat(self, x_morph, x_rrseq, hrv=None):
        m = self.morph(x_morph).flatten(1)     # (B,64)
        r = self.rhythm(x_rrseq)               # (B,64) 共享 encoder
        z = torch.cat([m, r], dim=1)
        if self.n_hrv > 0:
            z = torch.cat([z, hrv], dim=1)
        return self.beat_head(z)

    def forward_af(self, x_rrseq, hrv=None):
        r = self.rhythm(x_rrseq)               # (B,64) 共享 encoder
        z = torch.cat([r, hrv], dim=1) if self.n_hrv > 0 else r
        return self.af_head(z)


class UncertaintyWeighting(nn.Module):
    """Kendall et al. 2018 的多任務不確定度加權（進階 λ 設定）。

    每個任務學一個 log 方差 s_i；total = Σ exp(-s_i)·L_i + s_i。
    避免手掃 λ，讓兩個任務依各自雜訊自動平衡。回傳加權後總損失。
    """

    def __init__(self, n_tasks=2):
        super().__init__()
        self.log_var = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses):
        total = 0.0
        for i, loss in enumerate(losses):
            total = total + torch.exp(-self.log_var[i]) * loss + self.log_var[i]
        return total

    def weights(self):
        """回傳目前等效的任務權重 exp(-s_i)（監控用）。"""
        return torch.exp(-self.log_var).detach().cpu().tolist()


def build_model(name, num_classes, dropout=0.3, n_rr_features=4, n_hrv=0):
    name = name.lower()
    if name == "ecgcnn":
        return ECGCNN(num_classes=num_classes, dropout=dropout)
    if name == "rhythmcnn":
        return RhythmCNN(num_classes=num_classes, n_hrv=n_hrv, dropout=dropout)
    if name == "multiscale":
        return MultiScaleECGNet(num_classes=num_classes, n_rr_features=n_rr_features, dropout=dropout)
    raise ValueError(f"未知 model: {name}")
