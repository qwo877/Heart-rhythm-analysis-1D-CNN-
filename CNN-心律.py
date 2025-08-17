import os
import argparse
import wfdb
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.optim as optim
import time

def build_symbol_map():
    # 類別編號: 0=N, 1=S, 2=V, 3=F, 4=Q
    symbol_to_class = {
        # N 類（正常及類似正常）
        'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
        # S 類（上心室異常）
        'A': 1, 'a': 1, 'J': 1, 'S': 1,
        # V 類（室性異常）
        'V': 2, 'E': 2,
        # F 類（融合）
        'F': 3,
        # Q 類（其他/未知/起搏等）
        '/': 4, 'f': 4, 'Q': 4, 'x': 4, 'p': 4, 't':4, 'T':4, '+':4, '|':4, '?':4
    }
    return symbol_to_class

def download_mitdb(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    print("Downloading MIT-BIH (may take some time) ...")
    wfdb.dl_database('mitdb', out_dir)
    print("Download finished.")

def load_beats(data_dir, window=200, map_unknown_to_Q=True, verbose=True):
    """
    回傳: X (N, window) numpy float32, y (N,) int32
    """
    sym_map = build_symbol_map()
    beats = []
    labels = []

    if not os.path.exists(data_dir) or len([f for f in os.listdir(data_dir) if f.endswith('.dat')]) == 0:
        download_mitdb(data_dir)
    try:
        record_list = wfdb.get_record_list('mitdb')
    except Exception:
        record_list = []
        for f in os.listdir(data_dir):
            if f.endswith('.dat'):
                record_list.append(os.path.splitext(f)[0])

    if verbose:
        print("Records to load:", len(record_list))

    for rec_name in record_list:
        rec_path = os.path.join(data_dir, rec_name)
        try:
            rec = wfdb.rdrecord(rec_path)
            try:
                ch_idx = rec.sig_name.index('MLII')
            except ValueError:
                ch_idx = 0
            signal = rec.p_signal[:, ch_idx].flatten()
            ann = wfdb.rdann(rec_path, 'atr')
        except Exception as e:
            if verbose:
                print(f"Read error for {rec_name}: {e}")
            continue

        if verbose:
            unique_syms = sorted(set([s.decode('utf-8') if isinstance(s, bytes) else s for s in ann.symbol]))
            print(f"{rec_name} symbols: {unique_syms}")

        for samp, sym in zip(ann.sample, ann.symbol):
            if isinstance(sym, bytes):
                try:
                    sym = sym.decode('utf-8')
                except:
                    sym = str(sym)

            if map_unknown_to_Q:
                cls = sym_map.get(sym, 4)
            else:
                if sym not in sym_map:
                    continue
                cls = sym_map[sym]

            start = int(samp - window//2)
            end = int(samp + window//2)
            if start < 0 or end > len(signal):
                continue

            beat = signal[start:end].astype(np.float32)
            beats.append(beat)
            labels.append(cls)

    if len(beats) == 0:
        raise RuntimeError("No beats extracted. Check your data_dir and symbol mapping.")

    X = np.stack(beats)
    y = np.array(labels, dtype=np.int64)
    if verbose:
        print("Collected beats:", X.shape, "labels:", Counter(y))
    return X, y

class ECGDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        y = int(self.y[idx])

        if self.augment:
            x = x + 0.01 * np.random.randn(*x.shape)
            shift = np.random.randint(-5, 6)
            x = np.roll(x, shift)
        x = (x - x.mean()) / (x.std() + 1e-8)
        x = x.astype(np.float32)
        return torch.from_numpy(x).unsqueeze(0), torch.tensor(y, dtype=torch.long)

class ECGCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(ECGCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
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
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, correct / total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            total_loss += loss.item() * xb.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(yb.cpu().numpy().tolist())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    X, y = load_beats(args.data_dir, window=args.window, map_unknown_to_Q=True, verbose=True)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    print("Split sizes:", X_train.shape[0], X_val.shape[0], X_test.shape[0])

    train_ds = ECGDataset(X_train, y_train, augment=True)
    val_ds = ECGDataset(X_val, y_val, augment=False)
    test_ds = ECGDataset(X_test, y_test, augment=False)

    if args.use_sampler:
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / (class_counts + 1e-8)
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=2)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    num_classes = len(np.unique(y))
    model = ECGCNN(num_classes=num_classes).to(device)

    counts = np.bincount(y_train)
    class_weights = torch.tensor((counts.sum() / (counts + 1e-8)), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights if args.use_class_weights else None)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = eval_epoch(model, val_loader, criterion, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs} - {dt:.1f}s - train_loss {train_loss:.4f} acc {train_acc:.4f} | val_loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch}, args.save_model_path)
            print("Saved best model to", args.save_model_path)

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history['train_acc'], label='train_acc')
    plt.plot(history['val_acc'], label='val_acc')
    plt.legend()
    plt.tight_layout()
    plt.savefig('train_history.png')
    print("Saved train_history.png")

    ckpt = torch.load(args.save_model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    test_loss, test_acc, all_preds, all_labels = eval_epoch(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f}, acc: {test_acc:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(all_labels, all_preds, digits=4))

    # 儲存最終模型
    # torch.save(model.state_dict(), 'model_final.pt')
    # print("Saved model_final.pt")

if __name__ == '__main__':
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./mitdb_data', help='下載或放置 mitdb 的資料夾')
    parser.add_argument('--window', type=int, default=200, help='每個 beat 的 window 長度（samples）')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--use_sampler', action='store_true', help='是否使用 WeightedRandomSampler 於訓練集')
    parser.add_argument('--use_class_weights', action='store_true', help='是否在 CrossEntropyLoss 中使用 class weights')
    parser.add_argument('--save_model_path', type=str, default='best_model.pt')

    args, unknown = parser.parse_known_args()
    main(args)
