"""共用訓練 / 評估核心（beat 與 AF 兩個任務都用這裡）。

模型選擇一律看 **val 上的任務指標**（beat=macro-F1、AF=AF-class F1），不是 accuracy。
"""
import csv
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def seed_worker(worker_id):
    """讓 DataLoader worker 的 augmentation 亂數可重現。"""
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)


def make_loader(dataset, batch_size, shuffle=False, sampler=None,
                num_workers=0, seed=42, drop_last=False):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    total_loss = 0.0
    preds_all, labels_all = [], []
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            yb = yb.to(device)
            if train:
                optimizer.zero_grad()
            # 多尺度模型的 xb 是 (morph, rr) tuple；單尺度是單一 tensor
            if isinstance(xb, (list, tuple)):
                xb = [t.to(device) for t in xb]
                out = model(*xb)
            else:
                xb = xb.to(device)
                out = model(xb)
            loss = criterion(out, yb)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * yb.size(0)
            preds_all.append(out.argmax(1).detach().cpu().numpy())
            labels_all.append(yb.detach().cpu().numpy())
    preds = np.concatenate(preds_all)
    labels = np.concatenate(labels_all)
    acc = float((preds == labels).mean())
    return total_loss / len(labels), acc, labels, preds


def plot_history(history, path, metric_name="metric"):
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.title("loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["train_metric"], label="train")
    plt.plot(history["val_metric"], label="val")
    plt.title(metric_name)
    plt.xlabel("epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def train_and_evaluate(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    epochs,
    device,
    output_dir,
    select_fn,
    report_fn,
    metric_name="metric",
    test_groups=None,
):
    """通用訓練迴圈：CSV log、以 val 任務指標選最佳、載回最佳在 test 出報表。

    select_fn(y_true, y_pred) -> float（越大越好，用於挑 val 最佳 epoch）
    report_fn(y_true, y_pred, groups) -> report 物件（最終 test 報表）
    """
    csv_path = os.path.join(output_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_acc", f"train_{metric_name}",
             "val_loss", "val_acc", f"val_{metric_name}", "seconds"]
        )

    best_val = -1.0
    best_path = os.path.join(output_dir, "best_model.pt")
    history = {"train_loss": [], "val_loss": [], "train_metric": [], "val_metric": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_y, tr_p = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va_loss, va_acc, va_y, va_p = run_epoch(model, val_loader, criterion, optimizer, device, False)
        tr_m = select_fn(tr_y, tr_p)
        va_m = select_fn(va_y, va_p)
        dt = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_metric"].append(tr_m)
        history["val_metric"].append(va_m)

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [epoch, f"{tr_loss:.6f}", f"{tr_acc:.6f}", f"{tr_m:.6f}",
                 f"{va_loss:.6f}", f"{va_acc:.6f}", f"{va_m:.6f}", f"{dt:.1f}"]
            )
        print(
            f"Epoch {epoch:02d}/{epochs} - {dt:4.1f}s | "
            f"train loss {tr_loss:.4f} {metric_name} {tr_m:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f} {metric_name} {va_m:.4f}"
        )

        if va_m > best_val:
            best_val = va_m
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch,
                 f"val_{metric_name}": va_m},
                best_path,
            )
            print(f"  -> 新最佳 val {metric_name}={va_m:.4f}，存檔 {best_path}")

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    te_loss, te_acc, te_y, te_p = run_epoch(model, test_loader, criterion, None, device, False)
    report = report_fn(te_y, te_p, test_groups)
    return history, report, te_acc, te_loss
