"""Loop training: mixed precision, class weights, scheduler, early stopping, checkpoint, history."""

import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn


def freeze_frozen_batchnorm(model):
    """BatchNorm di layer yang dibekukan tetap memakai statistik ImageNet (mode eval).

    Tanpa ini, model.train() akan memperbarui running mean/var walaupun bobotnya dibekukan,
    sehingga fitur pretrained ikut bergeser.
    """
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            if not any(p.requires_grad for p in m.parameters()):
                m.eval()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp=True):
    model.train()
    freeze_frozen_batchnorm(model)
    total_loss, correct, n = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        n += labels.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=True):
    """Kembalikan loss, accuracy, label asli, dan prediksi."""
    model.eval()
    total_loss, n = 0.0, 0
    all_labels, all_preds = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        all_labels.append(labels.cpu())
        all_preds.append(outputs.argmax(1).cpu())
    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    return total_loss / n, float((y_true == y_pred).mean()), y_true, y_pred


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro"))


def fit(
    model,
    loaders,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    checkpoint_path,
    patience=5,
    stage="train",
    best_val_loss=float("inf"),
    use_amp=True,
    verbose=True,
):
    """Latih model dengan early stopping berdasarkan val loss.

    Checkpoint hanya disimpan kalau val loss lebih baik dari best_val_loss. Dengan mengoper
    best_val_loss dari tahap sebelumnya, tahap fine-tuning hanya menimpa checkpoint kalau
    benar-benar lebih baik. Setelah selesai, bobot terbaik dimuat kembali ke model.
    Tanda * di log berarti val loss membaik dan checkpoint disimpan.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    history = []
    epochs_without_improvement = 0
    improved_once = False

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, scaler, device, use_amp)
        val_loss, val_acc, _, _ = evaluate(model, loaders["val"], criterion, device, use_amp)
        lr = optimizer.param_groups[0]["lr"]
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()
        epoch_time = time.perf_counter() - start

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            improved_once = True
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1

        history.append({
            "stage": stage, "epoch": epoch, "lr": lr,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "epoch_time": epoch_time, "is_best": is_best,
        })
        if verbose:
            print(
                f"[{stage}] epoch {epoch:02d} | train loss {train_loss:.3f} acc {train_acc:.3f} | "
                f"val loss {val_loss:.3f} acc {val_acc:.3f} | lr {lr:.0e} | {epoch_time:.1f}s{' *' if is_best else ''}"
            )
        if epochs_without_improvement >= patience:
            break

    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    best = min(history, key=lambda h: h["val_loss"])
    print(
        f"[{stage}] {len(history)} epoch, stop {'karena early stopping' if len(history) < epochs else 'di batas epoch'}. "
        f"Val loss terbaik tahap ini: {best['val_loss']:.4f} (epoch {best['epoch']}), "
        f"checkpoint {'diperbarui' if improved_once else 'tidak berubah'}."
    )
    return history, best_val_loss


def summarize_history(history):
    return {
        "epochs": len(history),
        "train_time_sec": float(np.sum([h["epoch_time"] for h in history])),
        "best_val_loss": float(min(h["val_loss"] for h in history)),
    }


def plot_history(history, title):
    """Kurva loss dan accuracy train vs val. Garis putus-putus menandai awal fine-tuning."""
    import matplotlib.pyplot as plt

    blue, orange, muted = "#2a78d6", "#eb6834", "#52514e"
    x = np.arange(1, len(history) + 1)
    best_idx = int(np.argmin([h["val_loss"] for h in history]))
    stage_starts = [i for i in range(1, len(history)) if history[i]["stage"] != history[i - 1]["stage"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, key, label in [(axes[0], "loss", "Loss (weighted cross-entropy)"), (axes[1], "acc", "Accuracy")]:
        ax.plot(x, [h[f"train_{key}"] for h in history], color=blue, linewidth=2, label="Train")
        ax.plot(x, [h[f"val_{key}"] for h in history], color=orange, linewidth=2, label="Validation")
        ax.scatter([x[best_idx]], [history[best_idx][f"val_{key}"]], s=60, color=orange,
                   edgecolor="white", linewidth=2, zorder=3, label="Checkpoint terbaik")
        for s in stage_starts:
            ax.axvline(s + 0.5, color=muted, linestyle="--", linewidth=1)
            ax.text(s + 0.7, 0.97, "fine-tuning", transform=ax.get_xaxis_transform(),
                    va="top", fontsize=9, color=muted)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(frameon=False, loc="best")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()
