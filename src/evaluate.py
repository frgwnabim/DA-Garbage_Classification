"""Evaluasi: prediksi, metrik, waktu inferensi, confusion matrix, Grad-CAM, dan uji ketahanan."""

import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import models as tv_models
from torchvision import transforms

from src.data import GarbageDataset, get_transforms


@torch.no_grad()
def predict(model, loader, device, use_amp=True):
    """Kembalikan label asli, prediksi, dan probabilitas softmax untuk seluruh loader."""
    model.eval()
    labels_all, probs_all = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
        probs_all.append(torch.softmax(logits.float(), dim=1).cpu())
        labels_all.append(labels)
    probs = torch.cat(probs_all).numpy()
    return torch.cat(labels_all).numpy(), probs.argmax(1), probs


@torch.no_grad()
def predict_many(models, loader, device, use_amp=True):
    """Prediksi beberapa model sekaligus dalam satu kali baca data (hemat waktu decode gambar)."""
    for m in models.values():
        m.eval()
    labels_all, preds_all = [], {k: [] for k in models}
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            for k, m in models.items():
                preds_all[k].append(m(images).argmax(1).cpu())
        labels_all.append(labels)
    y_true = torch.cat(labels_all).numpy()
    return y_true, {k: torch.cat(v).numpy() for k, v in preds_all.items()}


def summary_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def per_class_report(y_true, y_pred, classes):
    return classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)


@torch.no_grad()
def inference_time_ms(model, device, n_runs=50, warmup=10, img_size=224):
    """Rata-rata waktu inferensi satu gambar (batch 1, float32) di device tertentu."""
    model = model.to(device).eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / n_runs * 1000


def top_confusions(cm, classes, k=5):
    """Pasangan (asli -> prediksi) dengan jumlah kesalahan terbanyak."""
    pairs = [
        {"true": classes[i], "pred": classes[j], "count": int(cm[i, j]), "rate_of_true": float(cm[i, j] / cm[i].sum())}
        for i in range(len(classes)) for j in range(len(classes)) if i != j and cm[i, j] > 0
    ]
    return sorted(pairs, key=lambda p: (-p["count"], -p["rate_of_true"]))[:k]


def plot_confusion_matrices(cm, classes, title):
    """Confusion matrix versi angka dan versi ternormalisasi per baris (recall per kelas)."""
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, data, fmt, sub in [
        (axes[0], cm, "d", "Jumlah gambar"),
        (axes[1], cm_norm, ".2f", "Ternormalisasi per kelas asli"),
    ]:
        im = ax.imshow(data, cmap="Blues", vmin=0)
        threshold = data.max() / 2
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center", fontsize=10,
                        color="white" if data[i, j] > threshold else "#0b0b0b")
        ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right")
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Label asli")
        ax.set_title(sub)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def gradcam_target_layer(model):
    """Layer konvolusi terakhir, tempat Grad-CAM mengambil peta aktivasi."""
    if isinstance(model, tv_models.ResNet):
        return model.layer4[-1]
    if isinstance(model, tv_models.MobileNetV2):
        return model.features[-1]
    return model.features[-1]


def compute_gradcam(model, input_tensor, target_classes):
    """Heatmap Grad-CAM (nilai 0 sampai 1, ukuran 224 x 224) untuk batch gambar."""
    model.eval()
    with GradCAM(model=model, target_layers=[gradcam_target_layer(model)]) as cam:
        targets = [ClassifierOutputTarget(int(c)) for c in target_classes]
        return cam(input_tensor=input_tensor, targets=targets)


def overlay_cam(rgb_float, heatmap):
    """Tempelkan heatmap di atas gambar (rgb_float: array 224 x 224 x 3, nilai 0 sampai 1)."""
    return show_cam_on_image(rgb_float, heatmap, use_rgb=True, image_weight=0.55)


def border_attention_ratio(heatmaps, border_frac=0.125):
    """Rasio rata-rata heatmap di bingkai tepi dibanding rata-rata seluruh gambar.

    Nilai < 1 berarti perhatian model lebih sedikit di tepi (umumnya background) daripada
    rata-rata, nilai > 1 berarti model justru banyak melihat tepi.
    """
    h = heatmaps.shape[-1]
    b = int(h * border_frac)
    mask = np.zeros((h, h), dtype=bool)
    mask[:b, :] = mask[-b:, :] = mask[:, :b] = mask[:, -b:] = True
    return heatmaps[:, mask].mean(axis=1) / (heatmaps.reshape(len(heatmaps), -1).mean(axis=1) + 1e-8)


# ---------------------------------------------------------------- uji ketahanan

def _edge_fill(img):
    """Warna median piksel tepi, dipakai mengisi sudut kosong setelah rotasi."""
    arr = np.asarray(img)
    edge = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    return tuple(int(v) for v in np.median(edge, axis=0))


def perturbations():
    """Gangguan yang meniru kondisi foto di dunia nyata. Semua deterministik."""
    return {
        "Asli": lambda im: im,
        "Gelap (60%)": lambda im: ImageEnhance.Brightness(im).enhance(0.6),
        "Sangat gelap (35%)": lambda im: ImageEnhance.Brightness(im).enhance(0.35),
        "Blur ringan (r=2)": lambda im: im.filter(ImageFilter.GaussianBlur(2)),
        "Blur berat (r=5)": lambda im: im.filter(ImageFilter.GaussianBlur(5)),
        "Rotasi 30 derajat": lambda im: im.rotate(30, resample=Image.BILINEAR, fillcolor=_edge_fill(im)),
        "Rotasi 90 derajat": lambda im: im.rotate(90, expand=True),
    }


def perturbed_loader(data_dir, items, classes, perturb, batch_size=32):
    tf = transforms.Compose([transforms.Lambda(perturb), get_transforms(train=False)])
    ds = GarbageDataset(data_dir, items, classes, tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


__all__ = [
    "predict", "predict_many", "summary_metrics", "per_class_report", "inference_time_ms",
    "confusion_matrix", "top_confusions", "plot_confusion_matrices", "compute_gradcam",
    "overlay_cam", "border_attention_ratio", "perturbations", "perturbed_loader",
]
