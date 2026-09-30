"""Definisi model: CNN sederhana dari nol, MobileNetV2, dan ResNet50 pretrained."""

import io
import time

import torch
from torch import nn
from torchvision import models


def conv_block(in_ch, out_ch):
    """Dua konvolusi 3x3 + BatchNorm + ReLU, lalu max pooling (ukuran spasial jadi setengah)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    """Baseline: 4 blok konvolusi (32, 64, 128, 256 filter), global average pooling, lalu classifier."""

    def __init__(self, num_classes=6, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_mobilenet_v2(num_classes=6, dropout=0.2):
    """MobileNetV2 dengan bobot ImageNet, classifier diganti untuk num_classes kelas."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return model


def build_resnet50(num_classes=6, dropout=0.2):
    """ResNet50 dengan bobot ImageNet, layer fc diganti untuk num_classes kelas."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return model


def get_head(model):
    """Bagian classifier (head) yang selalu dilatih."""
    return model.fc if isinstance(model, models.ResNet) else model.classifier


def freeze_backbone(model):
    """Tahap 1: bekukan semua layer kecuali classifier."""
    for p in model.parameters():
        p.requires_grad = False
    for p in get_head(model).parameters():
        p.requires_grad = True


def unfreeze_last_blocks(model, n_blocks):
    """Tahap 2: buka n_blocks blok terakhir backbone (plus classifier) untuk fine-tuning."""
    freeze_backbone(model)
    if isinstance(model, models.ResNet):
        blocks = [model.layer1, model.layer2, model.layer3, model.layer4]
    else:
        blocks = list(model.features)
    for block in blocks[-n_blocks:]:
        for p in block.parameters():
            p.requires_grad = True


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def state_dict_size_mb(model):
    """Ukuran bobot model (float32) dalam MB, mendekati ukuran file ONNX nanti."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / 1024**2


@torch.no_grad()
def measure_latency_ms(model, n_runs=30, warmup=5, img_size=224):
    """Rata-rata waktu inferensi satu gambar di CPU (float32), sebagai gambaran kecepatan di browser."""
    model = model.to("cpu").eval()
    x = torch.randn(1, 3, img_size, img_size)
    for _ in range(warmup):
        model(x)
    start = time.perf_counter()
    for _ in range(n_runs):
        model(x)
    return (time.perf_counter() - start) / n_runs * 1000
