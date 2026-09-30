"""Deployment: export ONNX, verifikasi, quantization, konfigurasi preprocessing, dan contoh gambar."""

import json
import logging
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image

from src.data import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE

RESIZE_SIZE = 256  # sisi pendek di-resize ke 256, lalu center crop 224 (sama dengan get_transforms(train=False))
INPUT_NAME = "input"
OUTPUT_NAME = "logits"

# Exporter ONNX mencetak peringatan soal triton (tidak relevan untuk export CPU)
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)


def export_onnx(model, path, opset=18):
    """Export model ke satu file ONNX dengan input tetap 1x3x224x224 (tanpa dynamic batch)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu").eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        model, (dummy,), str(path),
        input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
        opset_version=opset, dynamo=True, external_data=False, verbose=False,
    )
    onnx.checker.check_model(onnx.load(str(path)))
    return path


def onnx_session(path):
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def onnx_logits(session, batch):
    """Jalankan ONNX satu per satu gambar (model di-export dengan batch 1)."""
    return np.concatenate([session.run([OUTPUT_NAME], {INPUT_NAME: x[None]})[0] for x in batch])


def onnx_info(path):
    model = onnx.load(str(path))
    return {
        "opset": max(o.version for o in model.opset_import if o.domain in ("", "ai.onnx")),
        "operators": sorted({n.op_type for n in model.graph.node}),
        "size_mb": Path(path).stat().st_size / 1024**2,
    }


def quantize_dynamic_int8(src, dst):
    """Dynamic quantization: bobot disimpan int8 (sekitar 4 kali lebih kecil), aktivasi tetap float."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    # value_info dari exporter dynamo bisa bentrok dengan shape inference quantizer, jadi dibuang dulu
    model = onnx.load(str(src))
    del model.graph.value_info[:]
    clean = Path(dst).with_suffix(".tmp.onnx")
    onnx.save(model, str(clean))
    logging.disable(logging.WARNING)  # sembunyikan saran pre-processing dari quantizer
    try:
        quantize_dynamic(str(clean), str(dst), weight_type=QuantType.QInt8)
    finally:
        logging.disable(logging.NOTSET)
        clean.unlink(missing_ok=True)
    return Path(dst)


def preprocessing_config():
    """Semua parameter preprocessing yang harus ditiru persis oleh browser."""
    return {
        "input_name": INPUT_NAME,
        "output_name": OUTPUT_NAME,
        "input_shape": [1, 3, IMG_SIZE, IMG_SIZE],
        "layout": "NCHW",
        "color_order": "RGB",
        "resize": {"shorter_side": RESIZE_SIZE, "interpolation": "bilinear"},
        "center_crop": IMG_SIZE,
        "scale": 1 / 255,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "output": "logits, terapkan softmax untuk mendapatkan probabilitas",
        "steps": [
            "Ubah gambar ke RGB",
            f"Resize sisi pendek ke {RESIZE_SIZE} px (bilinear), jaga rasio aspek",
            f"Center crop {IMG_SIZE} x {IMG_SIZE}",
            "Bagi nilai piksel dengan 255 (rentang 0 sampai 1)",
            "Normalisasi per channel: (x - mean) / std",
            "Susun sebagai float32 NCHW [1, 3, 224, 224]",
        ],
    }


def preprocess_from_config(image, config):
    """Preprocessing hanya dari isi preprocessing.json (meniru yang akan dilakukan browser)."""
    image = image.convert("RGB")
    w, h = image.size
    short = config["resize"]["shorter_side"]
    scale = short / min(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    image = image.resize((new_w, new_h), Image.BILINEAR)
    crop = config["center_crop"]
    left, top = round((new_w - crop) / 2), round((new_h - crop) / 2)
    image = image.crop((left, top, left + crop, top + crop))
    arr = np.asarray(image, dtype=np.float32) * config["scale"]
    arr = (arr - np.array(config["mean"], dtype=np.float32)) / np.array(config["std"], dtype=np.float32)
    return arr.transpose(2, 0, 1)


def save_samples(items, data_dir, out_dir, classes, per_class=2, max_size=400, seed=42, quality=88):
    """Simpan contoh gambar per kelas (JPEG, sisi terpanjang maksimal max_size) untuk demo website."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    manifest = []
    for cls in classes:
        cls_items = [rel for rel, c in items if c == cls]
        for n, rel in enumerate(sorted(rng.choice(cls_items, per_class, replace=False)), start=1):
            img = Image.open(Path(data_dir) / rel).convert("RGB")
            img.thumbnail((max_size, max_size), Image.LANCZOS)
            name = f"{cls}_{n}.jpg"
            img.save(out_dir / name, "JPEG", quality=quality, optimize=True)
            manifest.append({"file": name, "label": cls, "source": rel, "width": img.width, "height": img.height})
    (out_dir / "samples.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def annotate_samples(samples, samples_dir, session, config, classes):
    """Tambahkan prediksi yang diharapkan ke samples.json.

    Prediksi dihitung dari file JPEG 400 px (bukan gambar asli), memakai preprocessing dari
    preprocessing.json, jadi website seharusnya menghasilkan prediksi dan confidence yang sama.
    """
    samples_dir = Path(samples_dir)
    for s in samples:
        logits = onnx_logits(session, preprocess_from_config(Image.open(samples_dir / s["file"]), config)[None])[0]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        s["expected_prediction"] = classes[int(probs.argmax())]
        s["expected_confidence"] = round(float(probs.max()), 4)
    (samples_dir / "samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    return samples
