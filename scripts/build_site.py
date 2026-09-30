"""Salin model dan data hasil analisis ke site/ supaya website demo bisa dideploy sebagai static site.

Jalankan dari mana saja:  python scripts/build_site.py

Isi yang disiapkan:
- models/garbage_classifier.onnx, labels.json, preprocessing.json  ->  site/models/
- outputs/*.json (kecuali splits.json, daftar file split yang tidak dipakai website)  ->  site/data/
- ringkasan dataset (jumlah gambar per kelas per split) dari outputs/splits.json  ->  site/data/dataset_summary.json
- outputs/*.png (chart dan contoh Grad-CAM)  ->  site/assets/img/*.jpg, diperkecil supaya ringan dimuat

Contoh gambar di site/assets/samples/ dibuat oleh notebook (bagian Deployment), bukan oleh script ini,
karena butuh dataset asli yang tidak di-commit.
"""

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
MODEL_FILES = ["garbage_classifier.onnx", "labels.json", "preprocessing.json"]
EXCLUDED_OUTPUTS = {"splits.json"}
IMAGE_MAX_WIDTH = 1600
IMAGE_QUALITY = 82


def report(src, dst):
    print(f"  {src.relative_to(ROOT)}  ->  {dst.relative_to(ROOT)}  ({dst.stat().st_size / 1024:.0f} KB)")


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    report(src, dst)


def png_to_jpg(src, dst):
    """Konversi PNG ke JPEG (latar putih), lebar maksimal IMAGE_MAX_WIDTH."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, "white")
        background.paste(img, mask=img.split()[-1])
        img = background
    img = img.convert("RGB")
    if img.width > IMAGE_MAX_WIDTH:
        img = img.resize((IMAGE_MAX_WIDTH, round(img.height * IMAGE_MAX_WIDTH / img.width)), Image.LANCZOS)
    img.save(dst, "JPEG", quality=IMAGE_QUALITY, optimize=True, progressive=True)
    report(src, dst)


def dataset_summary(splits_path):
    """Jumlah gambar per kelas per split, tanpa daftar nama file."""
    payload = json.loads(splits_path.read_text(encoding="utf-8"))
    classes = payload["classes"]
    per_split = {
        name: {c: Counter(d["label"] for d in items)[c] for c in classes}
        for name, items in payload["splits"].items()
    }
    removed = payload.get("removed_duplicates", [])
    total = sum(sum(v.values()) for v in per_split.values())
    return {
        "classes": classes,
        "per_split": per_split,
        "per_class": {c: sum(per_split[s][c] for s in per_split) for c in classes},
        "total_after_cleaning": total,
        "total_original": total + len(removed),
        "removed_duplicates": removed,
        "image_size": [512, 384],
        "seed": payload.get("seed"),
    }


def main():
    missing = [f for f in MODEL_FILES if not (ROOT / "models" / f).exists()]
    if missing:
        sys.exit(f"File model belum ada: {missing}. Jalankan bagian Deployment di notebook dulu.")

    print("Model:")
    for name in MODEL_FILES:
        copy(ROOT / "models" / name, SITE / "models" / name)

    print("Data hasil analisis:")
    json_files = sorted(p for p in (ROOT / "outputs").glob("*.json") if p.name not in EXCLUDED_OUTPUTS)
    for src in json_files:
        copy(src, SITE / "data" / src.name)
    summary_path = SITE / "data" / "dataset_summary.json"
    summary_path.write_text(json.dumps(dataset_summary(ROOT / "outputs" / "splits.json"), indent=2), encoding="utf-8")
    report(ROOT / "outputs" / "splits.json", summary_path)

    print("Gambar:")
    for src in sorted((ROOT / "outputs").glob("*.png")):
        png_to_jpg(src, SITE / "assets" / "img" / f"{src.stem}.jpg")

    samples = SITE / "assets" / "samples"
    n_samples = len(list(samples.glob("*.jpg"))) if samples.exists() else 0
    print(f"Contoh gambar di {samples.relative_to(ROOT)}: {n_samples} file")
    print("Selesai.")


if __name__ == "__main__":
    main()
