"""Salin model dan data hasil analisis ke site/ supaya website demo bisa dideploy sebagai static site.

Jalankan dari mana saja:  python scripts/build_site.py

Isi yang disalin:
- models/garbage_classifier.onnx, labels.json, preprocessing.json  ->  site/models/
- outputs/*.json (kecuali splits.json, daftar file split yang tidak dipakai website)  ->  site/data/

Contoh gambar di site/assets/samples/ dibuat oleh notebook (bagian Deployment), bukan oleh script ini,
karena butuh dataset asli yang tidak di-commit.
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILES = ["garbage_classifier.onnx", "labels.json", "preprocessing.json"]
EXCLUDED_OUTPUTS = {"splits.json"}


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  {src.relative_to(ROOT)}  ->  {dst.relative_to(ROOT)}  ({dst.stat().st_size / 1024:.0f} KB)")


def main():
    missing = [f for f in MODEL_FILES if not (ROOT / "models" / f).exists()]
    if missing:
        sys.exit(f"File model belum ada: {missing}. Jalankan bagian Deployment di notebook dulu.")

    print("Model:")
    for name in MODEL_FILES:
        copy(ROOT / "models" / name, ROOT / "site" / "models" / name)

    print("Data hasil analisis:")
    json_files = sorted(p for p in (ROOT / "outputs").glob("*.json") if p.name not in EXCLUDED_OUTPUTS)
    for src in json_files:
        copy(src, ROOT / "site" / "data" / src.name)

    samples = ROOT / "site" / "assets" / "samples"
    n_samples = len(list(samples.glob("*.jpg"))) if samples.exists() else 0
    print(f"Contoh gambar di {samples.relative_to(ROOT)}: {n_samples} file")
    print("Selesai.")


if __name__ == "__main__":
    main()
