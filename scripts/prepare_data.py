"""Rapikan dataset hasil download Kaggle menjadi struktur data/<kelas>/.

Dataset Kaggle "Garbage classification" berisi folder bertingkat
("Garbage classification/Garbage classification/<kelas>/"). Script ini mencari folder
keenam kelas di mana pun di dalam folder sumber, lalu menyalin gambarnya ke data/<kelas>/.

Contoh:
    python scripts/prepare_data.py path/ke/hasil-ekstrak-zip
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def find_class_dirs(source):
    """Folder bernama kelas yang benar-benar berisi gambar (bukan folder kosong)."""
    found = {}
    for cls in CLASSES:
        candidates = [
            d for d in source.rglob(cls)
            if d.is_dir() and any(f.suffix.lower() in IMAGE_EXTS for f in d.iterdir())
        ]
        if not candidates:
            sys.exit(f"Folder kelas '{cls}' tidak ditemukan di {source}")
        # Kalau ada lebih dari satu (misalnya salinan ganda di zip), ambil yang gambarnya paling banyak
        found[cls] = max(candidates, key=lambda d: sum(1 for _ in d.iterdir()))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="folder hasil ekstrak dataset Kaggle")
    parser.add_argument("--dest", type=Path, default=ROOT / "data", help="folder tujuan (default: data/)")
    args = parser.parse_args()

    if not args.source.is_dir():
        sys.exit(f"Folder sumber tidak ada: {args.source}")
    class_dirs = find_class_dirs(args.source)

    total = 0
    for cls, src in class_dirs.items():
        dst = args.dest / cls
        dst.mkdir(parents=True, exist_ok=True)
        files = sorted(f for f in src.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        for f in files:
            shutil.copy2(f, dst / f.name)
        total += len(files)
        print(f"  {cls:10s} {len(files):5d} gambar  <-  {src}")
    print(f"Total {total} gambar disalin ke {args.dest}")
    if total != 2527:
        print("Peringatan: dataset asli berisi 2.527 gambar. Pastikan file yang di-download sudah benar.")


if __name__ == "__main__":
    main()
