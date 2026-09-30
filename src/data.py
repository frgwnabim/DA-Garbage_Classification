"""Dataset, transform, split, dan dataloader untuk klasifikasi sampah."""

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
# Mean dan std ImageNet, dipakai karena model pretrained dilatih dengan normalisasi ini
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def set_seed(seed=SEED):
    """Kunci semua sumber acak supaya hasil bisa direproduksi."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN deterministik: sedikit lebih lambat, tapi hasil training bisa diulang
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_classes(data_dir):
    """Nama kelas = nama subfolder di data_dir, diurutkan alfabetis."""
    return sorted(p.name for p in Path(data_dir).iterdir() if p.is_dir())


def list_images(data_dir):
    """Kembalikan daftar (path relatif terhadap data_dir, nama kelas)."""
    data_dir = Path(data_dir)
    items = []
    for cls in get_classes(data_dir):
        for f in sorted((data_dir / cls).iterdir()):
            if f.suffix.lower() in IMAGE_EXTS:
                items.append((f"{cls}/{f.name}", cls))
    return items


def find_cross_class_duplicates(data_dir, items):
    """Cari file yang isinya identik (hash MD5 sama) tapi labelnya berbeda."""
    by_hash = defaultdict(list)
    for rel, cls in items:
        digest = hashlib.md5((Path(data_dir) / rel).read_bytes()).hexdigest()
        by_hash[digest].append((rel, cls))
    return [group for group in by_hash.values() if len({c for _, c in group}) > 1]


def remove_items(items, paths_to_remove):
    drop = set(paths_to_remove)
    return [(rel, cls) for rel, cls in items if rel not in drop]


def make_splits(items, val_size=0.15, test_size=0.15, seed=SEED):
    """Split stratified train/val/test. Stratified menjaga proporsi kelas di tiap split."""
    paths = [rel for rel, _ in items]
    labels = [cls for _, cls in items]
    train_p, temp_p, train_y, temp_y = train_test_split(
        paths, labels, test_size=val_size + test_size, stratify=labels, random_state=seed
    )
    rel_test = test_size / (val_size + test_size)
    val_p, test_p, val_y, test_y = train_test_split(
        temp_p, temp_y, test_size=rel_test, stratify=temp_y, random_state=seed
    )
    return {
        "train": sorted(zip(train_p, train_y)),
        "val": sorted(zip(val_p, val_y)),
        "test": sorted(zip(test_p, test_y)),
    }


def save_splits(splits, classes, path, extra=None):
    payload = {
        "seed": SEED,
        "classes": classes,
        "splits": {k: [{"path": p, "label": c} for p, c in v] for k, v in splits.items()},
    }
    if extra:
        payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_splits(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    splits = {k: [(d["path"], d["label"]) for d in v] for k, v in payload["splits"].items()}
    return splits, payload["classes"]


def get_transforms(train):
    """Transform train memakai augmentasi, transform val/test hanya resize dan normalisasi."""
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        normalize,
    ])


def denormalize(tensor):
    """Balikkan normalisasi ImageNet supaya tensor bisa ditampilkan sebagai gambar."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


class GarbageDataset(Dataset):
    def __init__(self, data_dir, items, classes, transform=None):
        self.data_dir = Path(data_dir)
        self.items = items
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rel, cls = self.items[idx]
        image = Image.open(self.data_dir / rel).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.class_to_idx[cls]


def compute_class_weights(items, classes):
    """Bobot kelas 'balanced': n_total / (n_kelas * n_c). Kelas yang jarang dapat bobot lebih besar."""
    counts = Counter(cls for _, cls in items)
    n_total, n_classes = len(items), len(classes)
    weights = [n_total / (n_classes * counts[c]) for c in classes]
    return torch.tensor(weights, dtype=torch.float32)


def make_dataloaders(data_dir, splits, classes, batch_size=BATCH_SIZE, num_workers=0, seed=SEED):
    """Buat DataLoader train/val/test.

    num_workers=0 default karena lebih stabil di notebook Windows. Kalau num_workers > 0,
    worker dibuat sekali lalu dipakai ulang (persistent_workers) supaya tidak ada biaya
    start ulang proses di setiap epoch.
    """
    generator = torch.Generator().manual_seed(seed)
    loaders = {}
    for name, items in splits.items():
        is_train = name == "train"
        ds = GarbageDataset(data_dir, items, classes, get_transforms(train=is_train))
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            generator=generator if is_train else None,
        )
    return loaders
