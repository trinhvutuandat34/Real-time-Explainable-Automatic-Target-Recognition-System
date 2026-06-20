"""Dataset classes and DataLoader factories for cadet_atr."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from utils.config import CLASSES, cfg


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

_MEAN = [0.5, 0.5, 0.5]
_STD  = [0.5, 0.5, 0.5]

_train_tf = transforms.Compose([
    transforms.Resize((cfg.img_size, cfg.img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

_val_tf = transforms.Compose([
    transforms.Resize((cfg.img_size, cfg.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


# ---------------------------------------------------------------------------
# SyntheticIRDataset
# ---------------------------------------------------------------------------

class SyntheticIRDataset(Dataset):
    """
    Folder-based dataset of synthetic IR images.

    Expected layout:
        data/synthetic/
            fixed_wing/  *.png *.jpg
            rotary_wing/ ...
            uav/         ...
            vessel/      ...
            vehicle_ground/ ...
            vehicle_apc/ ...
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        val_fraction: float = 0.15,
        seed: int = 42,
        transform=None,
    ):
        self.root      = Path(root)
        self.transform = transform or (_train_tf if split == "train" else _val_tf)
        self.samples: list[Tuple[Path, int]] = []

        rng = random.Random(seed)
        for cls_idx, cls_name in enumerate(CLASSES):
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                continue
            files = sorted(cls_dir.glob("*.*"))
            files = [f for f in files if f.suffix.lower() in {".png", ".jpg", ".jpeg"}]
            rng.shuffle(files)
            n_val = max(1, int(len(files) * val_fraction))
            if split == "val":
                chosen = files[:n_val]
            else:
                chosen = files[n_val:]
            self.samples.extend((f, cls_idx) for f in chosen)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


# ---------------------------------------------------------------------------
# Synthetic-only loaders
# ---------------------------------------------------------------------------

def make_loaders(
    data_root: Optional[str] = None,
    batch_size: Optional[int] = None,
    workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) from synthetic data."""
    root  = Path(data_root or cfg.data_root) / "synthetic"
    bs    = batch_size or cfg.batch_size

    train_ds = SyntheticIRDataset(root, split="train", seed=seed)
    val_ds   = SyntheticIRDataset(root, split="val",   seed=seed)

    _pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=workers, pin_memory=_pin, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=workers, pin_memory=_pin)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Real FLIR loader
# ---------------------------------------------------------------------------

def make_real_loader(
    real_root: Optional[str] = None,
    batch_size: Optional[int] = None,
    workers: int = 2,
    augment: bool = False,
) -> DataLoader:
    """
    Loader for real FLIR images.  Expects ImageFolder layout:
        <real_root>/<class_name>/<image_files>
    where class names match CLASSES.
    """
    root = Path(real_root or (Path(cfg.data_root) / "real"))
    tf   = _train_tf if augment else _val_tf
    bs   = batch_size or max(4, cfg.batch_size // 4)

    ds = datasets.ImageFolder(str(root), transform=tf)
    return DataLoader(ds, batch_size=bs, shuffle=augment,
                      num_workers=workers, pin_memory=torch.cuda.is_available())


# ---------------------------------------------------------------------------
# Synthetic fallback: generate placeholder images
# ---------------------------------------------------------------------------

def generate_placeholder_synthetic(data_root: str, n_per_class: int = 40) -> None:
    """
    Create tiny synthetic IR-like placeholder images (Gaussian blobs on noise)
    so the codebase can run without real data or Stable Diffusion.
    """
    import cv2 as cv

    root = Path(data_root) / "synthetic"
    rng  = np.random.default_rng(42)

    for cls_name in CLASSES:
        cls_dir = root / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        existing = list(cls_dir.glob("*.png"))
        if len(existing) >= n_per_class:
            continue
        for i in range(len(existing), n_per_class):
            base = (rng.standard_normal((224, 224)) * 20 + 30).clip(0, 255)
            # add a bright Gaussian blob to simulate a target
            cx = rng.integers(60, 164); cy = rng.integers(60, 164)
            for dy in range(-20, 21):
                for dx in range(-20, 21):
                    r = np.sqrt(dx**2 + dy**2)
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < 224 and 0 <= nx < 224:
                        val = 200.0 * np.exp(-r**2 / 80)
                        base[ny, nx] = min(255.0, base[ny, nx] + val)
            img = base.clip(0, 255).astype(np.uint8)
            cv.imwrite(str(cls_dir / f"placeholder_{i:04d}.png"), img)

    print(f"[dataset] Placeholder synthetic data written to {root}")
