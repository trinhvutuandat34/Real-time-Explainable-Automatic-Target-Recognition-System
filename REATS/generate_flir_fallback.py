#!/usr/bin/env python3
"""
REATS Synthetic / FLIR-ADAS Fallback Generator
Maps FLIR ADAS thermal images → 6 REATS classes to seed the dataset
while the diffusion model pipeline is being set up.

Two modes
---------
crop        : crop annotated FLIR bboxes, intensity-remap the ROI
background  : use FLIR background texture + synthetic target blob overlay
both        : run both modes (default)
synth-only  : ignore FLIR entirely; generate purely synthetic IR patches

FLIR ADAS structure expected
----------------------------
<flir_root>/
  images/
    train/  *.jpg  (thermal 8-bit)
    val/    *.jpg
  video_thermal_train/  (optional — same images, ignored)
  annotations/
    train.json  (COCO JSON)
    val.json    (COCO JSON)

Download (Kaggle):
  kaggle datasets download -d deepnewbie/flir-thermal-images-dataset

Usage
-----
  python generate_flir_fallback.py --flir /path/to/flir_adas/ --out data/
  python generate_flir_fallback.py --flir /path/to/flir_adas/ --out data/ --mode crop
  python generate_flir_fallback.py --out data/ --mode synth-only
  python generate_flir_fallback.py --flir /path/to/flir_adas/ --out data/ --dry-run
"""

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASSES = ["F16", "LYNX", "MiG19", "MiG21", "PKG", "PTG"]
SPLITS  = ["train", "val", "test"]
TARGETS = {"train": 170, "val": 30, "test": 200}

IMG_SIZE = 224  # output patch size (square)

# FLIR ADAS v1 category names (lower-case)
_FLIR_V1 = {"person", "car", "bicycle", "other_vehicle"}
_FLIR_V2 = {"motorcycle", "bus", "truck", "scooter"}

# Thermal intensity targets per REATS class (mean, std)  [0-255 8-bit]
_THERMAL = {
    "F16":   (205, 25),
    "LYNX":  (178, 20),
    "MiG19": (198, 28),
    "MiG21": (192, 26),
    "PKG":   (158, 18),
    "PTG":   (152, 20),
}

# Target blob geometry (height, width) in a 224×224 patch
_BLOB_HW = {
    "F16":   (28, 68),
    "LYNX":  (48, 48),
    "MiG19": (24, 62),
    "MiG21": (26, 56),
    "PKG":   (42, 88),
    "PTG":   (36, 78),
}

# ---------------------------------------------------------------------------
# FLIR category → REATS class mapping
# Mapping logic:
#   person              → LYNX  (infantry/vehicle silhouette, upright)
#   bicycle  small/med  → MiG21 (fast-movers, narrow silhouette)
#   bicycle  large      → F16   (wide-wing signature)
#   car      small      → MiG19 (compact airframe)
#   car      medium     → PTG   (patrol torpedo gunboat)
#   car      large      → PKG   (patrol killer guided)
#   other_vehicle       → PKG   (large unclassified vehicle)
#   motorcycle/scooter  → MiG21
#   bus/truck           → PKG
# Size buckets based on bbox area (pixels in original FLIR 640×512 image)
# ---------------------------------------------------------------------------

_SMALL_AREA  = 32 * 32
_MEDIUM_AREA = 64 * 64


def _flir_category_to_reats(cat_name: str, bbox_area: float) -> str | None:
    """Return REATS class or None if unmappable."""
    cat = cat_name.lower().strip()
    if cat == "person":
        return "LYNX"
    if cat == "bicycle":
        if bbox_area < _SMALL_AREA:
            return "MiG21"
        if bbox_area < _MEDIUM_AREA:
            return "MiG21"
        return "F16"
    if cat == "car":
        if bbox_area < _SMALL_AREA:
            return "MiG19"
        if bbox_area < _MEDIUM_AREA:
            return "PTG"
        return "PKG"
    if cat in ("other_vehicle", "truck", "bus"):
        return "PKG"
    if cat in ("motorcycle", "scooter"):
        return "MiG21"
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_existing(data_root: Path) -> dict:
    """Count images already in data_root/{split}/{class}/."""
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    counts = {s: {c: 0 for c in CLASSES} for s in SPLITS}
    for split in SPLITS:
        for cls in CLASSES:
            d = data_root / split / cls
            if d.exists():
                counts[split][cls] = sum(
                    1 for p in d.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMG_EXTS
                )
    return counts


def _shortfall(existing: dict) -> dict:
    """Return {split: {cls: n_needed}}."""
    need = {}
    for split in SPLITS:
        need[split] = {}
        for cls in CLASSES:
            n = max(0, TARGETS[split] - existing[split][cls])
            if n:
                need[split][cls] = n
    return need


def _next_idx(data_root: Path, split: str, cls: str) -> int:
    """Return next file index for {cls}_{idx:05d}.png naming."""
    d = data_root / split / cls
    if not d.exists():
        return 0
    idxs = []
    for p in d.iterdir():
        stem = p.stem  # e.g. "F16_00042"
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            idxs.append(int(parts[1]))
    return max(idxs, default=-1) + 1


def _save(patch: np.ndarray, dst: Path, dry_run: bool) -> None:
    """Save uint8 greyscale patch as PNG."""
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Convert greyscale → RGB so the classifier sees 3-channel input
    rgb = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(dst), rgb)


# ---------------------------------------------------------------------------
# Intensity remapping
# ---------------------------------------------------------------------------

def intensity_remap(gray: np.ndarray, target_mean: float, target_std: float,
                    clip: bool = True) -> np.ndarray:
    """Linearly remap pixel intensities so the patch matches the target class signature."""
    src = gray.astype(np.float32)
    m, s = src.mean(), src.std() + 1e-6
    out = (src - m) / s * target_std + target_mean
    if clip:
        out = np.clip(out, 0, 255)
    return out.astype(np.uint8)


# ---------------------------------------------------------------------------
# Blob overlay (synthetic target on real/synthetic background)
# ---------------------------------------------------------------------------

def _make_blob(cls: str, rng: np.random.Generator) -> np.ndarray:
    """Generate a hot target blob (H×W) matching class thermal profile."""
    h, w = _BLOB_HW[cls]
    mean, std = _THERMAL[cls]
    # Gaussian blob with slight noise
    blob = rng.normal(mean, std, (h, w)).clip(0, 255).astype(np.uint8)
    # Soft edges via Gaussian blur
    ksize = max(3, min(h, w) // 3 | 1)
    blob = cv2.GaussianBlur(blob, (ksize, ksize), 0)
    return blob


def add_blob(background: np.ndarray, cls: str,
             rng: np.random.Generator) -> np.ndarray:
    """Paste a synthetic target blob onto a 224×224 greyscale background."""
    out = background.astype(np.float32)
    bh, bw = _BLOB_HW[cls]
    max_y = IMG_SIZE - bh
    max_x = IMG_SIZE - bw
    if max_y <= 0 or max_x <= 0:
        return background.copy()
    y = int(rng.integers(max_y // 4, max(max_y // 4 + 1, 3 * max_y // 4)))
    x = int(rng.integers(max_x // 4, max(max_x // 4 + 1, 3 * max_x // 4)))
    blob = _make_blob(cls, rng).astype(np.float32)
    # Alpha blend: blob dominates centre, fades at edges
    alpha = np.ones((bh, bw), np.float32) * 0.85
    out[y:y+bh, x:x+bw] = alpha * blob + (1 - alpha) * out[y:y+bh, x:x+bw]
    return out.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Purely synthetic IR patch (no FLIR needed)
# ---------------------------------------------------------------------------

def _synth_background(cls: str, rng: np.random.Generator) -> np.ndarray:
    """Generate a plausible IR background for the given class."""
    bg_mean = max(60, _THERMAL[cls][0] - 80)
    bg_std  = 15
    bg = rng.normal(bg_mean, bg_std, (IMG_SIZE, IMG_SIZE)).clip(0, 255).astype(np.uint8)
    # Add low-freq texture via blur + subtract sharpened version
    blurred = cv2.GaussianBlur(bg.astype(np.float32), (21, 21), 0)
    texture = bg.astype(np.float32) * 0.6 + blurred * 0.4
    return texture.clip(0, 255).astype(np.uint8)


def generate_synth_patch(cls: str, rng: np.random.Generator) -> np.ndarray:
    """Return a 224×224 uint8 greyscale synthetic IR patch for cls."""
    bg = _synth_background(cls, rng)
    return add_blob(bg, cls, rng)


# ---------------------------------------------------------------------------
# FLIR loading
# ---------------------------------------------------------------------------

def load_flir_annotations(ann_path: Path) -> dict:
    """
    Parse COCO JSON → {image_id: {"file_name": ..., "annotations": [...]}}
    Each annotation: {"bbox": [x,y,w,h], "category_name": str}
    """
    with open(ann_path) as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}

    img_map = {}
    for img in coco.get("images", []):
        img_map[img["id"]] = {"file_name": img["file_name"], "annotations": []}

    for ann in coco.get("annotations", []):
        iid = ann["image_id"]
        if iid not in img_map:
            continue
        cat_name = cat_id_to_name.get(ann.get("category_id"), "")
        img_map[iid]["annotations"].append({
            "bbox":          ann["bbox"],  # [x, y, w, h]
            "category_name": cat_name,
        })

    return img_map


def _find_flir_image(flir_root: Path, split: str, file_name: str) -> Path | None:
    """Locate a FLIR image file; tries multiple sub-paths."""
    candidates = [
        flir_root / "images" / split / file_name,
        flir_root / "images" / split / Path(file_name).name,
        flir_root / split / file_name,
        flir_root / file_name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


# ---------------------------------------------------------------------------
# Crop mode
# ---------------------------------------------------------------------------

def generate_crop_mode(
    flir_root:  Path,
    data_root:  Path,
    need:       dict,
    rng:        np.random.Generator,
    dry_run:    bool,
    verbose:    bool,
) -> int:
    """Crop FLIR bboxes, intensity-remap, resize to 224×224, save."""
    total = 0
    flir_splits = [("train", "train"), ("val", "val")]

    # Build pool: {reats_class: [(flir_path, x, y, w, h), ...]}
    pool: dict[str, list] = defaultdict(list)

    for flir_split, _ in flir_splits:
        ann_path = flir_root / "annotations" / f"{flir_split}.json"
        if not ann_path.exists():
            if verbose:
                print(f"  [crop] annotation not found: {ann_path} — skipping")
            continue

        img_map = load_flir_annotations(ann_path)

        for iid, info in img_map.items():
            fpath = _find_flir_image(flir_root, flir_split, info["file_name"])
            if fpath is None:
                continue
            for ann in info["annotations"]:
                x, y, w, h = [int(v) for v in ann["bbox"]]
                if w < 8 or h < 8:
                    continue
                area = w * h
                reats_cls = _flir_category_to_reats(ann["category_name"], area)
                if reats_cls is None:
                    continue
                pool[reats_cls].append((fpath, x, y, w, h))

    if verbose:
        print(f"  [crop] pool sizes: " +
              ", ".join(f"{c}={len(pool[c])}" for c in CLASSES))

    for split in SPLITS:
        for cls in CLASSES:
            n = need[split].get(cls, 0)
            if n == 0:
                continue
            items = pool.get(cls, [])
            if not items:
                if verbose:
                    print(f"  [crop] {split}/{cls}: no pool items, skipping {n}")
                continue

            idx = _next_idx(data_root, split, cls)
            generated = 0
            attempts = 0
            max_attempts = n * 10

            while generated < n and attempts < max_attempts:
                attempts += 1
                fpath, bx, by, bw, bh = items[int(rng.integers(len(items)))]
                try:
                    img = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    ih, iw = img.shape
                    # Clamp bbox to image
                    x1 = max(0, bx)
                    y1 = max(0, by)
                    x2 = min(iw, bx + bw)
                    y2 = min(ih, by + bh)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = img[y1:y2, x1:x2]
                    # Slight random jitter (±10%)
                    jy = int(rng.integers(-max(1, (y2-y1)//10),
                                          max(1, (y2-y1)//10) + 1))
                    jx = int(rng.integers(-max(1, (x2-x1)//10),
                                          max(1, (x2-x1)//10) + 1))
                    ny1 = max(0, y1+jy); ny2 = min(ih, y2+jy)
                    nx1 = max(0, x1+jx); nx2 = min(iw, x2+jx)
                    crop = img[ny1:ny2, nx1:nx2]
                    if crop.size == 0:
                        continue

                    patch = cv2.resize(crop, (IMG_SIZE, IMG_SIZE),
                                       interpolation=cv2.INTER_LINEAR)
                    m, s = _THERMAL[cls]
                    patch = intensity_remap(patch, m, s)
                    dst = data_root / split / cls / f"{cls}_{idx:05d}.png"
                    _save(patch, dst, dry_run)
                    idx += 1
                    generated += 1
                    total += 1
                except Exception:
                    continue

            if verbose:
                verb = "Would generate" if dry_run else "Generated"
                print(f"  [crop] {split}/{cls}: {verb} {generated}/{n}")

    return total


# ---------------------------------------------------------------------------
# Background mode
# ---------------------------------------------------------------------------

def generate_background_mode(
    flir_root:  Path | None,
    data_root:  Path,
    need:       dict,
    rng:        np.random.Generator,
    dry_run:    bool,
    verbose:    bool,
) -> int:
    """Use FLIR images as backgrounds (or synth if no FLIR), add target blob."""
    total = 0

    # Collect background images
    bg_pool: list[Path] = []
    if flir_root is not None:
        for sub in ["images/train", "images/val", "train", "val"]:
            d = flir_root / sub
            if d.exists():
                bg_pool.extend(p for p in d.rglob("*.jpg"))
                bg_pool.extend(p for p in d.rglob("*.png"))

    use_real_bg = len(bg_pool) > 0
    if verbose:
        src = f"{len(bg_pool)} FLIR images" if use_real_bg else "synthetic backgrounds"
        print(f"  [bg] background source: {src}")

    for split in SPLITS:
        for cls in CLASSES:
            n = need[split].get(cls, 0)
            if n == 0:
                continue

            idx = _next_idx(data_root, split, cls)
            generated = 0
            attempts = 0
            max_attempts = n * 5

            while generated < n and attempts < max_attempts:
                attempts += 1
                try:
                    if use_real_bg:
                        bg_path = bg_pool[int(rng.integers(len(bg_pool)))]
                        img = cv2.imread(str(bg_path), cv2.IMREAD_GRAYSCALE)
                        if img is None:
                            continue
                        # Random 224×224 crop from FLIR image
                        ih, iw = img.shape
                        if ih < IMG_SIZE or iw < IMG_SIZE:
                            bg = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        else:
                            ry = int(rng.integers(0, ih - IMG_SIZE + 1))
                            rx = int(rng.integers(0, iw - IMG_SIZE + 1))
                            bg = img[ry:ry+IMG_SIZE, rx:rx+IMG_SIZE]
                        # Suppress bright objects from background before overlay
                        bg = np.where(bg > 200, bg - 40, bg).clip(0, 255).astype(np.uint8)
                    else:
                        bg = _synth_background(cls, rng)

                    patch = add_blob(bg, cls, rng)
                    dst = data_root / split / cls / f"{cls}_{idx:05d}.png"
                    _save(patch, dst, dry_run)
                    idx += 1
                    generated += 1
                    total += 1
                except Exception:
                    continue

            if verbose:
                verb = "Would generate" if dry_run else "Generated"
                print(f"  [bg ] {split}/{cls}: {verb} {generated}/{n}")

    return total


# ---------------------------------------------------------------------------
# Synth-only mode
# ---------------------------------------------------------------------------

def generate_synth_only(
    data_root: Path,
    need:      dict,
    rng:       np.random.Generator,
    dry_run:   bool,
    verbose:   bool,
) -> int:
    """Pure synthetic IR patches, no FLIR required."""
    total = 0
    for split in SPLITS:
        for cls in CLASSES:
            n = need[split].get(cls, 0)
            if n == 0:
                continue
            idx = _next_idx(data_root, split, cls)
            for i in range(n):
                patch = generate_synth_patch(cls, rng)
                dst = data_root / split / cls / f"{cls}_{idx:05d}.png"
                _save(patch, dst, dry_run)
                idx += 1
                total += 1
            if verbose:
                verb = "Would generate" if dry_run else "Generated"
                print(f"  [synth] {split}/{cls}: {verb} {n}")
    return total


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def _print_need(need: dict) -> None:
    W = 58
    print(f"\n{'─'*W}")
    print("  Shortfall to fill:")
    any_need = False
    for split in SPLITS:
        for cls in CLASSES:
            n = need[split].get(cls, 0)
            if n:
                print(f"    {split}/{cls:<8} → {n} images needed")
                any_need = True
    if not any_need:
        print("    (none — dataset already complete)")
    print(f"{'─'*W}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="REATS Synthetic / FLIR-ADAS Fallback Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  crop        crop FLIR bboxes, intensity-remap (requires --flir)
  background  FLIR texture + synthetic target blob overlay
  both        run crop first, then background to fill remaining (default)
  synth-only  purely synthetic; no --flir needed

Examples:
  python generate_flir_fallback.py --flir /data/flir_adas/ --out data/
  python generate_flir_fallback.py --flir /data/flir_adas/ --out data/ --mode crop
  python generate_flir_fallback.py --out data/ --mode synth-only
  python generate_flir_fallback.py --flir /data/flir_adas/ --out data/ --dry-run

Kaggle download:
  kaggle datasets download -d deepnewbie/flir-thermal-images-dataset
  unzip flir-thermal-images-dataset.zip -d /kaggle/working/flir_adas/
""",
    )
    p.add_argument("--flir",    default=None,
                   help="path to FLIR ADAS dataset root (optional for synth-only)")
    p.add_argument("--out",     default="data/",
                   help="output data root (default: data/)")
    p.add_argument("--mode",    default="both",
                   choices=["crop", "background", "both", "synth-only"],
                   help="generation strategy (default: both)")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be generated without writing files")
    p.add_argument("--seed",    type=int, default=42,
                   help="random seed (default: 42)")
    p.add_argument("--quiet",   action="store_true",
                   help="suppress per-class output")
    args = p.parse_args()

    verbose   = not args.quiet
    data_root = Path(args.out)
    flir_root = Path(args.flir) if args.flir else None

    W = 58
    print(f"\n{'='*W}")
    print("  REATS Synthetic / FLIR Fallback Generator")
    print(f"  Mode    : {args.mode}")
    print(f"  Output  : {data_root.resolve()}")
    if flir_root:
        print(f"  FLIR    : {flir_root.resolve()}")
    if args.dry_run:
        print(f"  *** DRY RUN — no files will be written ***")
    print(f"{'='*W}")

    # Validate FLIR path
    if args.mode in ("crop", "both") and flir_root is None:
        print(f"\n  WARNING: --mode {args.mode} uses FLIR crops but --flir not provided.")
        print(  "  Falling back to background + synth modes.\n")
        if args.mode == "crop":
            args.mode = "synth-only"
        else:
            args.mode = "background"

    if flir_root and not flir_root.exists():
        print(f"\n  ERROR: FLIR root not found: {flir_root}")
        print(  "\n  Download instructions:")
        print(  "    kaggle datasets download -d deepnewbie/flir-thermal-images-dataset")
        print(  "    unzip flir-thermal-images-dataset.zip -d ./flir_adas/")
        print(  "\n  Falling back to synth-only mode.\n")
        flir_root = None
        if args.mode in ("crop", "both"):
            args.mode = "synth-only"

    rng = np.random.default_rng(args.seed)

    # Count existing + compute shortfall
    existing = _count_existing(data_root)
    need     = _shortfall(existing)
    _print_need(need)

    total_needed = sum(n for s in need.values() for n in s.values())
    if total_needed == 0:
        print("  Dataset already complete. Nothing to do.\n")
        return

    total_generated = 0

    if args.mode == "synth-only":
        total_generated += generate_synth_only(data_root, need, rng,
                                               args.dry_run, verbose)

    elif args.mode == "crop":
        total_generated += generate_crop_mode(flir_root, data_root, need, rng,
                                              args.dry_run, verbose)

    elif args.mode == "background":
        total_generated += generate_background_mode(flir_root, data_root, need, rng,
                                                    args.dry_run, verbose)

    elif args.mode == "both":
        # Pass 1: crop mode
        print("\n  Pass 1/2 — crop mode")
        total_generated += generate_crop_mode(flir_root, data_root, need, rng,
                                              args.dry_run, verbose)
        # Recalculate shortfall
        if not args.dry_run:
            existing = _count_existing(data_root)
            need     = _shortfall(existing)
        # Pass 2: background mode fills the rest
        remaining = sum(n for s in need.values() for n in s.values())
        if remaining:
            print(f"\n  Pass 2/2 — background mode ({remaining} still needed)")
            total_generated += generate_background_mode(flir_root, data_root, need, rng,
                                                        args.dry_run, verbose)

    verb = "Would generate" if args.dry_run else "Generated"
    print(f"\n{'='*W}")
    print(f"  {verb} {total_generated} images total.")

    if not args.dry_run and total_generated > 0:
        print(f"\n  Run dataset_validator.py to verify the result:")
        print(f"    python dataset_validator.py --root {data_root}")
    print(f"{'='*W}\n")


if __name__ == "__main__":
    main()
