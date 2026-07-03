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
# Constants — loaded from config (single source of truth)
# ---------------------------------------------------------------------------

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent))
from config import CLASSES, NUM_CLASSES, TARGET_META
from ingestion.preprocessor import _CLASS_THERMAL as _THERMAL


def _domain_of(cls: str) -> str:
    return TARGET_META.get(cls, {}).get("domain", "GROUND")

SPLITS  = ["train", "val", "test"]
TARGETS = {"train": 170, "val": 30, "test": 200}

IMG_SIZE = 224  # output patch size (square)

# Target blob geometry (height, width) in a 224×224 patch — top-down UAV view
_BLOB_HW = {
    # AIR fighters (narrow fuselage, wide wingspan from above)
    "F16": (20, 60), "F15": (22, 64), "F22": (24, 52), "F35": (18, 52),
    "Su27": (24, 72), "Su35": (24, 70), "MiG29": (22, 60),
    "MiG19": (18, 56), "MiG21": (16, 52), "J20": (24, 68),
    # AIR bombers (large wingspan)
    "B52": (26, 110), "Tu22M": (22, 90), "Tu95": (28, 100),
    # AIR attack helicopters (rotor disc visible from above)
    "AH64": (44, 52), "Mi24": (44, 54), "Ka52": (46, 50),
    # AIR transport helicopters
    "LYNX": (40, 48), "UH60": (44, 52), "CH47": (32, 78),
    # AIR UAVs (small, narrow)
    "MQ9": (18, 50), "TB2": (22, 50), "Shahed136": (14, 36),
    "RQ4": (28, 78), "WZ7": (22, 54),
    # GROUND MBT (boxy from above)
    "M1Abrams": (26, 42), "T72": (22, 36), "T90": (24, 38), "Leopard2": (24, 38),
    # GROUND IFV/APC (smaller)
    "BMP2": (18, 32), "Bradley": (18, 32), "BTR80": (16, 28), "K21": (18, 32),
    # GROUND artillery
    "M109": (18, 36), "BM21": (16, 48),
    # GROUND air defense
    "Patriot": (14, 26), "Buk": (16, 28), "Pantsir": (16, 26),
    # NAVAL (elongated hull, top-down)
    "PKG": (26, 88), "PTG": (22, 78), "FastAttack": (20, 68),
    "Destroyer": (30, 150), "Frigate": (26, 120), "Corvette": (24, 96),
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
    """Map FLIR ADAS civilian categories to the nearest REATS military class."""
    cat = cat_name.lower().strip()
    # Ground vehicles — map to wheeled/tracked vehicle classes by size
    if cat in ("car",):
        return "BTR80"                     # small wheeled vehicle
    if cat in ("truck", "bus", "other_vehicle", "van"):
        return "M109"                      # large ground vehicle
    if cat in ("motorcycle", "scooter", "bicycle"):
        return None                        # no good military equivalent
    if cat == "person":
        return None                        # not in taxonomy
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


def _merge_provenance(data_root: Path, per_cls: dict, bucket: str,
                      dry_run: bool) -> None:
    """Record generated image counts by data-provenance bucket.

    Shares data/provenance.json with the ingestion pipeline (which writes
    the 'real' bucket). Buckets:
      real      — genuine target pixels from an annotated dataset
      remapped  — real FLIR ROI, intensity-remapped to a class signature
      synthetic — procedurally generated target (blob/shape composite)
    Downstream evaluation reads this to report metrics per provenance —
    'real' accuracy is field-relevant; 'synthetic' is architecture validation.
    """
    if dry_run or not per_cls:
        return
    path = data_root / "provenance.json"
    prov: dict = {}
    if path.exists():
        try:
            prov = json.loads(path.read_text())
        except Exception:
            prov = {}
    for cls, n in per_cls.items():
        if n:
            entry = prov.setdefault(cls, {})
            entry[bucket] = entry.get(bucket, 0) + int(n)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prov, indent=2, sort_keys=True))


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

def _silhouette(cls: str, h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Soft [0,1] silhouette mask shaped by domain (top-down view).

    A shaped mask means the classifier must key on *structure*, not just a
    bright rectangle of the right aspect ratio — which is what made the old
    synthetic set trivially separable and inflated the accuracy number.
    """
    dom    = _domain_of(cls)
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ny = (yy - cy) / (h / 2.0 + 1e-6)          # (h,1) in ~[-1,1]
    nx = (xx - cx) / (w / 2.0 + 1e-6)          # (1,w) in ~[-1,1]

    if dom == "AIR":
        # fuselage (thin, long) + wings (thin, wide) → cross/plus footprint
        fuse = (np.abs(nx) < 0.22) & (np.abs(ny) < 0.98)
        wing = (np.abs(ny) < 0.24) & (np.abs(nx) < 0.98)
        mask = (fuse | wing).astype(np.float32)
    elif dom == "NAVAL":
        # elongated hull with a tapered bow (narrower toward +x)
        taper = 0.45 * (1.0 - 0.55 * np.clip(nx, 0.0, 1.0))
        mask  = ((nx ** 2) + (ny / (taper + 1e-3)) ** 2 < 1.0).astype(np.float32)
    else:  # GROUND — boxy, rounded corners (super-ellipse)
        mask = ((np.abs(nx) ** 4 + np.abs(ny) ** 4) < 1.0).astype(np.float32)

    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(h, w) / 22.0)
    return mask / (mask.max() + 1e-6)


def _apply_hotspots(body: np.ndarray, cls: str, rng: np.random.Generator) -> np.ndarray:
    """Add engine/exhaust hot-spots — IR targets are not uniformly warm."""
    h, w   = body.shape
    dom    = _domain_of(cls)
    n_spot = {"AIR": 2, "NAVAL": 1, "GROUND": 1}.get(dom, 1)
    yy, xx = np.ogrid[:h, :w]
    for _ in range(n_spot):
        hy = int(rng.integers(int(0.2 * h), int(0.8 * h) + 1))
        hx = int(rng.integers(int(0.2 * w), int(0.8 * w) + 1))
        r  = max(2.0, min(h, w) * 0.16)
        g  = np.exp(-(((yy - hy) ** 2 + (xx - hx) ** 2) / (2.0 * r * r)))
        body = body + g.astype(np.float32) * float(rng.uniform(25, 55))
    return body


def _make_target(cls: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (body HxW float32, mask HxW float32) — an oriented IR target."""
    h, w      = _BLOB_HW[cls]
    mean, std = _THERMAL[cls]
    grad = np.linspace(-0.5, 0.5, w, dtype=np.float32)[None, :] * std * 0.6
    body = rng.normal(mean, std * 0.5, (h, w)).astype(np.float32) + grad
    body = _apply_hotspots(body, cls, rng)
    return body, _silhouette(cls, h, w, rng)


def add_blob(background: np.ndarray, cls: str,
             rng: np.random.Generator) -> np.ndarray:
    """Composite a shaped, randomly-oriented IR target onto a 224×224 background."""
    out        = background.astype(np.float32)
    body, mask = _make_target(cls, rng)
    bh, bw     = body.shape

    # Place body+mask on a square canvas, then rotate to a random heading
    S  = int(np.ceil(np.hypot(bh, bw))) + 2
    cb = np.zeros((S, S), np.float32)
    cm = np.zeros((S, S), np.float32)
    y0, x0 = (S - bh) // 2, (S - bw) // 2
    cb[y0:y0 + bh, x0:x0 + bw] = body
    cm[y0:y0 + bh, x0:x0 + bw] = mask

    ang = float(rng.uniform(0, 360))
    M   = cv2.getRotationMatrix2D((S / 2.0, S / 2.0), ang, 1.0)
    cb  = cv2.warpAffine(cb, M, (S, S), flags=cv2.INTER_LINEAR, borderValue=0.0)
    cm  = cv2.warpAffine(cm, M, (S, S), flags=cv2.INTER_LINEAR, borderValue=0.0)

    # Shrink to fit if the rotated target is larger than the patch
    if S > IMG_SIZE:
        cb = cv2.resize(cb, (IMG_SIZE, IMG_SIZE))
        cm = cv2.resize(cm, (IMG_SIZE, IMG_SIZE))
        S  = IMG_SIZE

    max_y, max_x = IMG_SIZE - S, IMG_SIZE - S
    y = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
    x = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0

    a      = np.clip(cm, 0.0, 1.0)
    region = out[y:y + S, x:x + S]
    out[y:y + S, x:x + S] = a * cb + (1.0 - a) * region
    return out.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Purely synthetic IR patch (no FLIR needed)
# ---------------------------------------------------------------------------

def _synth_background(cls: str, rng: np.random.Generator) -> np.ndarray:
    """Generate a plausible IR background with low-freq texture and clutter.

    The clutter decoys (faint warm blobs, cooler than the real target) stop
    the classifier from cheating with a pure brightness threshold, so the
    synthetic accuracy better reflects genuine shape discrimination.
    """
    bg_mean = max(50, _THERMAL[cls][0] - 90)
    bg      = rng.normal(bg_mean, 16, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
    bg      = 0.6 * bg + 0.4 * cv2.GaussianBlur(bg, (21, 21), 0)

    yy, xx = np.ogrid[:IMG_SIZE, :IMG_SIZE]
    for _ in range(int(rng.integers(0, 4))):        # 0–3 decoys
        ry = int(rng.integers(0, IMG_SIZE))
        rx = int(rng.integers(0, IMG_SIZE))
        r  = float(rng.integers(6, 20))
        g  = np.exp(-(((yy - ry) ** 2 + (xx - rx) ** 2) / (2.0 * r * r)))
        bg = bg + g.astype(np.float32) * float(rng.uniform(15, 40))
    return bg.clip(0, 255).astype(np.uint8)


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
    per_cls: dict = defaultdict(int)
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
                    per_cls[cls] += 1
                except Exception:
                    continue

            if verbose:
                verb = "Would generate" if dry_run else "Generated"
                print(f"  [crop] {split}/{cls}: {verb} {generated}/{n}")

    return total, per_cls


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
    per_cls: dict = defaultdict(int)

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
                    per_cls[cls] += 1
                except Exception:
                    continue

            if verbose:
                verb = "Would generate" if dry_run else "Generated"
                print(f"  [bg ] {split}/{cls}: {verb} {generated}/{n}")

    return total, per_cls


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
    per_cls: dict = defaultdict(int)
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
                per_cls[cls] += 1
            if verbose:
                verb = "Would generate" if dry_run else "Generated"
                print(f"  [synth] {split}/{cls}: {verb} {n}")
    return total, per_cls


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
        n, pc = generate_synth_only(data_root, need, rng, args.dry_run, verbose)
        total_generated += n
        _merge_provenance(data_root, pc, "synthetic", args.dry_run)

    elif args.mode == "crop":
        n, pc = generate_crop_mode(flir_root, data_root, need, rng,
                                   args.dry_run, verbose)
        total_generated += n
        _merge_provenance(data_root, pc, "remapped", args.dry_run)

    elif args.mode == "background":
        n, pc = generate_background_mode(flir_root, data_root, need, rng,
                                         args.dry_run, verbose)
        total_generated += n
        _merge_provenance(data_root, pc, "synthetic", args.dry_run)

    elif args.mode == "both":
        # Pass 1: crop mode (real FLIR ROI → 'remapped' provenance)
        print("\n  Pass 1/2 — crop mode")
        n, pc = generate_crop_mode(flir_root, data_root, need, rng,
                                   args.dry_run, verbose)
        total_generated += n
        _merge_provenance(data_root, pc, "remapped", args.dry_run)
        # Recalculate shortfall
        if not args.dry_run:
            existing = _count_existing(data_root)
            need     = _shortfall(existing)
        # Pass 2: background mode fills the rest ('synthetic' provenance)
        remaining = sum(n for s in need.values() for n in s.values())
        if remaining:
            print(f"\n  Pass 2/2 — background mode ({remaining} still needed)")
            n, pc = generate_background_mode(flir_root, data_root, need, rng,
                                             args.dry_run, verbose)
            total_generated += n
            _merge_provenance(data_root, pc, "synthetic", args.dry_run)

    verb = "Would generate" if args.dry_run else "Generated"
    print(f"\n{'='*W}")
    print(f"  {verb} {total_generated} images total.")

    if not args.dry_run and total_generated > 0:
        print(f"\n  Run dataset_validator.py to verify the result:")
        print(f"    python dataset_validator.py --root {data_root}")
    print(f"{'='*W}\n")


if __name__ == "__main__":
    main()
