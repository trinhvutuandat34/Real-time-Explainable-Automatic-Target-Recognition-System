"""
Image preprocessing utilities for the REATS ingestion pipeline.

Key transforms:
  optical_to_pseudo_thermal  — convert RGB optical image to grayscale pseudo-IR
  extract_roi                — crop bbox from image and resize to IMG_SIZE×IMG_SIZE
  save_patch                 — write the final patch to disk
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


IMG_SIZE = 224   # output square patch size


# ---------------------------------------------------------------------------
# Optical → pseudo-thermal conversion
# ---------------------------------------------------------------------------

def optical_to_pseudo_thermal(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert an optical RGB image to a grayscale pseudo-thermal patch.

    Strategy:
    1. Convert to LAB, invert the L channel (bright = hot in thermal).
    2. Apply CLAHE for local contrast enhancement.
    3. Histogram-stretch to [0, 255].

    Returns uint8 grayscale (H, W).
    """
    # Handle already-grayscale input
    if img_bgr.ndim == 2:
        gray = img_bgr
    else:
        lab    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0].astype(np.float32)
        # Invert: dark objects become hot (typical for IR of vehicles)
        inverted = 255.0 - l_chan
        gray     = inverted.astype(np.uint8)

    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray   = clahe.apply(gray)

    # Stretch to full range
    lo, hi = gray.min(), gray.max()
    if hi > lo:
        gray = ((gray.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)
    return gray


# ---------------------------------------------------------------------------
# Thermal intensity remapping (match class-specific IR signature)
# ---------------------------------------------------------------------------

# Per-class thermal mean/std for intensity remapping.
# Organised by platform type — adjust these to match your sensor.
_CLASS_THERMAL: dict[str, tuple[float, float]] = {
    # Fighters
    "F16": (210, 28), "F15": (212, 30), "F22": (208, 27), "F35": (209, 28),
    "Su27": (215, 30), "Su35": (216, 31), "MiG29": (211, 28),
    "MiG19": (198, 26), "MiG21": (200, 27), "J20": (210, 29),
    # Bombers
    "B52": (205, 25), "Tu22M": (207, 26), "Tu95": (200, 24),
    # Attack helicopters
    "AH64": (200, 25), "Mi24": (198, 24), "Ka52": (197, 24),
    # Transport helicopters
    "LYNX": (178, 20), "UH60": (180, 21), "CH47": (183, 22),
    # UAVs
    "MQ9": (195, 23), "TB2": (190, 22), "Shahed136": (188, 22),
    "RQ4": (192, 22), "WZ7": (190, 21),
    # Ground MBTs
    "M1Abrams": (185, 22), "T72": (182, 21), "T90": (184, 22), "Leopard2": (183, 21),
    # IFV/APC
    "BMP2": (175, 20), "Bradley": (176, 20), "BTR80": (170, 19), "K21": (174, 20),
    # Artillery
    "M109": (178, 21), "BM21": (176, 20),
    # Air defense
    "Patriot": (172, 19), "Buk": (174, 20), "Pantsir": (175, 20),
    # Naval (water cooling)
    "PKG": (158, 18), "PTG": (152, 20), "FastAttack": (160, 18),
    "Destroyer": (165, 18), "Frigate": (163, 18), "Corvette": (162, 18),
}


def remap_intensity(
    gray: np.ndarray,
    target_class: str,
) -> np.ndarray:
    """
    Linearly remap pixel intensities to match the target class IR signature.
    Applied after optical_to_pseudo_thermal for optical-source images.
    """
    mean_t, std_t = _CLASS_THERMAL.get(target_class, (185, 22))
    src = gray.astype(np.float32)
    m, s = src.mean(), src.std() + 1e-6
    out = (src - m) / s * std_t + mean_t
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# ROI extraction
# ---------------------------------------------------------------------------

def extract_roi(
    img: np.ndarray,
    bbox: list[int] | None,
    *,
    pad_frac: float = 0.1,
) -> np.ndarray:
    """
    Crop the bounding box region (with optional padding) from img.
    If bbox is None, the full image is used as the ROI.
    Returns a square patch at IMG_SIZE×IMG_SIZE.
    """
    if img is None or img.size == 0:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    if bbox is None:
        roi = img
    else:
        x1, y1, x2, y2 = bbox
        h, w = img.shape[:2]
        pad_x = max(1, int((x2 - x1) * pad_frac))
        pad_y = max(1, int((y2 - y1) * pad_frac))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        roi = img[y1:y2, x1:x2]

    if roi.size == 0:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    # Convert to grayscale if needed
    if roi.ndim == 3 and roi.shape[2] == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    elif roi.ndim == 3:
        roi = roi[:, :, 0]

    return cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Patch save
# ---------------------------------------------------------------------------

def save_patch(
    patch: np.ndarray,
    dst: Path,
) -> None:
    """
    Save a grayscale uint8 patch as a 3-channel PNG
    (ConvNeXt expects 3 channels via Grayscale transform at load time).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(dst), rgb)


# ---------------------------------------------------------------------------
# Full preprocessing chain for one annotation
# ---------------------------------------------------------------------------

def process_annotation(
    ann: dict,
    target_class: str,
    is_thermal: bool,
) -> np.ndarray | None:
    """
    Load image (or video frame), apply optical→thermal conversion (if needed),
    extract ROI. Returns uint8 grayscale patch (IMG_SIZE×IMG_SIZE) or None.

    Video frames: annotation must carry _frame_idx; VideoCapture seeks to that
    frame instead of using cv2.imread.
    """
    try:
        frame_idx = ann.get("_frame_idx")

        if frame_idx is not None:
            cap = cv2.VideoCapture(str(ann["image_path"]))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, img = cap.read()
            cap.release()
            if not ok or img is None:
                return None
        else:
            img = cv2.imread(str(ann["image_path"]), cv2.IMREAD_COLOR)
            if img is None:
                return None

        if not is_thermal:
            gray = optical_to_pseudo_thermal(img)
            gray = remap_intensity(gray, target_class)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        patch = extract_roi(gray, ann.get("bbox"))
        return patch
    except Exception:
        return None
