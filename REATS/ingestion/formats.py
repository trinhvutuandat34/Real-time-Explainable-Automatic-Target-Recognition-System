"""
Annotation format parsers — return a common list of dicts:
    [{"image_path": Path, "bbox": [x1,y1,x2,y2], "label": str, "area": float}, ...]

Supported formats:
  - COCO JSON (FLIR, FLIR ADAS v2)
  - YOLO txt  (HIT-UAV, ship datasets)
  - Pascal VOC XML (HRSC2016)
  - CSV        (Airbus aircraft, some ship sets)
  - Folder     (one sub-folder per class, no bbox → whole image as ROI)
"""

from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def _img_exts():
    return {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _build_img_cache(root: Path) -> dict[str, Path]:
    """
    Scan root once and return {stem: path} for O(1) image lookups.
    This replaces per-annotation rglob calls which are catastrophically slow
    on large datasets (FLIR_Thermal: 100k+ annotations × rglob = hours).
    """
    cache: dict[str, Path] = {}
    exts = _img_exts()
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            cache.setdefault(p.stem, p)
    return cache


def _find_image(
    root: Path,
    fname: str,
    cache: dict[str, Path] | None = None,
) -> Path | None:
    """Locate an image file by name under root.

    Uses cache for O(1) lookup when provided. Falls back to rglob only when
    cache is absent (avoid calling without cache on large datasets).
    """
    stem = Path(fname).stem
    exts = _img_exts()
    # 1. Exact path
    direct = root / fname
    if direct.exists():
        return direct
    # 2. Extension swap at root level
    for ext in exts:
        cand = root / (stem + ext)
        if cand.exists():
            return cand
    # 3. Pre-built cache (fast)
    if cache is not None:
        return cache.get(stem)
    # 4. Last resort: recursive scan (slow — only hit if no cache)
    for ext in exts:
        found = list(root.rglob(stem + ext))
        if found:
            return found[0]
    return None


# ---------------------------------------------------------------------------
# COCO JSON
# ---------------------------------------------------------------------------

def parse_coco(
    ann_json: Path,
    img_root: Path,
) -> list[dict]:
    """Parse a COCO-format annotation file."""
    with open(ann_json) as f:
        coco = json.load(f)

    cat_map  = {c["id"]: c["name"] for c in coco.get("categories", [])}
    img_meta = {i["id"]: i for i in coco.get("images", [])}
    results  = []

    # Build once — avoids one rglob per annotation (critical for FLIR 100k+ entries)
    img_cache = _build_img_cache(img_root)

    for ann in coco.get("annotations", []):
        img_info = img_meta.get(ann["image_id"])
        if img_info is None:
            continue
        fname = img_info["file_name"]
        ipath = _find_image(img_root, fname, cache=img_cache)
        if ipath is None:
            continue
        x, y, w, h = ann["bbox"]
        if w < 8 or h < 8:
            continue
        label = cat_map.get(ann.get("category_id"), "")
        results.append({
            "image_path": ipath,
            "bbox":       [int(x), int(y), int(x + w), int(y + h)],
            "label":      label.lower().strip(),
            "area":       float(w * h),
        })
    return results


# ---------------------------------------------------------------------------
# YOLO txt
# ---------------------------------------------------------------------------

def parse_yolo(
    label_root: Path,
    img_root:   Path,
    class_names: list[str],
) -> list[dict]:
    """Parse YOLO txt annotations (normalised cx,cy,w,h)."""
    import cv2

    results   = []
    exts      = _img_exts()
    img_cache = _build_img_cache(img_root)

    for txt_path in sorted(label_root.rglob("*.txt")):
        stem  = txt_path.stem
        ipath = _find_image(img_root, stem, cache=img_cache)
        if ipath is None:
            continue

        img = cv2.imread(str(ipath), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        ih, iw = img.shape[:2]

        for line in txt_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_idx = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = int((cx - bw / 2) * iw)
            y1 = int((cy - bh / 2) * ih)
            x2 = int((cx + bw / 2) * iw)
            y2 = int((cy + bh / 2) * ih)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(iw, x2), min(ih, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            label = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
            results.append({
                "image_path": ipath,
                "bbox":       [x1, y1, x2, y2],
                "label":      label.lower().strip(),
                "area":       float((x2 - x1) * (y2 - y1)),
            })
    return results


# ---------------------------------------------------------------------------
# Pascal VOC XML  (HRSC2016 variant)
# ---------------------------------------------------------------------------

def parse_xml(
    xml_root: Path,
    img_root: Path,
) -> list[dict]:
    """Parse Pascal VOC XML annotations. Also handles HRSC2016 XML variant."""
    results   = []
    img_cache = _build_img_cache(img_root)

    for xml_path in sorted(xml_root.rglob("*.xml")):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError:
            continue

        # Image file reference
        fname = None
        for tag in ("filename", "Filename", "HRSC_Image"):
            el = root.find(tag)
            if el is not None and el.text:
                fname = el.text.strip()
                break
        if fname is None:
            fname = xml_path.stem

        ipath = _find_image(img_root, fname, cache=img_cache)
        if ipath is None:
            # Try xml_root parent as fallback
            ipath = _find_image(xml_path.parent, fname)
        if ipath is None:
            continue

        # Objects: standard VOC or HRSC HRSC_Objects/HRSC_Object
        objects = (
            root.findall("object")
            or root.findall(".//HRSC_Object")
        )
        for obj in objects:
            # Label
            name_el = obj.find("name") or obj.find("Class_ID")
            if name_el is None or not name_el.text:
                continue
            label = name_el.text.strip()

            # Bbox — handles VOC <bndbox>, HRSC <box> (lowercase), and <Box>
            bnd = obj.find("bndbox") or obj.find("box") or obj.find("Box")
            if bnd is None:
                continue
            try:
                x1 = int(float(
                    bnd.findtext("xmin") or bnd.findtext("box_xmin")
                    or bnd.findtext("x") or 0))
                y1 = int(float(
                    bnd.findtext("ymin") or bnd.findtext("box_ymin")
                    or bnd.findtext("y") or 0))
                x2 = int(float(
                    bnd.findtext("xmax") or bnd.findtext("box_xmax") or "0")
                    or float(bnd.findtext("x") or 0) + float(bnd.findtext("w") or 0))
                y2 = int(float(
                    bnd.findtext("ymax") or bnd.findtext("box_ymax") or "0")
                    or float(bnd.findtext("y") or 0) + float(bnd.findtext("h") or 0))
            except (TypeError, ValueError):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            results.append({
                "image_path": ipath,
                "bbox":       [x1, y1, x2, y2],
                "label":      label.lower().strip(),
                "area":       float((x2 - x1) * (y2 - y1)),
            })
    return results


# ---------------------------------------------------------------------------
# CSV  (Airbus-style: image_path,x1,y1,x2,y2,label or similar)
# ---------------------------------------------------------------------------

def parse_csv(
    csv_path: Path,
    img_root: Path,
    *,
    image_col: str = "image_path",
    label_col: str = "label",
    x1_col: str = "x1",
    y1_col: str = "y1",
    x2_col: str = "x2",
    y2_col: str = "y2",
) -> list[dict]:
    """Parse a CSV annotation file with configurable column names."""
    results   = []
    img_cache = _build_img_cache(img_root)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get(image_col) or row.get("filename") or row.get("image")
            if not fname:
                continue
            ipath = _find_image(img_root, fname, cache=img_cache)
            if ipath is None:
                continue
            try:
                x1 = int(float(row.get(x1_col, 0) or 0))
                y1 = int(float(row.get(y1_col, 0) or 0))
                x2 = int(float(row.get(x2_col, 0) or 0))
                y2 = int(float(row.get(y2_col, 0) or 0))
            except (ValueError, TypeError):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            label = (
                row.get(label_col) or row.get("class") or row.get("category") or ""
            ).strip().lower()
            results.append({
                "image_path": ipath,
                "bbox":       [x1, y1, x2, y2],
                "label":      label,
                "area":       float((x2 - x1) * (y2 - y1)),
            })
    return results


# ---------------------------------------------------------------------------
# Folder-based  (sub-folder = class, whole image as ROI)
# ---------------------------------------------------------------------------

def parse_folder(root: Path, class_names: list[str] | None = None) -> list[dict]:
    """
    Each sub-directory is a class label. The whole image is used as ROI
    (bbox = full image). Useful when no bbox annotations exist.
    """
    exts    = _img_exts()
    results = []
    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue
        label = cls_dir.name.lower().strip()
        if class_names and label not in {c.lower() for c in class_names}:
            continue
        for img_path in cls_dir.rglob("*"):
            if img_path.suffix.lower() not in exts:
                continue
            results.append({
                "image_path": img_path,
                "bbox":       None,   # full image
                "label":      label,
                "area":       float("inf"),
            })
    return results


# ---------------------------------------------------------------------------
# Video folder  (sub-folder = class, frames sampled from each video)
# ---------------------------------------------------------------------------

def _video_exts() -> set[str]:
    return {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def parse_video_folder(
    root: Path,
    class_names: list[str] | None = None,
    frames_per_video: int = 8,
) -> list[dict]:
    """
    Like parse_folder() but for video files (mp4/avi/mov/mkv/wmv).

    Each sub-directory is a class label. frames_per_video evenly-spaced
    frames are sampled from every video file. Annotations carry a
    _frame_idx field; process_annotation() reads that frame via
    cv2.VideoCapture instead of cv2.imread.
    """
    import cv2

    video_exts = _video_exts()
    results: list[dict] = []

    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue
        label = cls_dir.name.lower().strip()
        if class_names and label not in {c.lower() for c in class_names}:
            continue

        for vid_path in sorted(cls_dir.rglob("*")):
            if vid_path.suffix.lower() not in video_exts:
                continue

            cap      = cv2.VideoCapture(str(vid_path))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if n_frames <= 0:
                continue

            n = min(frames_per_video, n_frames)
            for i in range(n):
                frame_idx = int(i * n_frames / n)
                results.append({
                    "image_path": vid_path,
                    "bbox":       None,
                    "label":      label,
                    "area":       float("inf"),
                    "_frame_idx": frame_idx,
                })

    return results
