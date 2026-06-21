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
from typing import Iterator


def _img_exts():
    return {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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

    for ann in coco.get("annotations", []):
        img_info = img_meta.get(ann["image_id"])
        if img_info is None:
            continue
        fname  = img_info["file_name"]
        ipath  = _find_image(img_root, fname)
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
    results = []
    exts    = _img_exts()

    for txt_path in sorted(label_root.rglob("*.txt")):
        # Find matching image
        stem   = txt_path.stem
        ipath  = None
        for ext in exts:
            cand = img_root / (stem + ext)
            if cand.exists():
                ipath = cand
                break
        if ipath is None:
            # Try searching recursively
            found = list(img_root.rglob(stem + ".*"))
            if found:
                ipath = found[0]
        if ipath is None:
            continue

        # Image dimensions needed to de-normalise
        import cv2
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
            # Convert to absolute pixel coords
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
    results = []
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

        ipath = _find_image(img_root, fname)
        if ipath is None:
            # Try xml_root itself
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
    results = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Flexible column lookup
            fname = row.get(image_col) or row.get("filename") or row.get("image")
            if not fname:
                continue
            ipath = _find_image(img_root, fname)
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
# Internal helper
# ---------------------------------------------------------------------------

def _find_image(root: Path, fname: str) -> Path | None:
    """Locate an image file by name under root, trying multiple extensions."""
    stem = Path(fname).stem
    exts = _img_exts()
    # Exact path first
    direct = root / fname
    if direct.exists():
        return direct
    # Try swapping extension
    for ext in exts:
        cand = root / (stem + ext)
        if cand.exists():
            return cand
    # Recursive search
    for ext in exts:
        found = list(root.rglob(stem + ext))
        if found:
            return found[0]
    return None
