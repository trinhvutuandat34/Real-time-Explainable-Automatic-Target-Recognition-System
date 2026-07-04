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


# Memo: {resolved root: cache}. Several parsers scan the same root
# (HIT_UAV_v2 runs 4 COCO files over one tree; the YOLO fallback loops
# over multiple labels/ dirs) — without this each call re-walks the
# entire dataset.
_IMG_CACHE_MEMO: dict[str, dict] = {}


def _build_img_cache(root: Path) -> dict[str, Path]:
    """
    Scan root once and return {stem: path} for O(1) image lookups.
    This replaces per-annotation rglob calls which are catastrophically slow
    on large datasets (FLIR_Thermal: 100k+ annotations × rglob = hours).
    """
    key = str(root.resolve())
    hit = _IMG_CACHE_MEMO.get(key)
    if hit is not None:
        return hit
    cache: dict[str, Path] = {}
    exts = _img_exts()
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            cache.setdefault(p.stem, p)
    _IMG_CACHE_MEMO[key] = cache
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
        fname = (
            img_info.get("file_name")
            or img_info.get("filename")
            or img_info.get("path")
        )
        if not fname:
            continue
        ipath = _find_image(img_root, fname, cache=img_cache)
        if ipath is None:
            continue
        bbox = ann.get("bbox", [])
        if len(bbox) < 4:
            continue
        x, y, w, h = bbox[:4]
        if len(bbox) >= 5:
            # Rotated-box COCO (e.g. HIT-UAV rotate_json): (xc, yc, w, h, angle).
            # Convert the centre to a top-left corner or every crop lands
            # half a box off-target.
            x, y = x - w / 2.0, y - h / 2.0
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
    from PIL import Image

    results   = []
    img_cache = _build_img_cache(img_root)

    for txt_path in sorted(label_root.rglob("*.txt")):
        stem  = txt_path.stem
        ipath = _find_image(img_root, stem, cache=img_cache)
        if ipath is None:
            continue

        # PIL reads only the header for .size — never cv2.imread here, a
        # full pixel decode per file makes large YOLO sets take minutes.
        try:
            with Image.open(ipath) as im:
                iw, ih = im.size
        except Exception:
            continue

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

def _find_first(parent, *tags):
    """First existing child element among tags, by explicit None check.

    NEVER use `find(a) or find(b)` — an ET Element with no children is
    FALSY, so a leaf tag like <name>ship</name> evaluates False and the
    chain silently falls through. This bug made every VOC dataset parse
    zero objects.
    """
    for tag in tags:
        el = parent.find(tag)
        if el is not None:
            return el
    return None


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
            # Try the xml's own directory — cached, or a recursive glob
            # fires per XML file (O(N²) on HRSC-sized annotation sets)
            ipath = _find_image(xml_path.parent, fname,
                                cache=_build_img_cache(xml_path.parent))
        if ipath is None:
            continue

        # Objects: standard VOC or HRSC HRSC_Objects/HRSC_Object
        objects = (
            root.findall("object")
            or root.findall(".//HRSC_Object")
        )
        for obj in objects:
            # Label
            name_el = _find_first(obj, "name", "Class_ID", "n")
            if name_el is None or not name_el.text:
                continue
            label = name_el.text.strip()

            # Bbox — VOC <bndbox>, HRSC <box>/<Box>; HRSC also puts
            # box_xmin/... directly on the object element itself.
            bnd    = _find_first(obj, "bndbox", "box", "Box")
            coords = bnd if bnd is not None else obj

            def _coord(*names) -> float | None:
                for n in names:
                    t = coords.findtext(n)
                    if t is not None and t.strip():
                        try:
                            return float(t)
                        except ValueError:
                            return None
                return None

            x1f = _coord("xmin", "box_xmin", "x")
            y1f = _coord("ymin", "box_ymin", "y")
            x2f = _coord("xmax", "box_xmax")
            y2f = _coord("ymax", "box_ymax")
            if x2f is None or y2f is None:
                wf = _coord("w", "width")
                hf = _coord("h", "height")
                if None in (x1f, y1f, wf, hf):
                    continue
                x2f, y2f = x1f + wf, y1f + hf
            if None in (x1f, y1f, x2f, y2f):
                continue
            x1, y1, x2, y2 = int(x1f), int(y1f), int(x2f), int(y2f)
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

    import re

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = (row.get(image_col) or row.get("filename")
                     or row.get("image") or row.get("image_id"))
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
                # WKT geometry column (Airbus aircraft sample:
                # "POLYGON ((x y, x y, ...))") — bbox = envelope of the points
                geom = row.get("geometry") or row.get("wkt") or ""
                nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", geom)]
                if len(nums) >= 4:
                    xs, ys = nums[0::2], nums[1::2]
                    x1, y1 = int(min(xs)), int(min(ys))
                    x2, y2 = int(max(xs)), int(max(ys))
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

def _class_leaf_walk(d: Path, exts: set[str], depth: int, max_depth: int):
    """See _iter_class_leaf_dirs. Yields class-leaf directories under `d`."""
    try:
        has_media = any(p.is_file() and p.suffix.lower() in exts for p in d.iterdir())
    except OSError:
        return
    if has_media or depth >= max_depth:
        yield d
        return
    try:
        subdirs = sorted(p for p in d.iterdir() if p.is_dir())
    except OSError:
        return
    if not subdirs:
        yield d   # dead end — yield anyway so the caller sees *something*
        return
    for sub in subdirs:
        yield from _class_leaf_walk(sub, exts, depth + 1, max_depth)


def _iter_class_leaf_dirs(root: Path, exts: set[str], max_depth: int = 4):
    """Yield the directories under `root` that hold real per-class content.

    A directory counts as a class leaf once it contains at least one direct
    file matching `exts` — that's the signal it's a genuine per-class
    folder, not an organisational wrapper. Some Kaggle mirrors add one or
    more such wrappers above the real per-class folders (a dataset slug, a
    version string, an internal tooling folder) with no media of their own;
    without this, `root`'s immediate children were read as the class labels
    even when they were really 'swim_dataset_1.0.0' or 'ships-aerial-images'
    rather than the actual class name one or more levels down.

    A directory with its own direct media is a leaf (its subdirectories, if
    any, are pulled in via the caller's rglob, not treated as separate
    classes); a directory with none is assumed to be a wrapper and is
    searched one level deeper, up to `max_depth`. For a normal, already
    well-formed dataset (real class folders directly under root, each
    containing images), this returns exactly root's immediate children —
    unchanged from the prior single-level behaviour.
    """
    if not root.is_dir():
        return
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        yield from _class_leaf_walk(cls_dir, exts, 0, max_depth)


def parse_folder(root: Path, class_names: list[str] | None = None) -> list[dict]:
    """
    Each sub-directory is a class label. The whole image is used as ROI
    (bbox = full image). Useful when no bbox annotations exist.
    """
    exts    = _img_exts()
    results = []
    for cls_dir in _iter_class_leaf_dirs(root, exts):
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
    media_exts = _img_exts() | video_exts
    results: list[dict] = []

    for cls_dir in _iter_class_leaf_dirs(root, media_exts):
        label = cls_dir.name.lower().strip()
        if class_names and label not in {c.lower() for c in class_names}:
            continue

        for vid_path in sorted(cls_dir.rglob("*")):
            suffix = vid_path.suffix.lower()
            if suffix in _img_exts():
                # Stray still images alongside the videos (e.g. a reference
                # .webp in dataset2/fixed_wing) — use as whole-frame samples
                results.append({
                    "image_path": vid_path,
                    "bbox":       None,
                    "label":      label,
                    "area":       float("inf"),
                })
                continue
            if suffix not in video_exts:
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
