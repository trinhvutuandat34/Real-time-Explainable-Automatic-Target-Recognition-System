"""
REATS Multi-Dataset Ingestion Pipeline

Loads annotations from any configured dataset, maps source labels to the
REATS 43-class taxonomy, extracts IR patches, and writes a standard
train / val / test split ready for Module B.

Usage (CLI):
    cd REATS
    python -m ingestion.pipeline \\
        --datasets flir_thermal:/kaggle/input/flir-thermal-images-dataset \\
                   hit_uav:/kaggle/input/hit-uav \\
                   hrsc2016:/kaggle/input/hrsc2016 \\
        --out data/ \\
        --train 170 --val 30 --test 200

Usage (Python):
    from ingestion.pipeline import IngestPipeline
    pipe = IngestPipeline(
        datasets={"FLIR_Thermal": "/path/to/flir", "HIT_UAV": "/path/to/hit_uav"},
        out_root="data/",
    )
    pipe.run()
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

_HERE = Path(__file__).parent
_REATS_ROOT = _HERE.parent
if str(_REATS_ROOT) not in sys.path:
    sys.path.insert(0, str(_REATS_ROOT))

from config import CLASSES, NUM_CLASSES
from ingestion.formats import (
    parse_coco, parse_yolo, parse_xml, parse_csv,
    parse_folder, parse_video_folder, parse_filename_prefix,
)
from ingestion.preprocessor import (
    process_annotation, save_patch,
    load_frame, to_ir_look, save_frame, write_yolo_labels,
)

# ---------------------------------------------------------------------------
# Label map loader
# ---------------------------------------------------------------------------

def _load_label_maps() -> dict:
    with open(_HERE / "label_maps.yaml") as f:
        return yaml.safe_load(f)


def _norm_label(s: str) -> str:
    """Normalise a raw label for map lookup: lowercase, strip, unify separators."""
    return str(s).strip().lower().replace("-", "_").replace(" ", "_")


def _yolo_names_near(labels_dir: Path, max_up: int = 3) -> list[str] | None:
    """Read a YOLO dataset's own class names from a data.yaml near a labels dir.

    Roboflow/Ultralytics exports ship a `data.yaml` (`names: [...]`) beside the
    train/val/test folders, so the class-index → name mapping travels with the
    data instead of being hard-coded here. Walks up from `labels_dir` (e.g.
    `<wrapper>/train/labels`) looking for data.yaml/data.yml/dataset.yaml, so a
    dataset that bundles several YOLO sub-datasets (each with its own data.yaml
    and its own class order) resolves each one correctly. Returns None if none
    found — callers fall back to the configured `yolo_classes`.
    """
    d = labels_dir
    for _ in range(max_up + 1):
        for yname in ("data.yaml", "data.yml", "dataset.yaml"):
            yf = d / yname
            if yf.exists():
                try:
                    y = yaml.safe_load(yf.read_text()) or {}
                except Exception:
                    y = {}
                names = y.get("names")
                if isinstance(names, dict):   # {0: ship, 1: person} → ordered list
                    names = [names[k] for k in sorted(names)]
                if isinstance(names, list) and names:
                    return [str(n) for n in names]
        if d.parent == d:
            break
        d = d.parent
    return None


def _resolve_label(
    raw_label: str,
    dataset_cfg: dict,
    area: float,
) -> tuple[str | None, bool]:
    """
    Resolve a source label to a REATS class ID.

    Returns (class_id | None, matched):
      matched=False → the raw label has no entry in the dataset's label map
                      (a candidate for a label_maps.yaml fix — report it!)
      matched=True, class None → explicit null mapping (intentional discard)
    """
    label_map  = dataset_cfg.get("labels", {}) or {}
    size_rules = dataset_cfg.get("size_rules", [])

    key = None
    if raw_label in label_map:
        key = raw_label
    elif raw_label.lower() in label_map:
        key = raw_label.lower()
    else:
        # Normalised lookup: 'Other Vehicle' / 'other-vehicle' → 'other_vehicle'
        norm_map = dataset_cfg.get("_norm_cache")
        if norm_map is None:
            norm_map = {_norm_label(k): k for k in label_map}
            dataset_cfg["_norm_cache"] = norm_map
        key = norm_map.get(_norm_label(raw_label))

    if key is None:
        return None, False

    mapped = label_map[key]
    if mapped is None:
        return None, True
    if mapped == "__size_rule__":
        for rule in size_rules:
            if "max_area_px" not in rule:
                return rule.get("class"), True
            if area <= rule["max_area_px"]:
                return rule.get("class"), True
        return None, True
    return mapped, True   # direct class ID


# ---------------------------------------------------------------------------
# Per-dataset annotation loader
# ---------------------------------------------------------------------------

_KNOWN_DATASETS: dict[str, dict] = {
    # Map dataset key → {format, sub-paths, yolo_classes}
    "FLIR_Thermal": {
        "format": "coco",
        "ann_paths": ["annotations/train.json", "annotations/val.json"],
        "img_roots": ["images/train", "images/val"],
    },
    "FLIR_ADAS_v2": {
        "format": "coco",
        "ann_paths": ["images/train/thermal_annotations.json",
                      "images/val/thermal_annotations.json"],
        "img_roots": ["images/train", "images/val"],
    },
    "HIT_UAV": {
        "format": "yolo",
        "label_dirs": ["labels/train", "labels/val", "labels/test"],
        "img_dirs":   ["images/train", "images/val", "images/test"],
        "yolo_classes": ["Person", "Car", "Bicycle", "OtherVehicle", "DontCare"],
    },
    "HRSC2016": {
        "format": "xml",
        "ann_dirs": ["Train/Annotations", "Test/Annotations"],
        "img_dirs": ["Train/AllImages", "Test/AllImages"],
    },
    "Ships_Aerial": {
        "format": "yolo",
        "label_dirs": ["labels"],
        "img_dirs":   ["images"],
        "yolo_classes": ["ship"],
    },
    "Ships_Google_Earth": {
        "format": "yolo",
        "label_dirs": ["labels"],
        "img_dirs":   ["images"],
        "yolo_classes": ["ship"],
    },
    "Ships_Vessels_Aerial": {
        "format": "yolo",
        "label_dirs": ["labels"],
        "img_dirs":   ["images"],
        "yolo_classes": ["ship", "vessel"],
    },
    # SWIM ship-wake — Pascal-VOC layout with ROTATED boxes (<robndbox>):
    # SWIM_Dataset_1.0.0/{Annotations(xml), JPEGImages(jpg), Landmarks(xml)}.
    # parse_xml reads <robndbox> and takes its axis-aligned envelope. Explicit
    # paths target Annotations (not the same-shaped Landmarks dir); the xml
    # rglob fallback still recovers it if the version-stamped wrapper changes.
    "SWIM": {
        "format": "xml",
        "ann_dirs": ["SWIM_Dataset_1.0.0/Annotations"],
        "img_dirs": ["SWIM_Dataset_1.0.0/JPEGImages"],
    },
    # Airbus aircraft sample (airbusgeo): images/*.jpg + annotations.csv with
    # image_id + WKT geometry columns (parse_csv extracts the envelope bbox).
    "CGI_Planes": {
        "format": "csv",
        "ann_file": "annotations.csv",
        "img_root": "images",
    },
    "Airbus_Aircraft": {
        "format": "csv",
        "ann_file": "annotations.csv",
        "img_root": "images",
    },
    "SwimmingPool_Car": {
        "format": "yolo",
        "label_dirs": ["labels"],
        "img_dirs":   ["images"],
        "yolo_classes": ["car", "swimming_pool"],
    },
    "Vehicle_Dataset": {
        "format": "folder",
        "img_root": ".",
    },
    "Aerial_Segmentation": {
        "format": "yolo",
        "label_dirs": ["labels"],
        "img_dirs":   ["images"],
        "yolo_classes": ["building", "road", "vegetation", "water", "vehicle",
                         "airplane", "ship"],
    },
    # HIT-UAV Infrared Thermal Dataset v1.2.1 — COCO JSON, nested structure
    # dataset root: /kaggle/input/datasets/trnhvtunt/dataset1/
    # Inner folder: suojiashun-HIT-UAV-Infrared-Thermal-Dataset-b53106c
    # Images live at dataset_root/Images/Images  (separate from annotations)
    "HIT_UAV_v2": {
        "format": "coco",
        "ann_paths": [
            "HIT-UAV-Infrared-Thermal-Dataset-v1.2.1/suojiashun-HIT-UAV-Infrared-Thermal-Dataset-b53106c/normal_json/train.json",
            "HIT-UAV-Infrared-Thermal-Dataset-v1.2.1/suojiashun-HIT-UAV-Infrared-Thermal-Dataset-b53106c/normal_json/val.json",
            "HIT-UAV-Infrared-Thermal-Dataset-v1.2.1/suojiashun-HIT-UAV-Infrared-Thermal-Dataset-b53106c/rotate_json/train.json",
            "HIT-UAV-Infrared-Thermal-Dataset-v1.2.1/suojiashun-HIT-UAV-Infrared-Thermal-Dataset-b53106c/rotate_json/val.json",
        ],
        # Images are at dataset_root/Images/Images — use "." so _build_img_cache
        # scans the whole root recursively and finds them regardless of nesting.
        "img_roots": [".", ".", ".", "."],
    },
    # Video-based classification dataset (trnhvtunt/dataset2)
    # Sub-folders fixed_wing / rotary_wing contain mp4 video files.
    # parse_video_folder() samples 8 frames per video.
    "Dataset2_Folders": {
        "format": "video_folder",
        "img_root": ".",
    },
    # Aerial imagery for roof segmentation — no bbox, no military labels.
    # Included for completeness; all labels map to null in label_maps.yaml.
    "Aerial_Roof_Seg": {
        "format": "folder",
        "img_root": ".",
    },
    # SARScope — SAR maritime ship detection (Roboflow YOLO under SaRscope/).
    # Class names come from the dataset's own data.yaml (['background','ship']);
    # yolo_classes here is only the fallback if that file is missing.
    "SARScope_Maritime": {
        "format": "yolo",
        "yolo_classes": ["background", "ship"],
    },
    # Thermal_Ships — genuinely IR/thermal ship imagery. Bundles THREE YOLO
    # sub-datasets (massmind_yolo / "IR boats.yolov11" / "Thermal ships.yolov11"),
    # each with its own data.yaml and its own class order — the rglob fallback
    # reads per-sub-dataset names via _yolo_names_near, so one yolo_classes
    # fallback here does not have to be correct for all three.
    "Thermal_Ships": {
        "format": "yolo",
        "yolo_classes": ["vessel", "person"],
    },
    # Ships in satellite imagery (shipsnet) — no annotation files; 80×80 tiles
    # named '<label>__<scene>__<coords>.png' with label 1=ship / 0=no-ship.
    "Ships_Satellite": {
        "format": "filename_prefix",
        "img_root": "shipsnet",     # skip the unlabeled scenes/ folder
        "prefix_sep": "__",
        "prefix_index": 0,
    },
}


def _autodetect_annotations(
    dataset_key: str,
    dataset_root: Path,
) -> list[dict]:
    """
    Fallback when configured format yields 0 annotations.
    Tries XML → COCO JSON → YOLO in any subdir → folder, in that order.
    Handles Kaggle datasets whose actual structure differs from the expected layout.
    """
    info    = _KNOWN_DATASETS.get(dataset_key, {})
    classes = info.get("yolo_classes", ["object"])

    # ── 1. XML: any dir named Annotations / annotation / labels_xml ─────
    _xml_dir_names = {"annotations", "annotation", "labels_xml", "xmllabels"}
    for candidate in sorted(dataset_root.rglob("*")):
        if not candidate.is_dir():
            continue
        if candidate.name.lower() not in _xml_dir_names:
            continue
        parent  = candidate.parent
        img_dir = next(
            (parent / d for d in ("images", "Images", "JPEGImages", "imgs", "AllImages")
             if (parent / d).exists()),
            parent,
        )
        anns = parse_xml(candidate, img_dir)
        if anns:
            return anns

    # ── 2. XML: any directory that contains .xml files ──────────────────
    xml_dirs: set[Path] = set()
    for xf in dataset_root.rglob("*.xml"):
        xml_dirs.add(xf.parent)
    for xml_dir in sorted(xml_dirs):
        parent  = xml_dir.parent
        img_dir = next(
            (parent / d for d in ("images", "Images", "JPEGImages", "imgs")
             if (parent / d).exists()),
            xml_dir,   # images alongside XMLs
        )
        anns = parse_xml(xml_dir, img_dir)
        if anns:
            return anns

    # ── 3. COCO JSON: any JSON with annotation/train/val in name ────────
    for jf in sorted(dataset_root.rglob("*.json")):
        name = jf.name.lower()
        if not any(k in name for k in ("annotation", "train", "val", "test",
                                       "instance", "coco")):
            continue
        try:
            parent   = jf.parent
            img_root = next(
                (parent / d for d in ("images", "imgs", "JPEGImages") if (parent / d).exists()),
                parent,
            )
            anns = parse_coco(jf, img_root)
            if anns:
                return anns
        except Exception:
            pass

    # ── 4. YOLO txt: any labels/ subdirectory ───────────────────────────
    all_yolo: list[dict] = []
    for lbl_dir in dataset_root.rglob("labels"):
        if not lbl_dir.is_dir():
            continue
        names   = _yolo_names_near(lbl_dir) or classes
        img_dir = lbl_dir.parent / "images"
        all_yolo += parse_yolo(lbl_dir, img_dir if img_dir.exists() else dataset_root, names)
    if all_yolo:
        return all_yolo

    # ── 5. Video folder: mp4/avi/mov files in class sub-directories ─────
    from ingestion.formats import _video_exts
    if any(dataset_root.rglob(f"*{ext}") for ext in _video_exts()):
        anns = parse_video_folder(dataset_root)
        if anns:
            return anns

    # ── 6. Folder-based (class name = subdir name) ───────────────────────
    return parse_folder(dataset_root)


def load_dataset_annotations(
    dataset_key: str,
    dataset_root: Path,
    format_hint: str | None = None,
) -> list[dict]:
    """
    Load raw annotations from a dataset directory.
    Returns list of dicts: {image_path, bbox, label, area}.
    If the configured format yields 0 results, auto-detects the actual format.
    """
    info = _KNOWN_DATASETS.get(dataset_key, {})
    fmt  = format_hint or info.get("format", "folder")
    annotations: list[dict] = []

    if fmt == "coco":
        for ann_rel, img_rel in zip(
            info.get("ann_paths", ["annotations.json"]),
            info.get("img_roots", ["images"]),
        ):
            ann_path = dataset_root / ann_rel
            img_root = dataset_root / img_rel
            if ann_path.exists() and img_root.exists():
                annotations += parse_coco(ann_path, img_root)
        if not annotations:
            # Configured paths missing — parse every candidate JSON exactly
            # once. (The old per-missing-path retry parsed the same first
            # match repeatedly, duplicating annotations and wall time.)
            seen: set = set()
            for cand in sorted(dataset_root.rglob("*.json")):
                cname = cand.name.lower()
                if not any(k in cname for k in ("annotation", "train", "val", "coco")):
                    continue
                key = str(cand.resolve())
                if key in seen:
                    continue
                seen.add(key)
                try:
                    got = parse_coco(cand, cand.parent)
                    if not got:
                        got = parse_coco(cand, dataset_root)
                    annotations += got
                except Exception:
                    continue

    elif fmt == "yolo":
        classes = info.get("yolo_classes", ["object"])
        for lbl_rel, img_rel in zip(
            info.get("label_dirs", ["labels"]),
            info.get("img_dirs",   ["images"]),
        ):
            lbl_dir = dataset_root / lbl_rel
            img_dir = dataset_root / img_rel
            if lbl_dir.exists():
                annotations += parse_yolo(lbl_dir, img_dir if img_dir.exists() else dataset_root, classes)
        if not annotations:
            for lbl_dir in dataset_root.rglob("labels"):
                if lbl_dir.is_dir():
                    # Pair each labels/ dir with its sibling images/ dir, and
                    # prefer the sub-dataset's own data.yaml class names — a
                    # single dataset root may bundle several YOLO sets with
                    # different class orders (e.g. Thermal_Ships).
                    names   = _yolo_names_near(lbl_dir) or classes
                    img_dir = lbl_dir.parent / "images"
                    annotations += parse_yolo(
                        lbl_dir, img_dir if img_dir.exists() else dataset_root, names)

    elif fmt == "xml":
        for ann_rel, img_rel in zip(
            info.get("ann_dirs", ["Annotations"]),
            info.get("img_dirs", ["JPEGImages"]),
        ):
            ann_dir = dataset_root / ann_rel
            img_dir = dataset_root / img_rel
            if ann_dir.exists():
                annotations += parse_xml(ann_dir, img_dir if img_dir.exists() else dataset_root)
        if not annotations:
            for ann_dir in dataset_root.rglob("Annotations"):
                if ann_dir.is_dir():
                    parent  = ann_dir.parent
                    img_dir = next(
                        (parent / d for d in ("AllImages", "Images", "images", "JPEGImages")
                         if (parent / d).exists()),
                        dataset_root,
                    )
                    annotations += parse_xml(ann_dir, img_dir)

    elif fmt == "csv":
        csv_file = dataset_root / info.get("ann_file", "annotations.csv")
        if not csv_file.exists():
            found = list(dataset_root.glob("*.csv"))
            if found:
                csv_file = found[0]
        img_root = dataset_root / info.get("img_root", "images")
        if csv_file.exists():
            annotations += parse_csv(csv_file, img_root if img_root.exists() else dataset_root)

    elif fmt == "folder":
        img_root = dataset_root / info.get("img_root", ".")
        annotations += parse_folder(img_root if img_root.exists() else dataset_root)

    elif fmt == "video_folder":
        img_root = dataset_root / info.get("img_root", ".")
        annotations += parse_video_folder(img_root if img_root.exists() else dataset_root)

    elif fmt == "filename_prefix":
        img_root = dataset_root / info.get("img_root", ".")
        annotations += parse_filename_prefix(
            img_root if img_root.exists() else dataset_root,
            sep=info.get("prefix_sep", "__"),
            label_index=info.get("prefix_index", 0),
        )

    # ── Universal fallback: auto-detect actual format ────────────────────
    if not annotations:
        annotations = _autodetect_annotations(dataset_key, dataset_root)

    return annotations


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class IngestPipeline:
    """
    Orchestrates multi-dataset ingestion → unified train/val/test split.

    Parameters
    ----------
    datasets : dict[str, str | Path]
        Map of dataset_key → path_to_dataset_root.
        Keys must match entries in label_maps.yaml (e.g. "FLIR_Thermal").
    out_root : str | Path
        Destination for data/{split}/{class}/ tree.
    split_targets : dict
        {"train": N, "val": M, "test": K} — images per class per split.
    seed : int
        Random seed for reproducible splits.
    """

    def __init__(
        self,
        datasets: dict[str, str | Path],
        out_root: str | Path = "data/",
        split_targets: dict[str, int] | None = None,
        seed: int = 42,
    ):
        self.datasets      = {k: Path(v) for k, v in datasets.items()}
        self.out_root      = Path(out_root)
        self.split_targets = split_targets or {"train": 170, "val": 30, "test": 200}
        self.rng           = random.Random(seed)
        self.label_maps    = _load_label_maps()

    # ------------------------------------------------------------------

    def _collect_by_class(self) -> dict[str, list[dict]]:
        """
        Load all datasets, map labels, group by REATS class.
        Returns {class_id: [ann_dict, ...]}.
        """
        by_class: dict[str, list[dict]] = defaultdict(list)
        # Cap per-class pool at 4× the max target to avoid memory bloat from
        # popular civilian labels like "car" (appears millions of times across datasets)
        max_total = max(self.split_targets.values())
        pool_cap  = max_total * 4

        for ds_key, ds_root in self.datasets.items():
            if not ds_root.exists():
                print(f"  [ingest] WARNING: {ds_key} root not found: {ds_root} — skipping")
                continue

            ds_cfg = self.label_maps.get(ds_key)
            if ds_cfg is None:
                print(f"  [ingest] WARNING: {ds_key} not found in label_maps.yaml — skipping")
                continue

            is_thermal = ds_cfg.get("thermal", False)
            print(f"  [ingest] Loading {ds_key} from {ds_root} (thermal={is_thermal})")
            try:
                anns = load_dataset_annotations(ds_key, ds_root)
            except Exception as exc:
                print(f"           → ERROR: {exc} — skipping dataset")
                continue
            print(f"           → {len(anns)} raw annotations")

            mapped = discarded = capped = 0
            unmapped: Counter = Counter()
            for ann in anns:
                cls_id, matched = _resolve_label(ann["label"], ds_cfg, ann.get("area", 0))
                if not matched:
                    unmapped[ann["label"]] += 1
                    continue
                if cls_id is None or cls_id not in CLASSES:
                    discarded += 1
                    continue
                if len(by_class[cls_id]) >= pool_cap:
                    capped += 1   # mapped fine — pool for this class is already full
                    continue
                ann["_class"]    = cls_id
                ann["_thermal"]  = is_thermal
                ann["_dataset"]  = ds_key
                by_class[cls_id].append(ann)
                mapped += 1
            print(f"           → {mapped} mapped, {discarded} null-discarded, "
                  f"{capped} pool-capped")
            if unmapped:
                top = ", ".join(f"'{lbl}'×{n}" for lbl, n in unmapped.most_common(8))
                print(f"           → UNMAPPED ({sum(unmapped.values())} anns, "
                      f"{len(unmapped)} distinct): {top}")
                print(f"             fix: add these labels to ingestion/label_maps.yaml "
                      f"under {ds_key}")

        return by_class

    # ------------------------------------------------------------------

    def _existing_counts(self) -> dict[str, dict[str, int]]:
        """Count images already in out_root/{split}/{class}/."""
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        counts = {s: {c: 0 for c in CLASSES} for s in self.split_targets}
        for split in self.split_targets:
            for cls in CLASSES:
                d = self.out_root / split / cls
                if d.exists():
                    counts[split][cls] = sum(
                        1 for p in d.rglob("*")
                        if p.is_file() and p.suffix.lower() in exts
                    )
        return counts

    # ------------------------------------------------------------------

    def run(self, dry_run: bool = False) -> dict:
        """
        Run the full ingestion pipeline.
        Returns {split: {class: n_written}} counts.
        """
        W = 66
        print(f"\n{'='*W}")
        print("  REATS Multi-Dataset Ingestion Pipeline")
        print(f"  Datasets : {list(self.datasets.keys())}")
        print(f"  Output   : {self.out_root.resolve()}")
        print(f"  Targets  : {self.split_targets}")
        if dry_run:
            print("  *** DRY RUN — no files will be written ***")
        print(f"{'='*W}\n")

        by_class = self._collect_by_class()
        existing = self._existing_counts()
        written  = {s: {c: 0 for c in CLASSES} for s in self.split_targets}

        splits = list(self.split_targets.keys())  # ["train", "val", "test"]

        for cls in CLASSES:
            pool = list(by_class.get(cls, []))
            if not pool:
                print(f"  {cls:<14} — no annotations found in any dataset")
                continue

            self.rng.shuffle(pool)
            total_need  = sum(
                max(0, self.split_targets[s] - existing[s][cls])
                for s in splits
            )
            print(f"  {cls:<14}  pool={len(pool):>5}  need={total_need}")

            # Oversample if pool is smaller than need
            if len(pool) < total_need:
                repeats = (total_need // len(pool)) + 1
                pool    = (pool * repeats)[:total_need]
                self.rng.shuffle(pool)

            idx = 0
            for split in splits:
                need = max(0, self.split_targets[split] - existing[split][cls])
                if need == 0:
                    continue
                batch = pool[idx: idx + need]
                idx  += need

                for ann in batch:
                    patch = process_annotation(ann, cls, ann["_thermal"])
                    if patch is None:
                        continue
                    if not dry_run:
                        n = existing[split][cls] + written[split][cls]
                        dst = self.out_root / split / cls / f"{cls}_{n:05d}.png"
                        save_patch(patch, dst)
                    written[split][cls] += 1

        # Summary
        print(f"\n{'─'*W}")
        print("  Summary (written this run):")
        total = 0
        for split in splits:
            n = sum(written[split].values())
            total += n
            print(f"    {split:<6}  {n}")
        print(f"    {'TOTAL':<6}  {total}")
        print(f"{'='*W}\n")

        if not dry_run and total > 0:
            self._update_provenance(written)
        return written

    # ------------------------------------------------------------------

    def run_detection(
        self,
        split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
        dry_run: bool = False,
        detection_root: "str | Path | None" = None,
    ) -> dict:
        """
        Write a YOLO-format detection split — {detection_root}/{split}/images/*.jpg +
        {detection_root}/{split}/labels/*.txt (class cx cy w h, normalised) — for
        Module A (IRDetector.train() / MosaicDataset). This is separate from
        run(), which crops one classification patch per annotation for
        Module B: detection needs the *full* frame plus every box on it
        together, so annotations are grouped by source image here instead of
        by class.

        detection_root defaults to out_root/"detection" — deliberately NOT
        out_root/{split}/ itself. run()'s classification loader
        (torchvision ImageFolder, via build_loaders()) treats every
        subdirectory of out_root/{split}/ as a class name; writing
        out_root/{split}/images/ and .../labels/ there would add two bogus
        "classes" to Module B's training set, and ImageFolder hard-crashes
        the moment it finds one with zero image-extension files in it
        ("Found no valid file for the classes labels" — "labels" there is a
        literal class name, not a description). Keeping detection data in
        its own subtree makes that collision impossible regardless of what
        run() does.

        Only annotations carrying a real bbox are used. Folder/video-folder
        datasets with no localisation info use bbox=None ("whole image is
        the object", the convention run()/extract_roi rely on for
        classification crops) — labelling the entire frame as one box would
        be systematically wrong wherever the target doesn't fill the frame,
        so those are skipped here rather than turned into bad boxes.

        Resume-safe like run(): an image already written by a prior call
        (same deterministic stem, regardless of which split it landed in) is
        left untouched rather than reshuffled. self.rng is stateful and
        shared with run(), so a second call — e.g. after attaching a new
        Kaggle dataset mid-session and re-running ingestion — would otherwise
        draw a different shuffle and could reassign an already-written image
        to a different split, leaking the same physical image across
        train/val/test since nothing would remove the old copy.

        Returns {split: {"images": n, "boxes": n}}.
        """
        assert abs(sum(split_ratios) - 1.0) < 1e-6, "split_ratios must sum to 1.0"
        splits = ["train", "val", "test"]
        det_root = Path(detection_root) if detection_root is not None else self.out_root / "detection"

        by_class = self._collect_by_class()

        # Group by unique source image — the same frame can carry several
        # annotations (different objects, possibly different classes).
        groups: dict[tuple, list[dict]] = defaultdict(list)
        skipped_no_bbox = 0
        for anns in by_class.values():
            for ann in anns:
                if ann.get("bbox") is None:
                    skipped_no_bbox += 1
                    continue
                key = (ann["_dataset"], str(ann["image_path"]), ann.get("_frame_idx"))
                groups[key].append(ann)

        def _stem_for(key: tuple) -> str:
            ds_key, image_path, frame_idx = key
            stem_src = Path(image_path).stem
            readable = f"{ds_key}__{stem_src}" + (f"__f{frame_idx:04d}" if frame_idx is not None else "")
            readable = "".join(c if (c.isalnum() or c in "_-") else "_" for c in readable)[:150]
            # The readable prefix alone can collide: two different source
            # images can share a basename (per-video-frame folders, per-scene
            # dumps that both number frames from 0001.jpg), and sanitisation
            # collapses distinct characters (space, ':', etc.) to the same
            # "_". A short hash of the FULL, unsanitised key makes every stem
            # unique regardless of what the readable part collides on, while
            # staying deterministic across calls — required for resume-safety
            # (same source must always produce the same stem).
            digest = hashlib.sha1(f"{ds_key}|{image_path}|{frame_idx}".encode()).hexdigest()[:10]
            return f"{readable}_{digest}"

        # A stem counts as "already ingested" only if BOTH its image and
        # label file exist — not just the image. If a prior call was
        # interrupted (Kaggle timeout/OOM) between save_frame() and
        # write_yolo_labels() below, the .jpg exists but the .txt doesn't;
        # checking images/ alone would treat that stem as done and skip it
        # forever, permanently leaving an unlabelled image that either
        # breaks or silently mistrains MosaicDataset. Requiring both makes
        # an interrupted write self-heal on the next call instead.
        existing_stems: set[str] = set()
        for split in splits:
            img_dir = det_root / split / "images"
            lbl_dir = det_root / split / "labels"
            if img_dir.exists():
                existing_stems.update(
                    p.stem for p in img_dir.iterdir()
                    if p.is_file() and (lbl_dir / f"{p.stem}.txt").exists()
                )

        new_keys = [k for k in groups if _stem_for(k) not in existing_stems]
        resumed  = len(groups) - len(new_keys)

        self.rng.shuffle(new_keys)
        n       = len(new_keys)
        n_train = int(n * split_ratios[0])
        n_val   = int(n * split_ratios[1])
        n_test  = n - n_train - n_val
        split_of: dict[tuple, str] = {}
        for i, key in enumerate(new_keys):
            split_of[key] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")

        W = 66
        print(f"\n{'='*W}")
        print("  REATS Detection-Format Ingestion (YOLO images/ + labels/)")
        print(f"  Output        : {det_root.resolve()}")
        print(f"  Unique images : {len(groups)}  new={n}  "
              f"already-ingested={resumed}  (skipped {skipped_no_bbox} bbox-less annotation(s))")
        print(f"  Split ratios  : train={split_ratios[0]:.0%} "
              f"val={split_ratios[1]:.0%} test={split_ratios[2]:.0%}")
        if n > 0 and (n_val == 0 or n_test == 0):
            # Truncating int(n * ratio) on a small new-image batch can zero
            # out a split entirely (e.g. n=5 at 0.7/0.15/0.15 -> train=3
            # val=0 test=1). Each ingestion call only splits its OWN new
            # images — it doesn't rebalance across calls — so this is a
            # per-call gap, not necessarily a global one; it self-corrects
            # as more datasets get attached and ingested over time, but a
            # single small/first run can leave val or test with zero images.
            print(f"  WARNING: this run's split (train={n_train} val={n_val} test={n_test}) "
                  f"left a split empty — too few new images for these ratios. Training "
                  f"tolerates an empty val (mAP stays 0, no checkpoint saved) rather than "
                  f"crashing, but expect this to resolve once more datasets are ingested.")
        if dry_run:
            print("  *** DRY RUN — no files will be written ***")
        print(f"{'='*W}\n")

        written = {s: {"images": 0, "boxes": 0} for s in splits}
        decode_failed = 0

        for key in new_keys:
            anns = groups[key]
            split = split_of[key]
            ds_key, image_path, frame_idx = key
            is_thermal = anns[0]["_thermal"]

            img = load_frame(anns[0])
            if img is None:
                decode_failed += 1
                continue
            gray = to_ir_look(img, is_thermal)
            h, w = gray.shape[:2]
            if h == 0 or w == 0:
                decode_failed += 1
                continue

            boxes = []
            for ann in anns:
                x1, y1, x2, y2 = ann["bbox"]
                x1, x2 = sorted((max(0, min(x1, w)), max(0, min(x2, w))))
                y1, y2 = sorted((max(0, min(y1, h)), max(0, min(y2, h))))
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                cx, cy, bw, bh = (min(max(v, 0.001), 0.999) for v in (cx, cy, bw, bh))
                boxes.append((CLASSES.index(ann["_class"]), cx, cy, bw, bh))

            if not boxes:
                continue

            stem = _stem_for(key)

            if not dry_run:
                save_frame(gray, det_root / split / "images" / f"{stem}.jpg")
                write_yolo_labels(det_root / split / "labels" / f"{stem}.txt", boxes)

            written[split]["images"] += 1
            written[split]["boxes"]  += len(boxes)

        print(f"{'─'*W}")
        print("  Summary (written this run):")
        for split in splits:
            print(f"    {split:<6}  images={written[split]['images']:<5}  boxes={written[split]['boxes']}")
        if decode_failed:
            print(f"    (skipped {decode_failed} image(s) that failed to decode)")
        print(f"{'='*W}\n")

        return written

    # ------------------------------------------------------------------

    def _update_provenance(self, written: dict) -> None:
        """
        Merge this run's per-class counts into out_root/provenance.json as
        'real' images. generate_flir_fallback.py adds 'remapped'/'synthetic'
        counts to the same file, so downstream evaluation can report metrics
        per data-provenance bucket (architecture validation vs field-relevant).
        """
        import json
        path = self.out_root / "provenance.json"
        prov: dict = {}
        if path.exists():
            try:
                prov = json.loads(path.read_text())
            except Exception:
                prov = {}
        for class_counts in written.values():
            for cls, n in class_counts.items():
                if n:
                    entry = prov.setdefault(cls, {})
                    entry["real"] = entry.get("real", 0) + n
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(prov, indent=2, sort_keys=True))
        print(f"  Provenance manifest updated: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="REATS Multi-Dataset Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dataset keys (must match label_maps.yaml):
  FLIR_Thermal, FLIR_ADAS_v2, HIT_UAV, HIT_UAV_v2, HRSC2016,
  Ships_Aerial, Ships_Google_Earth, Ships_Vessels_Aerial, SWIM,
  CGI_Planes, Airbus_Aircraft, SwimmingPool_Car, Vehicle_Dataset,
  Aerial_Segmentation, Dataset2_Folders, Aerial_Roof_Seg

Example (Kaggle):
  python -m ingestion.pipeline \\
      --datasets FLIR_Thermal:/kaggle/input/flir-thermal-images-dataset \\
                 HIT_UAV:/kaggle/input/hit-uav \\
                 HRSC2016:/kaggle/input/hrsc2016 \\
                 Ships_Aerial:/kaggle/input/ship-detection \\
                 CGI_Planes:/kaggle/input/cgi-planes-in-satellite-imagery \\
                 Airbus_Aircraft:/kaggle/input/airbus-aircraft-detection \\
      --out /kaggle/working/data/
""",
    )
    parser.add_argument(
        "--datasets", nargs="+", metavar="KEY:PATH",
        help="dataset_key:/path pairs, e.g. FLIR_Thermal:/data/flir",
    )
    parser.add_argument("--out",     default="data/",  help="output data root")
    parser.add_argument("--train",   type=int, default=170)
    parser.add_argument("--val",     type=int, default=30)
    parser.add_argument("--test",    type=int, default=200)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--detection", action="store_true",
        help="Also write a YOLO-format detection split (data/{split}/images,labels) "
             "for Module A, alongside the classification split for Module B.",
    )
    args = parser.parse_args()

    datasets = {}
    for pair in (args.datasets or []):
        if ":" not in pair:
            print(f"ERROR: expected KEY:PATH, got: {pair}")
            sys.exit(1)
        key, path = pair.split(":", 1)
        datasets[key.strip()] = path.strip()

    if not datasets:
        parser.print_help()
        sys.exit(0)

    pipe = IngestPipeline(
        datasets=datasets,
        out_root=args.out,
        split_targets={"train": args.train, "val": args.val, "test": args.test},
        seed=args.seed,
    )
    pipe.run(dry_run=args.dry_run)
    if args.detection:
        pipe.run_detection(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
