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
    parse_folder, parse_video_folder,
)
from ingestion.preprocessor import process_annotation, save_patch

# ---------------------------------------------------------------------------
# Label map loader
# ---------------------------------------------------------------------------

def _load_label_maps() -> dict:
    with open(_HERE / "label_maps.yaml") as f:
        return yaml.safe_load(f)


def _norm_label(s: str) -> str:
    """Normalise a raw label for map lookup: lowercase, strip, unify separators."""
    return str(s).strip().lower().replace("-", "_").replace(" ", "_")


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
    "SWIM": {
        "format": "yolo",
        "label_dirs": ["labels"],
        "img_dirs":   ["images"],
        "yolo_classes": ["ship", "wake"],
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
    for lbl_dir in dataset_root.rglob("labels"):
        if not lbl_dir.is_dir():
            continue
        img_dir = lbl_dir.parent / "images"
        anns = parse_yolo(lbl_dir, img_dir if img_dir.exists() else dataset_root, classes)
        if anns:
            return anns

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
            else:
                for cand in dataset_root.rglob("*.json"):
                    cname = cand.name.lower()
                    if any(k in cname for k in ("annotation", "train", "val", "coco")):
                        try:
                            annotations += parse_coco(cand, dataset_root)
                            break
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
                    annotations += parse_yolo(lbl_dir, dataset_root, classes)

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


if __name__ == "__main__":
    main()
