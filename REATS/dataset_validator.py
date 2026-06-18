#!/usr/bin/env python3
"""
REATS Dataset Validator & Organizer
Checks train/val/test split (170/30/200) and class structure.

Usage:
    python dataset_validator.py                          # validate data/
    python dataset_validator.py --root /path/to/data/   # validate a custom root
    python dataset_validator.py --source raw/ --organize # auto-split & copy
    python dataset_validator.py --source raw/ --organize --dry-run
    python dataset_validator.py --source raw/ --organize --train 50 --val 10 --test 50
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

CLASSES        = ["F16", "LYNX", "MiG19", "MiG21", "PKG", "PTG"]
SPLITS         = ["train", "val", "test"]
TARGETS        = {"train": 170, "val": 30, "test": 200}   # per class, per paper
IMG_EXTS       = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

OK   = "✓"
FAIL = "✗"
WARN = "⚠"
W    = 60


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def find_images(folder: Path) -> list:
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def _match_class(name: str):
    """Case-insensitive class name lookup."""
    return next((c for c in CLASSES if c.lower() == name.lower()), None)


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def validate(data_root: Path) -> dict:
    """
    Scan data_root/{split}/{class}/ and return counts + unknown folders.
    """
    counts  = {s: {c: 0 for c in CLASSES} for s in SPLITS}
    unknown = defaultdict(int)

    for split in SPLITS:
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        for d in split_dir.iterdir():
            if not d.is_dir():
                continue
            matched = _match_class(d.name)
            if matched:
                counts[split][matched] = len(find_images(d))
            else:
                unknown[f"{split}/{d.name}"] += len(find_images(d))

    return {"counts": counts, "unknown": dict(unknown)}


def print_report(report: dict, data_root: Path) -> bool:
    """Pretty-print the validation report. Returns True if dataset is complete."""
    counts  = report["counts"]
    unknown = report["unknown"]
    cw      = max(len(c) for c in CLASSES)

    print("\n" + "=" * W)
    print(f"  REATS Dataset Report")
    print(f"  Root: {data_root.resolve()}")
    print("=" * W)

    all_ok = True
    for split in SPLITS:
        target = TARGETS[split]
        print(f"\n  [{split.upper()}]  target: {target} images/class")
        print(f"  {'Class':<{cw}}  {'Found':>6}  {'Need':>6}  Status")
        print("  " + "-" * (cw + 28))
        for cls in CLASSES:
            n    = counts[split][cls]
            need = max(0, target - n)
            if n == target:
                status = f"{OK} OK"
            elif n > target:
                status = f"{WARN} +{n - target} extra"
            elif n == 0:
                status = f"{FAIL} MISSING"
                all_ok = False
            else:
                status = f"{FAIL} need {need} more"
                all_ok = False
            print(f"  {cls:<{cw}}  {n:>6}  {need:>6}  {status}")

    # Grand total row
    grand        = sum(counts[s][c] for s in SPLITS for c in CLASSES)
    grand_target = sum(TARGETS.values()) * len(CLASSES)
    pct          = 100 * grand / grand_target if grand_target else 0.0

    print("\n" + "-" * W)
    print(f"  Total:  {grand} / {grand_target} images  ({pct:.1f}%)")

    # Per-class totals (useful to spot imbalance)
    print(f"\n  Per-class totals (train+val+test):")
    print(f"  {'Class':<{cw}}  {'Total':>6}  {'Target':>7}  {'%':>6}")
    print("  " + "-" * (cw + 24))
    per_class_target = sum(TARGETS.values())          # 400
    for cls in CLASSES:
        n    = sum(counts[s][cls] for s in SPLITS)
        icon = OK if n >= per_class_target else FAIL
        print(f"  {cls:<{cw}}  {n:>6}  {per_class_target:>7}  "
              f"{100*n/per_class_target if per_class_target else 0:>5.1f}%  {icon}")

    if unknown:
        print(f"\n  {WARN} Unknown folders (not in CLASSES — check naming):")
        for path, cnt in sorted(unknown.items()):
            print(f"       {path}  ({cnt} images)")

    print("\n" + "=" * W)
    if all_ok:
        print(f"  {OK} Dataset complete — ready for training.\n")
    else:
        print(f"  {FAIL} Dataset incomplete. Summary of what's still needed:\n")
        for split in SPLITS:
            missing = {c: max(0, TARGETS[split] - counts[split][c])
                       for c in CLASSES if counts[split][c] < TARGETS[split]}
            if missing:
                print(f"     {split}/")
                for cls, n in missing.items():
                    print(f"       {cls:<{cw}} → {n} images needed")
        print()
    print("=" * W + "\n")

    return all_ok


# ---------------------------------------------------------------------------
# Organize
# ---------------------------------------------------------------------------

def organize(
    src_dir:   Path,
    data_root: Path,
    ratios:    tuple,
    seed:      int  = 42,
    dry_run:   bool = False,
):
    """
    Copy images from src_dir/{class}/ into data_root/{split}/{class}/.

    src_dir layout expected:
        src_dir/
          F16/    ← any number of images (recursive)
          LYNX/
          MiG19/
          ...

    Existing images in data_root are preserved; only the shortfall is filled.
    """
    random.seed(seed)
    split_sizes = dict(zip(SPLITS, ratios))

    print(f"\n{'=' * W}")
    print(f"  Organize: {src_dir} → {data_root}")
    if dry_run:
        print(f"  {WARN} DRY RUN — no files will be written")
    print(f"{'=' * W}\n")

    # First pass: count what's already in data_root
    existing = validate(data_root)["counts"]

    found_any = False
    total_copied = 0

    for cls_dir in sorted(src_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls = _match_class(cls_dir.name)
        if cls is None:
            print(f"  {WARN} Skipping unrecognised folder: {cls_dir.name}")
            continue

        images = find_images(cls_dir)
        random.shuffle(images)
        print(f"  {cls:<8}  {len(images):>4} source images")

        cursor = 0
        for split, target in split_sizes.items():
            already = existing[split][cls]
            need    = max(0, target - already)
            batch   = images[cursor: cursor + need]
            dest_dir = data_root / split / cls

            if not dry_run and need > 0:
                dest_dir.mkdir(parents=True, exist_ok=True)

            copied = 0
            for img in batch:
                # Avoid filename collisions by prefixing with class+index
                dest = dest_dir / f"{cls}_{cursor + copied:05d}{img.suffix.lower()}"
                if not dry_run:
                    shutil.copy2(img, dest)
                copied += 1

            total_copied += copied
            status = f"{copied} copied  (already had {already}/{target})"
            if need == 0:
                status = f"{OK} already complete ({already}/{target})"
            elif len(batch) < need:
                status = (f"{FAIL} only {copied} available, "
                          f"still need {need - len(batch)} more")
            print(f"           {split:<6}  {status}")

            cursor += need
        found_any = True
        print()

    if not found_any:
        print(f"  {FAIL} No matching class folders found in {src_dir}\n"
              f"       Expected subfolders named: {', '.join(CLASSES)}\n")
        return

    verb = "Would copy" if dry_run else "Copied"
    print(f"  {verb} {total_copied} images total.")
    print(f"{'=' * W}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="REATS Dataset Validator & Organizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dataset_validator.py
  python dataset_validator.py --root /data/reats/
  python dataset_validator.py --source raw/ --organize
  python dataset_validator.py --source raw/ --organize --dry-run
  python dataset_validator.py --source raw/ --organize --train 50 --val 10 --test 50
""",
    )
    parser.add_argument("--root",     default="data/",
                        help="path to data/ root (default: data/)")
    parser.add_argument("--source",   default=None,
                        help="source folder containing class subfolders (for --organize)")
    parser.add_argument("--organize", action="store_true",
                        help="copy & split images from --source into --root")
    parser.add_argument("--dry-run",  action="store_true",
                        help="show what organize would do without copying")
    parser.add_argument("--train",    type=int, default=170,
                        help="train images per class (default: 170)")
    parser.add_argument("--val",      type=int, default=30,
                        help="val images per class (default: 30)")
    parser.add_argument("--test",     type=int, default=200,
                        help="test images per class (default: 200)")
    parser.add_argument("--seed",     type=int, default=42,
                        help="random seed for shuffle (default: 42)")
    args = parser.parse_args()

    data_root = Path(args.root)

    if args.organize:
        if args.source is None:
            parser.error("--organize requires --source <folder>")
        organize(
            src_dir   = Path(args.source),
            data_root = data_root,
            ratios    = (args.train, args.val, args.test),
            seed      = args.seed,
            dry_run   = args.dry_run,
        )

    report = validate(data_root)
    print_report(report, data_root)


if __name__ == "__main__":
    main()
