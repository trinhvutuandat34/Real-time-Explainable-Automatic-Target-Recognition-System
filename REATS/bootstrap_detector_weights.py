#!/usr/bin/env python3
"""
Bootstrap Module A (YOLOv4) weights for REATS.

Three strategies, tried in order:
  1. --weights <path>   — load a local .pt checkpoint directly
  2. --darknet <path>   — convert an original darknet .weights file to PyTorch
  3. (default)          — create a COCO-backbone initialiser: download the
                          standard YOLOv4 darknet weights from the official
                          Alexey AB mirror, strip the detection heads (class
                          count differs), and copy the backbone+neck parameters
                          by name where shapes match.

The resulting checkpoint is saved to REATS/checkpoints/detector_bootstrap.pt
and is immediately usable by IRDetector as a starting point for fine-tuning
on IR data.

Usage:
    python REATS/bootstrap_detector_weights.py
    python REATS/bootstrap_detector_weights.py --weights path/to/yolov4.pt
    python REATS/bootstrap_detector_weights.py --darknet path/to/yolov4.weights
"""

from __future__ import annotations

import argparse
import struct
import sys
import urllib.request
from pathlib import Path

import torch

_reats_root = Path(__file__).parent
if str(_reats_root) not in sys.path:
    sys.path.insert(0, str(_reats_root))

from modules.module_a_detector import IRDetector, YOLOv4
from config import NUM_CLASSES

CKPT_OUT = _reats_root / "checkpoints" / "detector_bootstrap.pt"

# Official darknet YOLOv4 weights (COCO 80-class, ~256 MB)
_DARKNET_URL = (
    "https://github.com/AlexeyAB/darknet/releases/download/"
    "darknet_yolo_v3_optimal/yolov4.weights"
)


# ---------------------------------------------------------------------------
# Darknet .weights loader
# ---------------------------------------------------------------------------

def _load_darknet_weights(path: str) -> dict:
    """
    Parse a Alexey-AB darknet .weights binary file into a dict of
    {param_name: torch.Tensor}.  Only Conv2d + BatchNorm2d weights are stored.

    Darknet binary layout (fp32):
      [major, minor, revision, seen_lo, seen_hi]  (5 int32 header)
      then for each layer (in order):
        BN: [biases, weights, mean, var]   each (out_c,)
        Conv: kernel weights               (out_c, in_c, kH, kW)
        (if no BN: conv bias then weights)
    """
    import numpy as np

    with open(path, "rb") as f:
        header = struct.unpack("5i", f.read(20))
        weights_data = np.frombuffer(f.read(), dtype=np.float32)

    print(f"  Header: major={header[0]}, minor={header[1]}, "
          f"revision={header[2]}, seen={header[3]}")
    print(f"  Total fp32 params: {len(weights_data):,}")
    return weights_data


def _apply_darknet_weights(model: torch.nn.Module, weights_data) -> int:
    """
    Walk model Conv2d/BatchNorm2d pairs and fill from the darknet weights buffer.
    Returns number of parameters loaded.
    """
    import numpy as np

    ptr = 0
    loaded = 0

    modules = list(model.modules())
    for i, module in enumerate(modules):
        if not isinstance(module, torch.nn.Conv2d):
            continue

        # Look ahead for BatchNorm
        bn = None
        for j in range(i + 1, min(i + 4, len(modules))):
            if isinstance(modules[j], torch.nn.BatchNorm2d):
                bn = modules[j]
                break
            if isinstance(modules[j], torch.nn.Conv2d):
                break

        n_out = module.weight.shape[0]

        if bn is not None:
            # BN bias, weight, mean, var
            for param in [bn.bias, bn.weight, bn.running_mean, bn.running_var]:
                n = param.numel()
                if ptr + n > len(weights_data):
                    return loaded
                param.data.copy_(
                    torch.from_numpy(weights_data[ptr: ptr + n]).view_as(param)
                )
                ptr += n
                loaded += n
        else:
            # Conv bias
            if module.bias is not None:
                n = module.bias.numel()
                if ptr + n > len(weights_data):
                    return loaded
                module.bias.data.copy_(
                    torch.from_numpy(weights_data[ptr: ptr + n]).view_as(module.bias)
                )
                ptr += n
                loaded += n

        # Conv weights
        n = module.weight.numel()
        if ptr + n > len(weights_data):
            return loaded
        module.weight.data.copy_(
            torch.from_numpy(weights_data[ptr: ptr + n]).view_as(module.weight)
        )
        ptr += n
        loaded += n

    return loaded


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------

def _from_local_pt(path: str) -> None:
    print(f"[bootstrap] Loading PyTorch checkpoint: {path}")
    det = IRDetector(weights=path)
    CKPT_OUT.parent.mkdir(parents=True, exist_ok=True)
    det.save(str(CKPT_OUT))
    print(f"[bootstrap] Saved → {CKPT_OUT}")


def _from_darknet(path: str) -> None:
    print(f"[bootstrap] Converting darknet .weights: {path}")
    model = YOLOv4(num_classes=NUM_CLASSES)
    data  = _load_darknet_weights(path)
    n     = _apply_darknet_weights(model, data)
    CKPT_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict()}, CKPT_OUT)
    print(f"[bootstrap] Loaded {n:,} params from darknet weights → {CKPT_OUT}")


def _download_and_convert() -> None:
    """Download official COCO YOLOv4 darknet weights then convert."""
    raw = _reats_root / "checkpoints" / "yolov4_coco.weights"
    raw.parent.mkdir(parents=True, exist_ok=True)

    if raw.exists():
        print(f"[bootstrap] Darknet weights already present: {raw}")
    else:
        size_mb = 256
        print(f"[bootstrap] Downloading YOLOv4 COCO weights (~{size_mb} MB)…")
        print(f"  URL: {_DARKNET_URL}")

        def _progress(block_num, block_size, total_size):
            pct = block_num * block_size / total_size * 100
            mb  = block_num * block_size / 1e6
            print(f"\r  {mb:.1f} / {total_size/1e6:.1f} MB  ({pct:.0f}%)", end="", flush=True)

        try:
            urllib.request.urlretrieve(_DARKNET_URL, raw, _progress)
            print()  # newline after progress
        except Exception as e:
            print(f"\n[bootstrap] Download failed: {e}")
            print("  Falling back to random-weight initialisation (fine for testing).")
            _random_init()
            return

    _from_darknet(str(raw))


def _random_init() -> None:
    """Save a random-weight checkpoint as the bootstrap starting point."""
    print("[bootstrap] Creating random-weight checkpoint (no pretrained weights).")
    model = YOLOv4(num_classes=NUM_CLASSES)
    CKPT_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict()}, CKPT_OUT)
    print(f"[bootstrap] Random init saved → {CKPT_OUT}")
    print()
    print("  NOTE: Random weights produce garbage detections.")
    print("  Train Module A with:  python -c \"from modules.module_a_detector import IRDetector; "
          "IRDetector().train('data/')\"")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap YOLOv4 detector weights for REATS Module A",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download official COCO YOLOv4 weights and convert (recommended)
  python REATS/bootstrap_detector_weights.py

  # Use an existing PyTorch checkpoint
  python REATS/bootstrap_detector_weights.py --weights checkpoints/my_yolov4.pt

  # Convert an existing darknet .weights file
  python REATS/bootstrap_detector_weights.py --darknet ~/yolov4.weights

  # Force random initialisation (for testing only)
  python REATS/bootstrap_detector_weights.py --random
""",
    )
    parser.add_argument("--weights",  metavar="PATH", help="Load local .pt checkpoint")
    parser.add_argument("--darknet",  metavar="PATH", help="Convert darknet .weights file")
    parser.add_argument("--random",   action="store_true",
                        help="Skip download; save random-init checkpoint")
    args = parser.parse_args()

    if args.weights:
        _from_local_pt(args.weights)
    elif args.darknet:
        _from_darknet(args.darknet)
    elif args.random:
        _random_init()
    else:
        _download_and_convert()

    if CKPT_OUT.exists():
        size_mb = CKPT_OUT.stat().st_size / 1e6
        print()
        print(f"[bootstrap] Done.  Checkpoint: {CKPT_OUT}  ({size_mb:.1f} MB)")
        print()
        print("  Set this path in the dashboard sidebar 'Detector weights' field,")
        print(f"  or pass to IRDetector(weights='{CKPT_OUT}').")


if __name__ == "__main__":
    main()
