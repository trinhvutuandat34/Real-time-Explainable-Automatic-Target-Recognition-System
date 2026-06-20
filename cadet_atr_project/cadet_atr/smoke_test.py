#!/usr/bin/env python3
"""
cadet_atr smoke test — no GPU, no real data required.

Run from cadet_atr_project/cadet_atr/:
    python smoke_test.py
"""

import sys
import time
import traceback
from pathlib import Path

_here = Path(__file__).parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import numpy as np
import torch

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"

results: list[tuple[str, bool, str]] = []


def check(name: str, fn):
    try:
        t0  = time.perf_counter()
        msg = fn() or ""
        ms  = (time.perf_counter() - t0) * 1000
        tag = f"{PASS}  ({ms:.0f} ms)"
        if msg:
            tag += f"  — {msg}"
        print(tag, f"  [{name}]")
        results.append((name, True, ""))
    except Exception as e:
        print(f"{FAIL}  [{name}]")
        tb = traceback.format_exc().strip().splitlines()
        for line in tb[-6:]:
            print(f"         {line}")
        results.append((name, False, str(e)))


# ---------------------------------------------------------------------------

def _test_config():
    from utils.config import CLASSES, NUM_CLASSES, cfg
    assert len(CLASSES) == NUM_CLASSES == 6
    return f"{NUM_CLASSES} classes: {CLASSES}"

check("config", _test_config)


def _test_augmentation():
    from data.augmentation import IRSyntheticAugmentation, IRRealAugmentation
    x  = torch.rand(2, 3, 224, 224)
    y1 = IRSyntheticAugmentation()(x)
    y2 = IRRealAugmentation()(x)
    assert y1.shape == x.shape and y2.shape == x.shape
    return "shape preserved for both augmentors"

check("augmentation", _test_augmentation)


def _test_build_model():
    from models.convnext import build_model, get_backbone, get_feature_dim
    from utils.config import NUM_CLASSES
    m = build_model("convnext_tiny", NUM_CLASSES, pretrained=False)
    m.eval()
    with torch.no_grad():
        out = m(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, NUM_CLASSES)
    dim = get_feature_dim(m)
    return f"logits {tuple(out.shape)}, feature_dim={dim}"

check("model.build", _test_build_model)


def _test_placeholder_data():
    import tempfile, os
    from data.dataset import generate_placeholder_synthetic, make_loaders
    with tempfile.TemporaryDirectory() as tmp:
        generate_placeholder_synthetic(tmp, n_per_class=6)
        from utils.config import cfg
        cfg.data_root = tmp
        cfg.batch_size = 4
        train_loader, val_loader = make_loaders(data_root=tmp, batch_size=4)
        batch = next(iter(train_loader))
        imgs, labels = batch
        assert imgs.shape[1:] == (3, 224, 224)
        return f"train batch {tuple(imgs.shape)}, labels {tuple(labels.shape)}"

check("dataset.placeholder", _test_placeholder_data)


def _test_histogram_matching():
    import tempfile
    from data.dataset import generate_placeholder_synthetic, make_loaders
    from adaptation.strategies import build_reference_histogram, apply_histogram_matching
    with tempfile.TemporaryDirectory() as tmp:
        generate_placeholder_synthetic(tmp, n_per_class=6)
        _, val_loader = make_loaders(data_root=tmp, batch_size=4)
        cdf = build_reference_histogram(val_loader)
        assert cdf.shape == (3, 256)
        x = torch.rand(2, 3, 224, 224)
        y = apply_histogram_matching(x, cdf, cdf)   # identity: src == tgt
        assert y.shape == x.shape
        return f"CDF shape {cdf.shape}, matched shape {tuple(y.shape)}"

check("strategy.histogram", _test_histogram_matching)


def _test_domain_random():
    import tempfile
    from data.dataset import generate_placeholder_synthetic, SyntheticIRDataset
    from adaptation.strategies import BackgroundSwapDataset
    from torch.utils.data import DataLoader
    with tempfile.TemporaryDirectory() as tmp:
        generate_placeholder_synthetic(tmp, n_per_class=6)
        from pathlib import Path
        synth = Path(tmp) / "synthetic"
        ds  = SyntheticIRDataset(str(synth), split="train")
        aug = BackgroundSwapDataset(ds, bg_pool=ds, swap_prob=0.8)
        img, label = aug[0]
        assert img.shape == (3, 224, 224)
        return f"augmented sample shape {tuple(img.shape)}"

check("strategy.domain_random", _test_domain_random)


def _test_dann_model():
    from models.convnext import build_model
    from adaptation.strategies import DANNModel
    from utils.config import NUM_CLASSES
    backbone = build_model("convnext_tiny", NUM_CLASSES, pretrained=False)
    dann = DANNModel(backbone.features, num_classes=NUM_CLASSES)
    dummy = torch.zeros(2, 3, 224, 224)
    cls_out = dann(dummy, return_domain=False)
    cls_out2, dom_out = dann(dummy, return_domain=True)
    assert cls_out.shape  == (2, NUM_CLASSES)
    assert cls_out2.shape == (2, NUM_CLASSES)
    assert dom_out.shape  == (2, 2)
    return f"class logits {tuple(cls_out.shape)}, domain logits {tuple(dom_out.shape)}"

check("strategy.dann_model", _test_dann_model)


def _test_evaluator():
    from models.convnext import build_model
    from evaluation.evaluator import accuracy
    from utils.config import NUM_CLASSES
    import tempfile
    from data.dataset import generate_placeholder_synthetic, make_loaders
    with tempfile.TemporaryDirectory() as tmp:
        generate_placeholder_synthetic(tmp, n_per_class=6)
        model = build_model("convnext_tiny", NUM_CLASSES, pretrained=False)
        _, val_loader = make_loaders(data_root=tmp, batch_size=4)
        acc = accuracy(model, val_loader)
        assert 0.0 <= acc <= 1.0
        return f"random model acc={acc:.3f} (expected ~1/{NUM_CLASSES} = {1/NUM_CLASSES:.3f})"

check("evaluator.accuracy", _test_evaluator)


def _test_grad_cam():
    from models.convnext import build_model
    from utils.visualise import grad_cam_overlay
    from utils.config import NUM_CLASSES
    model = build_model("convnext_tiny", NUM_CLASSES, pretrained=False)
    x = torch.zeros(1, 3, 224, 224)
    overlay = grad_cam_overlay(model, x, class_idx=0, device="cpu")
    assert overlay.shape == (224, 224, 3)
    return f"overlay shape {overlay.shape}"

check("visualise.grad_cam", _test_grad_cam)

# ---------------------------------------------------------------------------

print()
print("=" * 54)
passed = sum(1 for _, ok, _ in results if ok)
total  = len(results)
print(f"  Results: {passed}/{total} passed")
print("=" * 54)
for name, ok, msg in results:
    icon = "✓" if ok else "✗"
    print(f"  {icon}  {name}" + (f"  — {msg}" if not ok else ""))
print()

sys.exit(0 if passed == total else 1)
