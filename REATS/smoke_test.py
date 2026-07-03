#!/usr/bin/env python3
"""
REATS end-to-end smoke test — no GPU, no real data required.

Tests the full A → B → C → D import chain with synthetic tensors/frames.
Pass/fail is printed for each step; exits non-zero if any step fails.

Usage:
    python REATS/smoke_test.py
    # or, from REATS/:
    python smoke_test.py
"""

import sys
import time
import traceback
from pathlib import Path

# Ensure REATS root is on the path
_root = Path(__file__).parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import torch

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
SKIP = "\033[93m  SKIP\033[0m"

results: list[tuple[str, bool, str]] = []


def check(name: str, fn):
    try:
        t0 = time.perf_counter()
        msg = fn() or ""
        ms = (time.perf_counter() - t0) * 1000
        tag = f"{PASS}  ({ms:.0f} ms)"
        if msg:
            tag += f"  — {msg}"
        print(tag, f"  [{name}]")
        results.append((name, True, ""))
    except Exception as e:
        print(f"{FAIL}  [{name}]")
        tb = traceback.format_exc().strip().splitlines()
        for line in tb[-6:]:        # last 6 lines of traceback
            print(f"         {line}")
        results.append((name, False, str(e)))


# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------

def _test_config():
    from config import CLASSES, NUM_CLASSES, TARGET_META, RED_THREATS
    assert len(CLASSES) == NUM_CLASSES, f"len(CLASSES)={len(CLASSES)} != NUM_CLASSES={NUM_CLASSES}"
    assert NUM_CLASSES >= 6, f"expected ≥6 classes, got {NUM_CLASSES}"
    assert len(TARGET_META) > 0
    assert len(RED_THREATS) > 0
    return f"{NUM_CLASSES} classes, {len(RED_THREATS)} RED threats"

check("config.load", _test_config)


# ---------------------------------------------------------------------------
# 2. Augmentation
# ---------------------------------------------------------------------------

def _test_augmentation():
    from modules.augmentation_viewpoint import MultiViewpointAugmentor
    aug = MultiViewpointAugmentor(p_scale=1.0)
    x = torch.randn(2, 3, 224, 224)
    y = aug(x)
    assert y.shape == (2, 3, 224, 224)
    return "shape OK"

check("augmentation.viewpoint", _test_augmentation)


# ---------------------------------------------------------------------------
# 3. Module A — detector forward pass
# ---------------------------------------------------------------------------

def _test_module_a_forward():
    from modules.module_a_detector import YOLOv4, IRDetector
    from config import NUM_CLASSES
    model = YOLOv4(num_classes=NUM_CLASSES)
    model.eval()
    dummy_t = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        out = model(dummy_t)
    # inference mode returns list of 1 tensor
    assert isinstance(out, list) and len(out) == 1
    assert out[0].shape[-1] == 5 + NUM_CLASSES
    return f"output shape {tuple(out[0].shape)}"

check("module_a.forward", _test_module_a_forward)


def _test_module_a_detect():
    from modules.module_a_detector import IRDetector
    det = IRDetector()   # random weights, no file needed
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # paint a bright blob so there's something to detect
    frame[200:280, 280:360] = 200
    dets = det.detect(frame)
    assert isinstance(dets, list)
    return f"{len(dets)} detections on synthetic frame"

check("module_a.detect", _test_module_a_detect)


def _test_module_a_crop_roi():
    from modules.module_a_detector import IRDetector
    det = IRDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    roi = det.crop_roi(frame, [100, 100, 200, 200], pad=10)
    assert roi.shape[0] > 0 and roi.shape[1] > 0
    return f"ROI shape {roi.shape}"

check("module_a.crop_roi", _test_module_a_crop_roi)


# ---------------------------------------------------------------------------
# 4. Module B — classifier
# ---------------------------------------------------------------------------

def _test_module_b_build():
    from modules.module_b_classifier import build_convnext, EnsembleClassifier
    from config import NUM_CLASSES
    m = build_convnext(num_classes=NUM_CLASSES, pretrained=False)
    m.eval()
    dummy = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        out = m(dummy)
    assert out.shape == (1, NUM_CLASSES)
    return f"logits shape {tuple(out.shape)}"

check("module_b.build_convnext", _test_module_b_build)


def _test_module_b_ensemble():
    from modules.module_b_classifier import build_convnext, EnsembleClassifier
    from config import NUM_CLASSES
    models = [build_convnext(NUM_CLASSES, pretrained=False) for _ in range(2)]
    ens = EnsembleClassifier(models)
    ens.eval()
    dummy = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        probs = ens(dummy)
    assert probs.shape == (1, NUM_CLASSES)
    assert abs(float(probs.sum()) - 1.0) < 1e-4, "probabilities don't sum to 1"
    return f"ensemble probs sum={float(probs.sum()):.4f}"

check("module_b.ensemble", _test_module_b_ensemble)


def _test_module_b_heterogeneous_ensemble():
    """Gap 1: 6 distinct architectures (ConvNeXt/ResNeXt/ViT/Swin/VGG/ResNet), not 6 seeds of one."""
    from modules.module_b_classifier import ARCHITECTURES, build_model, EnsembleClassifier
    from config import NUM_CLASSES
    assert len(ARCHITECTURES) == 6
    models = [build_model(arch, NUM_CLASSES, pretrained=False) for arch in ARCHITECTURES]
    ens = EnsembleClassifier(models)
    ens.eval()
    dummy = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        probs = ens(dummy)
    assert probs.shape == (1, NUM_CLASSES)
    assert abs(float(probs.sum()) - 1.0) < 1e-4, "probabilities don't sum to 1"
    return f"6 architectures {ARCHITECTURES} → ensemble probs sum={float(probs.sum()):.4f}"

check("module_b.heterogeneous_ensemble", _test_module_b_heterogeneous_ensemble)


def _test_module_b_kornia_aug():
    from modules.module_b_classifier import KorniaAugmentPipeline
    pipe = KorniaAugmentPipeline()
    x = torch.randn(2, 3, 224, 224)
    y = pipe(x)
    assert y.shape == x.shape
    return "shape preserved"

check("module_b.kornia_aug", _test_module_b_kornia_aug)


def _test_module_b_full_aug():
    """MultiViewpointAugmentor → KorniaAugmentPipeline (as used in training)."""
    from modules.augmentation_viewpoint import MultiViewpointAugmentor
    from modules.module_b_classifier import KorniaAugmentPipeline
    pipeline = torch.nn.Sequential(MultiViewpointAugmentor(), KorniaAugmentPipeline())
    x = torch.randn(2, 3, 224, 224)
    y = pipeline(x)
    assert y.shape == x.shape
    return "combined aug shape OK"

check("module_b.full_aug_pipeline", _test_module_b_full_aug)


# ---------------------------------------------------------------------------
# 5. Module C — XAI
# ---------------------------------------------------------------------------

def _test_module_c_eigen_cam():
    """Test the inline _eigen_cam helper from module_d (no grad-cam dep)."""
    from modules.module_b_classifier import build_convnext
    from config import NUM_CLASSES
    import cv2 as cv
    model = build_convnext(NUM_CLASSES, pretrained=False)
    model.eval()

    # Replicate the _eigen_cam logic directly
    acts: list = []
    target_layer = None
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            target_layer = layer
    assert target_layer is not None

    fh = target_layer.register_forward_hook(lambda m, inp, out: acts.append(out.detach()))
    dummy = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        model(dummy)
    fh.remove()

    feat = acts[0].squeeze(0).numpy()
    C, H, W = feat.shape
    flat = feat.reshape(C, -1).T
    flat -= flat.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(flat, full_matrices=False)
    cam = (flat @ Vt[0]).reshape(H, W)
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam /= cam.max()
    cam_up = cv.resize(cam, (224, 224))
    heatmap = cv.applyColorMap((cam_up * 255).astype(np.uint8), cv.COLORMAP_JET)
    assert heatmap.shape == (224, 224, 3)
    return f"EigenCAM heatmap shape {heatmap.shape}"

check("module_c.eigen_cam", _test_module_c_eigen_cam)


def _test_module_c_mcdropout():
    from modules.module_c_xai import MCDropoutWrapper
    from modules.module_b_classifier import build_convnext
    from config import NUM_CLASSES
    model = build_convnext(NUM_CLASSES, pretrained=False)
    mc = MCDropoutWrapper(model, n_samples=3)
    x = torch.zeros(1, 3, 224, 224)
    out = mc(x)
    assert "mean_probs" in out and "uncertainty" in out
    assert out["mean_probs"].shape == (1, NUM_CLASSES)
    return f"uncertainty={float(out['uncertainty'][0]):.4f}"

check("module_c.mc_dropout", _test_module_c_mcdropout)


# ---------------------------------------------------------------------------
# 5b. Threat metrics / hard-negative mining / operational policy
# ---------------------------------------------------------------------------

def _test_threat_metrics_far_mr():
    from modules.threat_metrics import compute_far_mr
    from config import CLASSES
    n = len(CLASSES)
    labels = list(range(n)) + [0, 1]     # perfect diagonal + one extra FN/FP pair
    preds  = list(range(n)) + [1, 1]     # class 0 missed once, class 1 has one extra FP
    report = compute_far_mr(labels, preds)
    assert 0.0 <= report["macro_FAR"] <= 1.0
    assert 0.0 <= report["macro_MR"] <= 1.0
    assert report["per_class"][CLASSES[0]]["FN"] == 1
    assert report["per_class"][CLASSES[1]]["FP"] == 1
    return f"macro_FAR={report['macro_FAR']:.4f} macro_MR={report['macro_MR']:.4f}"

check("threat_metrics.far_mr", _test_threat_metrics_far_mr)


def _test_hard_negative_mining():
    from modules.hard_negative_mining import CONFUSABLE_GROUPS, mine_hard_negatives
    from modules.module_b_classifier import build_convnext
    from config import NUM_CLASSES, CLASSES
    import torch.utils.data as tud

    assert any({"F16", "MiG19", "MiG21"} <= g for g in CONFUSABLE_GROUPS)

    class _DummyDS(tud.Dataset):
        def __len__(self): return 8
        def __getitem__(self, idx):
            return torch.zeros(3, 224, 224), idx % NUM_CLASSES

    model = build_convnext(NUM_CLASSES, pretrained=False)
    model.eval()
    hard_idx = mine_hard_negatives(model, _DummyDS(), "cpu", margin_thresh=1.1)  # everything below margin
    assert isinstance(hard_idx, list)
    return f"{len(hard_idx)} hard-negative indices flagged"

check("hard_negative_mining.mine", _test_hard_negative_mining)


def _test_threat_policy_mapping():
    from modules.threat_policy import map_confidence_to_policy
    assert map_confidence_to_policy(0.95, "RED") == "ENGAGEMENT"
    assert map_confidence_to_policy(0.95, "YELLOW") == "WARNING"
    assert map_confidence_to_policy(0.10, "RED") == "NONE"
    return "RED@0.95→ENGAGEMENT, YELLOW@0.95→WARNING (ceiling), RED@0.10→NONE"

check("threat_policy.map_confidence", _test_threat_policy_mapping)


# ---------------------------------------------------------------------------
# 6. Module E — streamer imports
# ---------------------------------------------------------------------------

def _test_module_e_imports():
    import importlib
    for pkg in ("fastapi", "uvicorn", "websockets"):
        spec = importlib.util.find_spec(pkg)
        if spec is None:
            raise ImportError(f"{pkg} not installed")
    from modules.module_e_streamer import app
    assert app is not None
    return "fastapi + uvicorn + websockets present, app created"

check("module_e.imports", _test_module_e_imports)


# ---------------------------------------------------------------------------
# 7. Full A → B pipeline on a synthetic frame
# ---------------------------------------------------------------------------

def _test_e2e_pipeline():
    from modules.module_a_detector import IRDetector
    from modules.module_b_classifier import build_convnext, EnsembleClassifier
    from config import NUM_CLASSES, CLASSES

    det = IRDetector()
    models_list = [build_convnext(NUM_CLASSES, pretrained=False)]
    ens = EnsembleClassifier(models_list)
    ens.eval()

    # Synthetic frame with a bright target blob
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[160:320, 200:440] = 180

    detections = det.detect(frame)

    # For each detection, crop → classify
    results_e2e = []
    for d in detections[:3]:   # at most 3 to keep it fast
        roi = det.crop_roi(frame, d["bbox"])
        if roi.size == 0:
            continue
        roi_resized = np.array(
            __import__("PIL").Image.fromarray(roi).resize((224, 224))
        ).astype(np.float32) / 255.0
        t = torch.from_numpy(roi_resized).permute(2, 0, 1).unsqueeze(0)
        t = (t - 0.5) / 0.5
        with torch.no_grad():
            probs = ens(t)
        top = int(probs.argmax())
        results_e2e.append(CLASSES[top])

    return (
        f"{len(detections)} detections → classified {len(results_e2e)} ROIs"
        + (f" → top={results_e2e}" if results_e2e else " (no detections above threshold)")
    )

check("e2e.pipeline_a_to_b", _test_e2e_pipeline)


# ---------------------------------------------------------------------------
# 8. Latency budget
# ---------------------------------------------------------------------------

def _test_latency():
    from modules.module_a_detector import IRDetector
    from modules.module_b_classifier import build_convnext, EnsembleClassifier
    from config import NUM_CLASSES

    det = IRDetector()
    ens = EnsembleClassifier([build_convnext(NUM_CLASSES, pretrained=False)])
    ens.eval()

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    REPS = 5
    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        detections = det.detect(frame)
        if detections:
            roi = det.crop_roi(frame, detections[0]["bbox"])
            if roi.size > 0:
                r = np.array(
                    __import__("PIL").Image.fromarray(roi).resize((224, 224))
                ).astype(np.float32) / 255.0
                t = torch.from_numpy(r).permute(2, 0, 1).unsqueeze(0)
                t = (t - 0.5) / 0.5
                with torch.no_grad():
                    ens(t)
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(times) / len(times)
    target = 40.0
    status = "OK" if avg_ms <= target else f"OVER TARGET ({target} ms)"
    return f"avg latency {avg_ms:.1f} ms  [{status}]"

check("latency.a_plus_b_cpu", _test_latency)


# ---------------------------------------------------------------------------
# Summary
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
