#!/usr/bin/env python3
"""
REATS performance metrics report.

Loads saved checkpoints and evaluates all metrics against paper targets:
  - Classification accuracy  ≥ 92%
  - ECE (calibration)        ≤ 0.05
  - mAP@0.5 (detection)      ≥ 75%
  - End-to-end latency        ≤ 40 ms/frame
  - Faithfulness AUC          ≥ 0.80
  - FPS                       ≥ 20

Usage (from REATS/):
    python metrics_report.py --cls-weights checkpoints/ensemble_0.pth,checkpoints/ensemble_1.pth
    python metrics_report.py --cls-weights checkpoints/best.pth --det-weights checkpoints/detector.pt
    python metrics_report.py --quick    # skip slow tests (faithfulness, mAP)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_root = Path(__file__).parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import CLASSES, NUM_CLASSES

TARGETS = {
    "accuracy":        ("≥ 92%",    0.92,  "ge"),
    "ece":             ("≤ 0.05",   0.05,  "le"),
    "map@0.5":         ("≥ 75%",    0.75,  "ge"),
    "latency_ms":      ("≤ 40 ms",  40.0,  "le"),
    "faithfulness_auc":("≥ 0.80",   0.80,  "ge"),
    "fps":             ("≥ 20",     20.0,  "ge"),
}


def _load_classifier(weights_csv: str):
    from modules.module_b_classifier import build_convnext, EnsembleClassifier
    paths  = [p.strip() for p in weights_csv.split(",") if p.strip()]
    models = []
    for w in paths:
        m = build_convnext(num_classes=NUM_CLASSES, pretrained=False)
        if Path(w).exists():
            ckpt  = torch.load(w, map_location="cpu")
            state = ckpt.get("model_state", ckpt)
            m.load_state_dict(state)
        else:
            print(f"  [warn] weights not found: {w} — using random init")
        m.eval()
        models.append(m)
    if len(models) == 1:
        return models[0], False
    return EnsembleClassifier(models), True


def _make_val_loader(data_root: str, batch_size: int = 32) -> DataLoader | None:
    val_dir = Path(data_root) / "val"
    if not val_dir.exists():
        return None
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    ds = datasets.ImageFolder(str(val_dir), transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)


def measure_accuracy_ece(model, loader, device, is_probs: bool) -> tuple[float, float]:
    from modules.module_b_classifier import compute_ece
    from torch.nn import CrossEntropyLoss
    from modules.module_b_classifier import evaluate
    model.eval()
    if hasattr(model, "models"):
        # EnsembleClassifier: evaluate needs logits, not probs
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs   = imgs.to(device)
                labels = labels.to(device)
                probs  = model(imgs)      # already softmaxed by EnsembleClassifier
                correct += (probs.argmax(1) == labels).sum().item()
                total   += imgs.size(0)
        acc = correct / max(total, 1)
    else:
        _, acc = evaluate(model, loader, CrossEntropyLoss(), device)

    ece = compute_ece(model, loader, device, is_probs=is_probs)
    return acc, ece


def measure_faithfulness(model, loader, device, n_samples: int = 20) -> float:
    """Faithfulness deletion AUC on a small sample."""
    try:
        from modules.module_c_xai import GradCAMExplainer
        m = model.models[0] if hasattr(model, "models") else model
        explainer = GradCAMExplainer(m)
        aucs = []
        for imgs, labels in loader:
            for i in range(min(n_samples, imgs.size(0))):
                img   = imgs[i:i+1].to(device)
                label = int(labels[i])
                try:
                    heatmap = explainer.explain(img, label)
                    # Deletion AUC: progressively mask top-k pixels, measure accuracy drop
                    flat = heatmap.flatten()
                    order = np.argsort(-flat)
                    accs_curve = []
                    masked = img.clone()
                    h, w  = img.shape[-2:]
                    step  = max(1, len(order) // 10)
                    with torch.no_grad():
                        for k in range(0, len(order), step):
                            if k > 0:
                                idxs = order[:k]
                                ys   = idxs // w
                                xs   = idxs  % w
                                masked[0, :, ys, xs] = 0.0
                            out  = m(masked)
                            conf = torch.softmax(out, -1)[0, label].item()
                            accs_curve.append(conf)
                    # AUC (higher = more faithful, features matter)
                    baseline = accs_curve[-1]
                    initial  = accs_curve[0]
                    if initial - baseline > 1e-4:
                        auc = np.trapz(accs_curve) / len(accs_curve)
                        aucs.append(auc)
                except Exception:
                    pass
            if len(aucs) >= n_samples:
                break
        return float(np.mean(aucs)) if aucs else 0.0
    except Exception as e:
        print(f"  [warn] faithfulness failed: {e}")
        return float("nan")


def measure_latency_fps(detector, model, device, n_reps: int = 20) -> tuple[float, float]:
    import numpy as np
    import cv2
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[160:320, 200:440] = 180

    model.eval()
    times = []
    for _ in range(n_reps):
        t0   = time.perf_counter()
        dets = detector.detect(frame)
        if dets:
            roi = detector.crop_roi(frame, dets[0]["bbox"])
            if roi.size > 0:
                from PIL import Image
                r = np.array(Image.fromarray(roi).resize((224, 224))).astype(np.float32) / 255.0
                t = torch.from_numpy(r).permute(2, 0, 1).unsqueeze(0).to(device)
                t = (t - 0.5) / 0.5
                with torch.no_grad():
                    model(t)
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = float(np.mean(times))
    fps    = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    return avg_ms, fps


def _fmt(metric: str, value: float) -> str:
    target_str, target_val, direction = TARGETS.get(metric, ("—", None, None))
    if target_val is None or np.isnan(value):
        return f"{value:.4f}  (target: {target_str})  ?"
    ok = (value >= target_val) if direction == "ge" else (value <= target_val)
    icon = "✅" if ok else "❌"
    if metric in ("accuracy",):
        return f"{value*100:.2f}%  (target: {target_str})  {icon}"
    if metric == "ece":
        return f"{value:.4f}  (target: {target_str})  {icon}"
    if metric == "latency_ms":
        return f"{value:.1f} ms  (target: {target_str})  {icon}"
    if metric == "fps":
        return f"{value:.1f}  (target: {target_str})  {icon}"
    if metric == "faithfulness_auc":
        return f"{value:.4f}  (target: {target_str})  {icon}"
    if metric == "map@0.5":
        return f"{value*100:.2f}%  (target: {target_str})  {icon}"
    return f"{value:.4f}  (target: {target_str})  {icon}"


def main():
    parser = argparse.ArgumentParser(description="REATS performance metrics report")
    parser.add_argument("--cls-weights", default="",
                        help="Comma-separated .pth paths (single or ensemble)")
    parser.add_argument("--det-weights", default="",
                        help="YOLOv4 detector .pt checkpoint")
    parser.add_argument("--data", default="data/",
                        help="Data root (expects data/val/ subfolder)")
    parser.add_argument("--device", default="")
    parser.add_argument("--quick", action="store_true",
                        help="Skip faithfulness and mAP (they take minutes)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nREATS Metrics Report  (device: {device})")
    print("=" * 60)

    # ── Classifier ────────────────────────────────────────────────
    if args.cls_weights:
        print("\n[Loading classifier]")
        model, is_ensemble = _load_classifier(args.cls_weights)
        model.to(device)
        val_loader = _make_val_loader(args.data)

        if val_loader:
            print("[Accuracy + ECE]")
            acc, ece = measure_accuracy_ece(model, val_loader, device, is_probs=is_ensemble)
            print(f"  Accuracy        : {_fmt('accuracy', acc)}")
            print(f"  ECE             : {_fmt('ece', ece)}")

            if not args.quick:
                print("[Faithfulness AUC]")
                faith = measure_faithfulness(model, val_loader, device)
                print(f"  Faithfulness AUC: {_fmt('faithfulness_auc', faith)}")
        else:
            print("  [skip] No val/ directory found — skipping accuracy/ECE")
    else:
        print("[skip] --cls-weights not provided")

    # ── Detector ──────────────────────────────────────────────────
    det_weights = args.det_weights or "checkpoints/detector_bootstrap.pt"
    print(f"\n[Loading detector: {det_weights}]")
    try:
        from modules.module_a_detector import IRDetector, compute_map
        detector = IRDetector(weights=det_weights if Path(det_weights).exists() else None)

        if not args.quick and args.det_weights:
            val_dir = Path(args.data) / "val"
            if val_dir.exists():
                print("[mAP@0.5]  (this may take a few minutes)")
                mAP = compute_map(detector.model, args.data, device=torch.device(device))
                print(f"  mAP@0.5         : {_fmt('map@0.5', mAP)}")
            else:
                print("  [skip] No val/ directory for mAP")
        else:
            print("  [skip] mAP (use --det-weights and omit --quick to enable)")

        print("[Latency + FPS]")
        if args.cls_weights:
            lat, fps = measure_latency_fps(detector, model, device)
        else:
            from modules.module_b_classifier import build_convnext
            dummy_model = build_convnext(NUM_CLASSES, pretrained=False).to(device)
            lat, fps = measure_latency_fps(detector, dummy_model, device)
        print(f"  Latency (A+B)   : {_fmt('latency_ms', lat)}")
        print(f"  FPS estimate    : {_fmt('fps', fps)}")

    except Exception as e:
        print(f"  [error] Detector failed: {e}")

    print("\n" + "=" * 60)
    print("  NOTE: Latency ≤ 40 ms and FPS ≥ 20 require GPU.")
    print("  CPU numbers are for architecture validation only.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
