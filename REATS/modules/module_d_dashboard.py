"""MODULE D — Operator Dashboard (production-grade Streamlit UI for REATS)."""

import io
import time
import zipfile
import csv

import cv2
import numpy as np
import torch
import streamlit as st
from pathlib import Path
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASSES = ["F16", "LYNX", "MiG19", "MiG21", "PKG", "PTG"]

THREAT_COLOR_BGR = {
    "F16":   (0,   0,   255),
    "MiG19": (0,   0,   255),
    "MiG21": (0,   0,   255),
    "LYNX":  (0,   165, 255),
    "PKG":   (0,   0,   255),
    "PTG":   (0,   0,   255),
}

RED_THREATS    = {"F16", "MiG19", "MiG21", "PKG", "PTG"}
ORANGE_THREATS = {"LYNX"}

METRIC_TARGETS = {
    "Accuracy": ("≥ 92%", 0.92),
    "mAP@0.5":  ("≥ 75%", 0.75),
    "Latency":  ("≤ 40 ms", 40.0),
    "ECE":      ("≤ 0.05", 0.05),
}

MAX_VIDEO_FRAMES = 30

# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_pipeline(det_weights: str, cls_weights_csv: str):
    """Load detector + classifier ensemble; returns (detector, classifier, param_counts)."""
    from modules.module_a_detector   import IRDetector
    from modules.module_b_classifier import build_convnext, EnsembleClassifier

    detector = IRDetector(weights=det_weights)
    det_params = sum(p.numel() for p in detector.model.parameters()) if hasattr(detector, "model") else 0

    models = []
    total_cls_params = 0
    for w in [p.strip() for p in cls_weights_csv.split(",") if p.strip()]:
        m = build_convnext(num_classes=len(CLASSES), pretrained=False)
        if Path(w).exists():
            m.load_state_dict(torch.load(w, map_location="cpu"))
        m.eval()
        total_cls_params += sum(p.numel() for p in m.parameters())
        models.append(m)

    if not models:
        m = build_convnext(num_classes=len(CLASSES), pretrained=False)
        m.eval()
        total_cls_params = sum(p.numel() for p in m.parameters())
        models.append(m)

    classifier = EnsembleClassifier(models)
    return detector, classifier, {"detector": det_params, "classifier": total_cls_params}


# ---------------------------------------------------------------------------
# Image transform (lazy singleton)
# ---------------------------------------------------------------------------

_TRANSFORM = None


def _get_transform():
    global _TRANSFORM
    if _TRANSFORM is None:
        from torchvision import transforms
        _TRANSFORM = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
    return _TRANSFORM


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def run_pipeline(
    frame: np.ndarray,
    detector,
    classifier,
    conf_thresh: float = 0.25,
    iou_thresh: float  = 0.45,
    run_xai: bool      = False,
    mc_dropout: bool   = False,
    mc_passes: int     = 10,
) -> dict:
    """Run full pipeline on one frame; returns detections + latency."""
    t0 = time.perf_counter()
    detections = detector.detect(frame, conf=conf_thresh, iou=iou_thresh)
    tf = _get_transform()
    results = []

    for det in detections:
        roi = detector.crop_roi(frame, det["bbox"])
        if roi.size == 0:
            continue
        roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR) if roi.ndim == 2 else roi
        tensor  = tf(roi_bgr).unsqueeze(0)

        # Standard inference
        with torch.no_grad():
            probs = classifier(tensor)[0]

        pred_idx  = int(probs.argmax())
        pred_cls  = CLASSES[pred_idx]
        conf      = float(probs[pred_idx])

        # MC Dropout uncertainty
        uncertainty = None
        if mc_dropout:
            classifier.train()  # enable dropout
            mc_probs = []
            for _ in range(mc_passes):
                with torch.no_grad():
                    mc_probs.append(classifier(tensor)[0].numpy())
            classifier.eval()
            mc_arr    = np.stack(mc_probs)               # (passes, C)
            mean_p    = mc_arr.mean(axis=0)
            entropy   = float(-np.sum(mean_p * np.log(mean_p + 1e-9)))
            uncertainty = {"entropy": round(entropy, 4), "std": mc_arr.std(axis=0).tolist()}

        # Grad-CAM (simple single-model approximation)
        heatmap = None
        if run_xai:
            heatmap = _grad_cam(classifier, tensor, pred_idx, roi_bgr.shape[:2])

        results.append({
            "bbox":        det["bbox"],
            "class":       pred_cls,
            "confidence":  round(conf, 4),
            "probs":       {CLASSES[i]: round(float(p), 4) for i, p in enumerate(probs)},
            "uncertainty": uncertainty,
            "heatmap":     heatmap,
        })

    return {"detections": results, "latency_ms": (time.perf_counter() - t0) * 1000}


def _grad_cam(classifier, tensor: torch.Tensor, target_idx: int, out_shape: tuple):
    """Minimal Grad-CAM approximation; returns RGB heatmap array or None."""
    try:
        model = classifier.models[0] if hasattr(classifier, "models") else classifier
        grads, acts = [], []

        def _fwd_hook(m, inp, out):
            acts.append(out.detach())

        def _bwd_hook(m, gin, gout):
            grads.append(gout[0].detach())

        # Hook last conv-like layer
        target_layer = None
        for layer in model.modules():
            if isinstance(layer, torch.nn.Conv2d):
                target_layer = layer
        if target_layer is None:
            return None

        fh = target_layer.register_forward_hook(_fwd_hook)
        bh = target_layer.register_backward_hook(_bwd_hook)

        tensor_req = tensor.clone().requires_grad_(True)
        logits = model(tensor_req)
        model.zero_grad()
        logits[0, target_idx].backward()

        fh.remove()
        bh.remove()

        if not acts or not grads:
            return None

        weights = grads[0].mean(dim=(2, 3), keepdim=True)
        cam     = (weights * acts[0]).sum(dim=1).squeeze().numpy()
        cam     = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        cam_resized = cv2.resize(cam, (out_shape[1], out_shape[0]))
        heatmap     = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return heatmap  # BGR uint8
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------

def draw_overlays(frame: np.ndarray, output: dict, opacity: float = 0.5) -> np.ndarray:
    """Draw bounding boxes + labels on frame; returns BGR uint8."""
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
    for det in output["detections"]:
        x1, y1, x2, y2 = det["bbox"]
        c = THREAT_COLOR_BGR.get(det["class"], (0, 255, 0))
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, 3)
        label = f"{det['class']} {det['confidence']:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(y1 - 8, th + 4)
        cv2.rectangle(vis, (x1, ty - th - 4), (x1 + tw + 4, ty), c, -1)
        cv2.putText(vis, label, (x1 + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    lat = output["latency_ms"]
    lat_color = (0, 200, 0) if lat <= 40 else (0, 0, 255)
    cv2.putText(vis, f"{lat:.1f} ms", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, lat_color, 2)
    return vis


def _blend_heatmap(roi_bgr: np.ndarray, heatmap: np.ndarray, opacity: float) -> np.ndarray:
    """Alpha-blend Grad-CAM heatmap onto ROI."""
    h, w = roi_bgr.shape[:2]
    hm = cv2.resize(heatmap, (w, h))
    return cv2.addWeighted(roi_bgr, 1 - opacity, hm, opacity, 0)


# ---------------------------------------------------------------------------
# Helper: frame bytes → RGB array
# ---------------------------------------------------------------------------

def _decode_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return bgr


def _bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar():
    """Render sidebar; returns (det_weights, cls_weights_csv, thresholds, flags)."""
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        det_weights  = st.text_input("Detector weights path", "checkpoints/yolov8_ir.pt")
        cls_weights  = st.text_input(
            "Classifier weights (comma-separated)",
            "checkpoints/convnext_best.pth",
            help="Multiple paths for ensemble, e.g. a.pth, b.pth",
        )
        load_btn = st.button("Load Models", type="primary", use_container_width=True)

        if load_btn:
            with st.spinner("Loading models…"):
                try:
                    pipeline = load_pipeline(det_weights, cls_weights)
                    st.session_state["pipeline"]    = pipeline
                    st.session_state["det_weights"] = det_weights
                    st.session_state["cls_weights"] = cls_weights
                except Exception as e:
                    st.error(f"Load failed: {e}")
                    st.session_state.pop("pipeline", None)

        if "pipeline" in st.session_state:
            _, _, pcounts = st.session_state["pipeline"]
            st.success("✓ Models ready")
            st.caption(
                f"Det params: {pcounts['detector']:,}   "
                f"Cls params: {pcounts['classifier']:,}"
            )
        else:
            st.info("No models loaded. Click **Load Models** to begin.")

        st.divider()
        st.subheader("Inference Settings")
        conf_thresh = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.01)
        iou_thresh  = st.slider("IoU threshold",         0.05, 0.95, 0.45, 0.01)

        st.divider()
        st.subheader("XAI / Uncertainty")
        xai_enabled = st.checkbox("Enable XAI (Grad-CAM)", value=False)
        hm_opacity  = 0.5
        if xai_enabled:
            hm_opacity = st.slider("Heatmap opacity", 0.3, 0.9, 0.5, 0.05)
        mc_dropout = st.checkbox("MC Dropout uncertainty", value=False)

        st.divider()
        st.subheader("📋 Metric Targets")
        for name, (target, _) in METRIC_TARGETS.items():
            st.markdown(f"- **{name}**: {target}")

    return det_weights, cls_weights, conf_thresh, iou_thresh, xai_enabled, hm_opacity, mc_dropout


# ---------------------------------------------------------------------------
# Tab 1 — Live Analysis
# ---------------------------------------------------------------------------

def _tab_live(conf_thresh, iou_thresh, xai_enabled, hm_opacity, mc_dropout):
    st.subheader("Upload Image or Video")
    media_type = st.radio("Input type", ["Image", "Video"], horizontal=True)

    pipeline = st.session_state.get("pipeline")

    if media_type == "Image":
        uploaded = st.file_uploader(
            "Upload IR image", type=["png", "jpg", "jpeg", "bmp", "tiff"], key="img_upload"
        )
        if uploaded is None:
            if pipeline is None:
                st.info("Load models from the sidebar, then upload an image.")
            else:
                st.info("Upload an IR image to run detection + classification.")
            return

        file_bytes = uploaded.read()
        frame = _decode_image(file_bytes)

        if pipeline is None:
            st.warning("Models not loaded — showing original image only.")
            st.image(_bgr_to_rgb(frame), caption="Original (no inference)", use_container_width=True)
            return

        detector, classifier, _ = pipeline
        with st.spinner("Running pipeline…"):
            output = run_pipeline(
                frame, detector, classifier,
                conf_thresh=conf_thresh, iou_thresh=iou_thresh,
                run_xai=xai_enabled, mc_dropout=mc_dropout,
            )

        vis = draw_overlays(frame, output, hm_opacity)
        lat = output["latency_ms"]

        # Columns: original | annotated [| grad-cam if XAI]
        n_cols = 3 if (xai_enabled and any(d["heatmap"] is not None for d in output["detections"])) else 2
        cols = st.columns(n_cols)
        cols[0].image(_bgr_to_rgb(frame), caption="Original", use_container_width=True)
        cols[1].image(_bgr_to_rgb(vis),   caption="Annotated", use_container_width=True)
        if n_cols == 3:
            # Stitch all ROI heatmaps side by side
            heatmap_imgs = [d["heatmap"] for d in output["detections"] if d["heatmap"] is not None]
            if heatmap_imgs:
                stitch = np.hstack([cv2.resize(h, (128, 128)) for h in heatmap_imgs])
                cols[2].image(_bgr_to_rgb(stitch), caption="Grad-CAM", use_container_width=True)

        # Latency metric
        lat_color = "normal" if lat <= 40 else "inverse"
        st.metric("⏱ Latency", f"{lat:.1f} ms", delta="OK" if lat <= 40 else f"+{lat-40:.1f} ms over target", delta_color=lat_color)

        # Threat level
        classes_found = {d["class"] for d in output["detections"]}
        if classes_found & RED_THREATS:
            st.error(f"🔴 THREAT LEVEL: RED — {', '.join(classes_found & RED_THREATS)} detected")
        elif classes_found & ORANGE_THREATS:
            st.warning(f"🟠 THREAT LEVEL: ORANGE — {', '.join(classes_found & ORANGE_THREATS)} detected")
        elif output["detections"]:
            st.success("🟢 THREAT LEVEL: LOW — no high-threat classes")
        else:
            st.info("No targets detected in this frame.")

        # Per-detection cards
        for i, det in enumerate(output["detections"]):
            with st.expander(f"Target {i+1}: {det['class']} ({det['confidence']:.0%})", expanded=True):
                c1, c2 = st.columns(2)
                c1.metric("Class", det["class"])
                c1.metric("Confidence", f"{det['confidence']:.1%}")
                c2.bar_chart(det["probs"])

                if mc_dropout and det["uncertainty"] is not None:
                    ent = det["uncertainty"]["entropy"]
                    max_ent = np.log(len(CLASSES))
                    norm_ent = min(ent / max_ent, 1.0)
                    st.progress(norm_ent, text=f"Uncertainty entropy: {ent:.3f}")
                    if ent > 0.5:
                        st.error("⚠️ HIGH UNCERTAINTY — interpret result with caution")

    else:  # Video
        uploaded = st.file_uploader(
            "Upload IR video", type=["mp4", "avi", "mov"], key="vid_upload"
        )
        if uploaded is None:
            st.info("Upload a video file (mp4/avi/mov).")
            return
        if pipeline is None:
            st.warning("Models not loaded. Load models from sidebar first.")
            return

        detector, classifier, _ = pipeline
        tmp_path = Path("/tmp/_reats_upload_video")
        tmp_path.write_bytes(uploaded.read())

        cap = cv2.VideoCapture(str(tmp_path))
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = 0
        annotated_frames = []
        latencies = []

        progress = st.progress(0, text="Processing video…")
        fps_placeholder = st.empty()

        t_start = time.perf_counter()
        while cap.isOpened() and frame_count < MAX_VIDEO_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            output = run_pipeline(frame, detector, classifier, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
            vis    = draw_overlays(frame, output)
            annotated_frames.append(vis)
            latencies.append(output["latency_ms"])
            frame_count += 1
            progress.progress(frame_count / MAX_VIDEO_FRAMES, text=f"Frame {frame_count}/{MAX_VIDEO_FRAMES}")
            elapsed = time.perf_counter() - t_start
            fps_placeholder.metric("Processing FPS", f"{frame_count / elapsed:.1f}")

        cap.release()
        tmp_path.unlink(missing_ok=True)
        progress.empty()

        st.success(f"Processed {frame_count} frames — avg latency {np.mean(latencies):.1f} ms")
        if annotated_frames:
            st.image(_bgr_to_rgb(annotated_frames[-1]), caption="Last annotated frame", use_container_width=True)

            # ZIP download
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for idx, f in enumerate(annotated_frames):
                    ok, buf = cv2.imencode(".jpg", f)
                    if ok:
                        zf.writestr(f"frame_{idx:04d}.jpg", buf.tobytes())
            zip_buf.seek(0)
            st.download_button("Download annotated frames (ZIP)", zip_buf, "reats_frames.zip", "application/zip")


# ---------------------------------------------------------------------------
# Tab 2 — Batch Processing
# ---------------------------------------------------------------------------

def _tab_batch(conf_thresh, iou_thresh):
    st.subheader("Batch Image Processing")
    pipeline = st.session_state.get("pipeline")
    if pipeline is None:
        st.warning("Load models from the sidebar first.")
        return

    uploaded_files = st.file_uploader(
        "Upload multiple images",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        accept_multiple_files=True,
        key="batch_upload",
    )
    if not uploaded_files:
        st.info("Upload one or more images to process as a batch.")
        return

    detector, classifier, _ = pipeline
    rows = []
    with st.spinner(f"Processing {len(uploaded_files)} images…"):
        for uf in uploaded_files:
            frame = _decode_image(uf.read())
            out   = run_pipeline(frame, detector, classifier, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
            if out["detections"]:
                best = max(out["detections"], key=lambda d: d["confidence"])
                rows.append({
                    "filename":       uf.name,
                    "detected_class": best["class"],
                    "confidence":     round(best["confidence"], 4),
                    "latency_ms":     round(out["latency_ms"], 2),
                    "n_detections":   len(out["detections"]),
                })
            else:
                rows.append({
                    "filename":       uf.name,
                    "detected_class": "none",
                    "confidence":     0.0,
                    "latency_ms":     round(out["latency_ms"], 2),
                    "n_detections":   0,
                })

    if not rows:
        st.info("No results.")
        return

    st.dataframe(rows, use_container_width=True)

    # Summary stats
    latencies = [r["latency_ms"] for r in rows]
    st.metric("Total frames",   len(rows))
    st.metric("Avg latency",    f"{np.mean(latencies):.1f} ms")

    # Class distribution
    class_counts: dict = {}
    for r in rows:
        class_counts[r["detected_class"]] = class_counts.get(r["detected_class"], 0) + 1
    st.subheader("Class distribution")
    st.bar_chart(class_counts)

    # CSV download
    csv_buf = io.StringIO()
    writer  = csv.DictWriter(csv_buf, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    st.download_button("Download results CSV", csv_buf.getvalue(), "batch_results.csv", "text/csv")


# ---------------------------------------------------------------------------
# Tab 3 — Calibration
# ---------------------------------------------------------------------------

def _tab_calibration():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st.subheader("Model Calibration")
    pipeline = st.session_state.get("pipeline")

    st.markdown(
        "Upload a **ZIP** containing class sub-folders of images "
        "(e.g. `test_set.zip/F16/*.jpg`, `test_set.zip/MiG21/*.png`, …)."
    )
    zip_file = st.file_uploader("Upload test-set ZIP", type=["zip"], key="cal_upload")

    T_scale = st.slider("Temperature scaling T", 0.5, 3.0, 1.0, 0.1)

    if zip_file is None:
        st.info("Upload a test-set ZIP to compute calibration metrics.")
        # Show placeholder reliability diagram
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(np.linspace(0.05, 0.95, 10), np.linspace(0.05, 0.95, 10), width=0.09, alpha=0.6, label="Perfect")
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
        ax.set_title("Reliability Diagram (placeholder)")
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        return

    if pipeline is None:
        st.warning("Load models from the sidebar first.")
        return

    detector, classifier, _ = pipeline

    with st.spinner("Extracting ZIP and running inference…"):
        zf      = zipfile.ZipFile(io.BytesIO(zip_file.read()))
        names   = zf.namelist()
        all_confs: list = []
        all_corrects: list = []
        per_class: dict = {}

        for name in names:
            parts = Path(name).parts
            if len(parts) < 2:
                continue
            label_name = parts[-2]
            if label_name not in CLASSES:
                continue
            true_idx = CLASSES.index(label_name)

            data  = zf.read(name)
            arr   = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            from torchvision import transforms as T
            tf = _get_transform()
            tensor = tf(frame).unsqueeze(0)
            with torch.no_grad():
                logits = classifier(tensor)[0]
                scaled = logits / T_scale
                probs  = torch.softmax(scaled, dim=0)

            pred_idx  = int(probs.argmax())
            conf      = float(probs[pred_idx])
            correct   = int(pred_idx == true_idx)
            all_confs.append(conf)
            all_corrects.append(correct)

            per_class.setdefault(label_name, {"correct": 0, "total": 0})
            per_class[label_name]["correct"] += correct
            per_class[label_name]["total"]   += 1

    if not all_confs:
        st.warning("No valid images found in ZIP (check folder structure).")
        return

    all_confs    = np.array(all_confs)
    all_corrects = np.array(all_corrects)

    # ECE computation
    n_bins   = 10
    bins     = np.linspace(0, 1, n_bins + 1)
    ece      = 0.0
    bin_accs, bin_confs = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (all_confs > lo) & (all_confs <= hi)
        if mask.sum() == 0:
            bin_accs.append(0.0)
            bin_confs.append((lo + hi) / 2)
            continue
        b_acc  = all_corrects[mask].mean()
        b_conf = all_confs[mask].mean()
        ece   += (mask.sum() / len(all_confs)) * abs(b_acc - b_conf)
        bin_accs.append(float(b_acc))
        bin_confs.append(float(b_conf))

    st.metric("ECE", f"{ece:.4f}", delta="OK" if ece <= 0.05 else f"+{ece-0.05:.4f} over target", delta_color="normal" if ece <= 0.05 else "inverse")
    st.metric("Overall accuracy", f"{all_corrects.mean():.1%}")

    # Reliability diagram
    fig, ax = plt.subplots(figsize=(5, 4))
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(n_bins)]
    ax.bar(bin_centers, bin_accs, width=0.08, alpha=0.7, label="Model", color="steelblue")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability Diagram (T={T_scale})")
    ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Per-class accuracy table
    st.subheader("Per-class Accuracy")
    pc_rows = [
        {"class": cls, "correct": v["correct"], "total": v["total"],
         "accuracy": f"{v['correct']/v['total']:.1%}" if v["total"] else "N/A"}
        for cls, v in sorted(per_class.items())
    ]
    st.dataframe(pc_rows, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4 — About
# ---------------------------------------------------------------------------

def _tab_about():
    st.subheader("REATS — Real-time Explainable Automatic Target Recognition System")
    st.markdown(
        """
**Architecture overview**

```
IR Frame
   │
   ▼
Module A — YOLOv8-based IR Detector
   │  bounding boxes + crops
   ▼
Module B — ConvNeXt Ensemble Classifier  (6 classes)
   │  class probabilities
   ▼
Module C — XAI (Grad-CAM)  +  MC Dropout Uncertainty
   │
   ▼
Module D — Operator Dashboard (this UI)
```

**Target classes:** F16, LYNX, MiG19, MiG21, PKG, PTG

**Threat levels**
- 🔴 RED — F16, MiG19, MiG21, PKG, PTG
- 🟠 ORANGE — LYNX

**Citation**
```
@misc{reats2024,
  title  = {REATS: Real-time Explainable Automatic Target Recognition},
  year   = {2024},
  note   = {GitHub placeholder}
}
```

**GitHub:** [https://github.com/your-org/REATS](https://github.com/your-org/REATS)
        """
    )

    st.divider()
    st.subheader("Environment")
    import platform
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_info = {
        "Python":      platform.python_version(),
        "PyTorch":     torch.__version__,
        "CUDA available": str(torch.cuda.is_available()),
        "Device":      device,
    }
    if torch.cuda.is_available():
        env_info["CUDA device"] = torch.cuda.get_device_name(0)
    for k, v in env_info.items():
        st.text(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="REATS Dashboard", layout="wide", page_icon="🎯")
    st.title("🎯 REATS — Real-time Explainable ATR")
    st.caption(
        "IR Frame → Module A (YOLOv8 detector) → Module B (ConvNeXt ensemble) "
        "→ Module C (XAI/uncertainty) → Module D (this dashboard)"
    )

    det_w, cls_w, conf_thresh, iou_thresh, xai_enabled, hm_opacity, mc_dropout = _render_sidebar()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎯 Live Analysis", "📦 Batch Processing", "📊 Calibration", "ℹ️ About"]
    )

    with tab1:
        _tab_live(conf_thresh, iou_thresh, xai_enabled, hm_opacity, mc_dropout)

    with tab2:
        _tab_batch(conf_thresh, iou_thresh)

    with tab3:
        _tab_calibration()

    with tab4:
        _tab_about()


if __name__ == "__main__":
    main()
