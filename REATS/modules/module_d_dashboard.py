"""
MODULE D — Operator Dashboard
Real-time Streamlit UI for the full REATS pipeline.
Target: ≤ 40ms per frame end-to-end (Module A → B → C).
Run: cd REATS && streamlit run modules/module_d_dashboard.py
"""

import time
import cv2
import numpy as np
import torch
import streamlit as st
from pathlib import Path

CLASSES      = ["F16", "LYNX", "MiG19", "MiG21", "PKG", "PTG"]
THREAT_COLOR = {
    "F16":   (255, 0,   0),
    "MiG19": (255, 0,   0),
    "MiG21": (255, 0,   0),
    "LYNX":  (255, 165, 0),
    "PKG":   (255, 0,   0),
    "PTG":   (255, 0,   0),
}


@st.cache_resource
def load_pipeline(det_weights: str, cls_weights: str):
    """Load and cache detector + classifier once per session."""
    from modules.module_a_detector    import IRDetector
    from modules.module_b_classifier  import build_convnext, EnsembleClassifier

    detector = IRDetector(weights=det_weights)
    model    = build_convnext(num_classes=6, pretrained=False)
    if Path(cls_weights).exists():
        model.load_state_dict(torch.load(cls_weights, map_location="cpu"))
    model.eval()
    return detector, EnsembleClassifier([model])


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


def run_pipeline(frame: np.ndarray, detector, classifier) -> dict:
    t0 = time.perf_counter()
    detections = detector.detect(frame)
    tf = _get_transform()
    results = []
    for det in detections:
        roi = detector.crop_roi(frame, det["bbox"])
        if roi.size == 0:
            continue
        roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR) if roi.ndim == 2 else roi
        tensor  = tf(roi_bgr).unsqueeze(0)
        with torch.no_grad():
            probs = classifier(tensor)[0]
        pred_cls = probs.argmax().item()
        results.append({
            "bbox":       det["bbox"],
            "class":      CLASSES[pred_cls],
            "confidence": round(float(probs[pred_cls]), 4),
            "probs":      {CLASSES[i]: round(float(p), 4) for i, p in enumerate(probs)},
        })
    return {"detections": results, "latency_ms": (time.perf_counter() - t0) * 1000}


def draw_overlays(frame: np.ndarray, output: dict) -> np.ndarray:
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
    for det in output["detections"]:
        x1, y1, x2, y2 = det["bbox"]
        c = THREAT_COLOR.get(det["class"], (0, 255, 0))
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
        cv2.putText(
            vis, f"{det['class']} {det['confidence']:.2f}",
            (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2,
        )
    lat = output["latency_ms"]
    cv2.putText(
        vis, f"{lat:.1f}ms", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 0) if lat <= 40 else (0, 0, 255), 2,
    )
    return vis


def main():
    st.set_page_config(page_title="REATS Dashboard", layout="wide", page_icon="🎯")
    st.title("REATS — Real-time Explainable ATR")
    st.caption(
        "Pipeline: IR Frame → Module A (YOLOv8) → Module B (ConvNeXt Ensemble) "
        "→ Module C (XAI) → Display"
    )

    with st.sidebar:
        st.header("Model Paths")
        det_weights = st.text_input("Detector weights",    "checkpoints/yolov8_ir.pt")
        cls_weights = st.text_input("Classifier weights",  "checkpoints/convnext_best.pth")
        load_btn    = st.button("Load models", type="primary")

        st.divider()
        st.subheader("Target Classes")
        for cls in CLASSES:
            st.text(f"  {cls}")

        st.divider()
        st.subheader("Metric Targets")
        st.markdown(
            "- Accuracy ≥ **92%**\n"
            "- mAP@0.5 ≥ **75%**\n"
            "- Latency ≤ **40 ms**\n"
            "- ECE ≤ **0.05**"
        )

    pipeline = None
    if load_btn:
        with st.spinner("Loading models…"):
            pipeline = load_pipeline(det_weights, cls_weights)
        st.sidebar.success("Models ready.")
        st.session_state["pipeline"] = pipeline
    elif "pipeline" in st.session_state:
        pipeline = st.session_state["pipeline"]

    col_in, col_out = st.columns(2)

    with col_in:
        st.subheader("Input IR Frame")
        uploaded = st.file_uploader("Upload frame", type=["png", "jpg", "jpeg", "bmp"])

    with col_out:
        st.subheader("ATR Output")
        if uploaded is not None and pipeline is not None:
            detector, classifier = pipeline
            raw   = np.frombuffer(uploaded.read(), np.uint8)
            frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            out   = run_pipeline(frame, detector, classifier)
            vis   = draw_overlays(frame, out)
            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)

            lat = out["latency_ms"]
            st.metric(
                "Latency", f"{lat:.1f} ms",
                delta="OK" if lat <= 40 else f"SLOW (+{lat - 40:.1f}ms)",
            )
            if out["detections"]:
                for i, det in enumerate(out["detections"]):
                    with st.expander(f"Target {i + 1}: {det['class']} ({det['confidence']:.0%})"):
                        st.json(det)
            else:
                st.info("No targets detected in this frame.")
        elif pipeline is None:
            st.info("Load models from the sidebar to begin.")
        else:
            st.info("Upload an IR frame to run detection + classification.")


if __name__ == "__main__":
    main()
