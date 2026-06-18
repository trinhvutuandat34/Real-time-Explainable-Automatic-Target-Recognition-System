"""
MODULE A — IR Detector
YOLOv8-based object detector for full-scene IR frames.
Target: mAP@0.5 ≥ 75%, FPS ≥ 20, Latency ≤ 40ms/frame
"""

import torch
import numpy as np
from pathlib import Path

CLASSES = ["F16", "LYNX", "MiG19", "MiG21", "PKG", "PTG"]


class IRDetector:
    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.25, iou: float = 0.45):
        from ultralytics import YOLO
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model  = YOLO(weights)
        self.conf   = conf
        self.iou    = iou

    def detect(self, frame: np.ndarray) -> list:
        """Detect targets in a full IR frame.

        Args:
            frame: (H, W, C) uint8 numpy array (grayscale-to-3ch or BGR)
        Returns:
            list of {bbox: [x1,y1,x2,y2], conf: float, class_id: int}
        """
        results = self.model.predict(
            frame, conf=self.conf, iou=self.iou, verbose=False, device=self.device
        )[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            detections.append({
                "bbox":     [x1, y1, x2, y2],
                "conf":     float(box.conf[0]),
                "class_id": int(box.cls[0]),
            })
        return detections

    def crop_roi(self, frame: np.ndarray, bbox: list, pad: int = 10) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        return frame[
            max(0, y1 - pad): min(h, y2 + pad),
            max(0, x1 - pad): min(w, x2 + pad),
        ]

    def train(
        self,
        data_yaml: str,
        epochs:    int = 100,
        imgsz:     int = 640,
        batch:     int = 16,
    ):
        return self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self.device,
            project="runs/detect",
            name="yolov8_ir",
        )


def make_yolo_yaml(data_root: str = "data/", out: str = "data/yolo.yaml") -> str:
    """Write a YOLO dataset.yaml from the data/ ImageFolder tree."""
    import yaml
    cfg = {
        "path":  str(Path(data_root).resolve()),
        "train": "train",
        "val":   "val",
        "test":  "test",
        "nc":    6,
        "names": CLASSES,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"[Module A] YOLO yaml → {out}")
    return out


if __name__ == "__main__":
    det = IRDetector()
    print(f"[Module A] Loaded on {det.device}")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    print(f"[Module A] Detections on dummy frame: {len(det.detect(dummy))}")
