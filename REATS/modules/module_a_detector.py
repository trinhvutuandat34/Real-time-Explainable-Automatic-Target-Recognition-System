"""
MODULE A — IR Detector
Full self-contained YOLOv4 (CSPDarknet53 + SPP + PANet) for full-scene IR frames.
Target: mAP@0.5 ≥ 75%, FPS ≥ 20, Latency ≤ 40ms/frame
No ultralytics / detectron2 dependency — pure PyTorch + torchvision.
"""

from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import batched_nms

__all__ = [
    "IRDetector", "YOLOv4", "YOLOv4Loss", "MosaicDataset",
    "compute_map", "make_yolo_yaml",
]

import sys as _sys
from pathlib import Path as _Path
_reats_root = str(_Path(__file__).parent.parent)
if _reats_root not in _sys.path:
    _sys.path.insert(0, _reats_root)
from config import CLASSES, NUM_CLASSES

# Default anchors for 640×640 IR imagery (small targets)
ANCHORS = [
    [(12, 16), (19, 36), (40, 28)],    # small objects  → 52×52 head
    [(36, 75), (76, 55), (72, 146)],   # medium objects → 26×26 head
    [(142, 110), (192, 243), (459, 401)],  # large objects → 13×13 head
]
STRIDES = [8, 16, 32]


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

class Mish(nn.Module):
    """x * tanh(softplus(x))"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


# ---------------------------------------------------------------------------
# Basic conv blocks
# ---------------------------------------------------------------------------

class CBM(nn.Module):
    """Conv + BN + Mish (backbone block)."""
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = -1):
        super().__init__()
        pad = (k - 1) // 2 if p < 0 else p
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=pad, bias=False)
        self.bn   = nn.BatchNorm2d(out_c, eps=1e-4, momentum=0.03)
        self.act  = Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class CBL(nn.Module):
    """Conv + BN + LeakyReLU (neck / head block)."""
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = -1):
        super().__init__()
        pad = (k - 1) // 2 if p < 0 else p
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=pad, bias=False)
        self.bn   = nn.BatchNorm2d(out_c, eps=1e-4, momentum=0.03)
        self.act  = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# ---------------------------------------------------------------------------
# Backbone building blocks
# ---------------------------------------------------------------------------

class ResUnit(nn.Module):
    """1×1 then 3×3 residual block (both CBM)."""
    def __init__(self, in_c: int, hidden_c: int):
        super().__init__()
        self.c1 = CBM(in_c, hidden_c, 1)
        self.c2 = CBM(hidden_c, in_c, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.c2(self.c1(x))


class CSPStage(nn.Module):
    """CSP bottleneck: split channels, run N ResUnits on one branch, concat, 1×1 out."""
    def __init__(self, in_c: int, out_c: int, n: int):
        super().__init__()
        mid = out_c // 2
        self.conv_down = CBM(in_c, out_c, 3, 2)   # stride-2 downsample
        self.conv_a    = CBM(out_c, mid, 1)
        self.conv_b    = CBM(out_c, mid, 1)
        self.res_units = nn.Sequential(*[ResUnit(mid, mid // 2) for _ in range(n)])
        self.conv_b_out = CBM(mid, mid, 1)
        self.conv_out  = CBM(mid * 2, out_c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x  = self.conv_down(x)
        a  = self.conv_a(x)
        b  = self.conv_b_out(self.res_units(self.conv_b(x)))
        return self.conv_out(torch.cat([b, a], dim=1))


# ---------------------------------------------------------------------------
# Backbone: CSPDarknet53
# ---------------------------------------------------------------------------

class CSPDarknet53(nn.Module):
    """Returns (C3, C4, C5) at strides (8, 16, 32)."""
    def __init__(self):
        super().__init__()
        self.stem   = CBM(3, 32, 3)                   # 640→640  ch32
        self.stage1 = CSPStage(32, 64, 1)             # 640→320  ch64
        self.stage2 = CSPStage(64, 128, 2)            # 320→160  ch128
        self.stage3 = CSPStage(128, 256, 8)           # 160→80   ch256  ← C3
        self.stage4 = CSPStage(256, 512, 8)           # 80→40    ch512  ← C4
        self.stage5 = CSPStage(512, 1024, 4)          # 40→20    ch1024 ← C5

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x  = self.stem(x)
        x  = self.stage1(x)
        x  = self.stage2(x)
        c3 = self.stage3(x)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return c3, c4, c5


# ---------------------------------------------------------------------------
# Neck: SPP + PANet
# ---------------------------------------------------------------------------

class SPP(nn.Module):
    """Spatial Pyramid Pooling with pool sizes 5/9/13 then 1×1+3×3+1×1."""
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        mid = in_c // 2
        self.pre  = nn.Sequential(CBL(in_c, mid, 1), CBL(mid, in_c, 3), CBL(in_c, mid, 1))
        self.pool5  = nn.MaxPool2d(5, 1, padding=2)
        self.pool9  = nn.MaxPool2d(9, 1, padding=4)
        self.pool13 = nn.MaxPool2d(13, 1, padding=6)
        self.post = nn.Sequential(
            CBL(mid * 4, in_c, 1),
            CBL(in_c, out_c, 3),
            CBL(out_c, out_c // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        x = torch.cat([x, self.pool5(x), self.pool9(x), self.pool13(x)], dim=1)
        return self.post(x)


class _UpBlock(nn.Module):
    """2× upsample + concat + 5 CBLs."""
    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        mid = out_c
        self.lat  = CBL(in_c, out_c, 1)
        self.skip = CBL(skip_c, out_c, 1)
        self.fuse = nn.Sequential(
            CBL(out_c * 2, mid, 1),
            CBL(mid, mid * 2, 3),
            CBL(mid * 2, mid, 1),
            CBL(mid, mid * 2, 3),
            CBL(mid * 2, mid, 1),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(self.lat(x), scale_factor=2, mode="nearest")
        s = self.skip(skip)
        return self.fuse(torch.cat([x, s], dim=1))


class _DownBlock(nn.Module):
    """stride-2 downsample + concat + 5 CBLs."""
    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        mid = out_c
        self.down = CBL(in_c, out_c, 3, 2)
        self.skip = CBL(skip_c, out_c, 1)
        self.fuse = nn.Sequential(
            CBL(out_c * 2, mid, 1),
            CBL(mid, mid * 2, 3),
            CBL(mid * 2, mid, 1),
            CBL(mid, mid * 2, 3),
            CBL(mid * 2, mid, 1),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        d = self.down(x)
        s = self.skip(skip)
        return self.fuse(torch.cat([d, s], dim=1))


class PANet(nn.Module):
    """Top-down FPN + bottom-up path, returns (P3, P4, P5)."""
    def __init__(self):
        super().__init__()
        # Top-down
        self.td_p5 = SPP(1024, 512)
        self.td_p4 = _UpBlock(512, 512, 256)
        self.td_p3 = _UpBlock(256, 256, 128)
        # Bottom-up
        self.bu_p4 = _DownBlock(128, 256, 256)
        self.bu_p5 = _DownBlock(256, 512, 512)

    def forward(
        self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p5 = self.td_p5(c5)            # 20×20 ch512
        p4 = self.td_p4(p5, c4)        # 40×40 ch256
        p3 = self.td_p3(p4, c3)        # 80×80 ch128
        p4 = self.bu_p4(p3, p4)        # 40×40 ch256
        p5 = self.bu_p5(p4, p5)        # 20×20 ch512
        return p3, p4, p5


# ---------------------------------------------------------------------------
# Detection head
# ---------------------------------------------------------------------------

class DetectHead(nn.Module):
    """Final 3×3 CBL + 1×1 Conv → raw (B, A*(5+C), H, W)."""
    def __init__(self, in_c: int, num_anchors: int, num_classes: int):
        super().__init__()
        mid = in_c * 2
        self.conv = nn.Sequential(
            CBL(in_c, mid, 3),
            nn.Conv2d(mid, num_anchors * (5 + num_classes), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------------
# Decode + NMS utilities
# ---------------------------------------------------------------------------

def decode_predictions(
    raw: torch.Tensor,
    anchors: List[Tuple[int, int]],
    stride: int,
    num_classes: int,
) -> torch.Tensor:
    """Sigmoid + anchor offsets → (B, A*H*W, 5+C) with abs (cx,cy,w,h,conf,cls...)."""
    B, _, H, W = raw.shape
    A = len(anchors)
    raw = raw.view(B, A, 5 + num_classes, H, W).permute(0, 1, 3, 4, 2).contiguous()
    raw = torch.sigmoid(raw)

    grid_y = torch.arange(H, dtype=torch.float32, device=raw.device)
    grid_x = torch.arange(W, dtype=torch.float32, device=raw.device)
    gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    grid = torch.stack([gx, gy], dim=-1).view(1, 1, H, W, 2)

    anc_t = torch.tensor(anchors, dtype=torch.float32, device=raw.device).view(1, A, 1, 1, 2)

    xy = (raw[..., :2] * 2 - 0.5 + grid) * stride
    wh = (raw[..., 2:4] * 2) ** 2 * anc_t
    conf_cls = raw[..., 4:]
    out = torch.cat([xy, wh, conf_cls], dim=-1)
    return out.view(B, A * H * W, 5 + num_classes)


def non_max_suppression(
    predictions: List[torch.Tensor],
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
) -> List[Optional[torch.Tensor]]:
    """Batched NMS. predictions: list of (A*H*W, 5+C) tensors per image."""
    results = []
    for pred in predictions:
        conf  = pred[:, 4]
        mask  = conf > conf_thres
        pred  = pred[mask]
        if pred.shape[0] == 0:
            results.append(None)
            continue
        scores = pred[:, 4] * pred[:, 5:].max(dim=1).values
        cls_id = pred[:, 5:].argmax(dim=1)
        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        x1 = cx - w / 2; y1 = cy - h / 2
        x2 = cx + w / 2; y2 = cy + h / 2
        boxes = torch.stack([x1, y1, x2, y2], dim=1)
        keep  = batched_nms(boxes, scores, cls_id, iou_thres)
        kept  = torch.cat([boxes[keep], scores[keep].unsqueeze(1), cls_id[keep].float().unsqueeze(1)], dim=1)
        results.append(kept)
    return results


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class YOLOv4(nn.Module):
    """CSPDarknet53 → SPP+PANet → 3 detection heads."""
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        anchors: List[List[Tuple[int, int]]] = None,
        strides: List[int] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.anchors  = anchors or ANCHORS
        self.strides  = strides or STRIDES

        self.backbone = CSPDarknet53()
        self.neck     = PANet()
        self.head_s   = DetectHead(128, 3, num_classes)   # stride 8  → small
        self.head_m   = DetectHead(256, 3, num_classes)   # stride 16 → medium
        self.head_l   = DetectHead(512, 3, num_classes)   # stride 32 → large

    def forward(
        self, x: torch.Tensor
    ) -> List[torch.Tensor]:
        c3, c4, c5     = self.backbone(x)
        p3, p4, p5     = self.neck(c3, c4, c5)

        raw_s = self.head_s(p3)   # small  anchors  52×52 @ 640
        raw_m = self.head_m(p4)   # medium anchors  26×26 @ 640
        raw_l = self.head_l(p5)   # large  anchors  13×13 @ 640

        if self.training:
            return [raw_s, raw_m, raw_l]

        # Inference: decode all scales and concatenate
        decoded = []
        for raw, anc, stride in zip(
            [raw_s, raw_m, raw_l], self.anchors, self.strides
        ):
            decoded.append(decode_predictions(raw, anc, stride, self.num_classes))
        return [torch.cat(decoded, dim=1)]   # (B, total_det, 5+C)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def ciou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """Complete IoU loss (cx,cy,w,h format)."""
    p_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    p_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    p_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    p_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

    t_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
    t_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
    t_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
    t_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

    inter_x1 = torch.max(p_x1, t_x1); inter_y1 = torch.max(p_y1, t_y1)
    inter_x2 = torch.min(p_x2, t_x2); inter_y2 = torch.min(p_y2, t_y2)
    inter    = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union    = (p_x2 - p_x1) * (p_y2 - p_y1) + (t_x2 - t_x1) * (t_y2 - t_y1) - inter
    iou      = inter / (union + 1e-7)

    cw = torch.max(p_x2, t_x2) - torch.min(p_x1, t_x1)
    ch = torch.max(p_y2, t_y2) - torch.min(p_y1, t_y1)
    c2 = cw ** 2 + ch ** 2 + 1e-7
    rho2 = (pred_boxes[:, 0] - target_boxes[:, 0]) ** 2 + \
           (pred_boxes[:, 1] - target_boxes[:, 1]) ** 2

    v = (4 / math.pi ** 2) * (
        torch.atan(target_boxes[:, 2] / (target_boxes[:, 3] + 1e-7)) -
        torch.atan(pred_boxes[:, 2]   / (pred_boxes[:, 3]   + 1e-7))
    ) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)

    ciou = iou - rho2 / c2 - alpha * v
    return (1 - ciou).mean()


class YOLOv4Loss(nn.Module):
    """Objectness BCE + class BCE + CIoU bbox, positive assignment by IoU threshold."""
    def __init__(
        self,
        anchors: List[List[Tuple[int, int]]] = None,
        strides: List[int] = None,
        num_classes: int = NUM_CLASSES,
        iou_thres: float = 0.5,
    ):
        super().__init__()
        self.anchors     = anchors or ANCHORS
        self.strides     = strides or STRIDES
        self.num_classes = num_classes
        self.iou_thres   = iou_thres
        self.bce         = nn.BCEWithLogitsLoss(reduction="mean")

    def _build_targets(
        self,
        raw: torch.Tensor,
        targets: torch.Tensor,
        anchors: List[Tuple[int, int]],
        stride: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return masks and matched gt for one scale."""
        B, _, H, W = raw.shape
        A = len(anchors)
        obj_mask   = torch.zeros(B, A, H, W, dtype=torch.bool,  device=raw.device)
        noobj_mask = torch.ones( B, A, H, W, dtype=torch.bool,  device=raw.device)
        tx = torch.zeros(B, A, H, W, device=raw.device)
        ty = torch.zeros(B, A, H, W, device=raw.device)
        tw = torch.zeros(B, A, H, W, device=raw.device)
        th = torch.zeros(B, A, H, W, device=raw.device)
        tc = torch.zeros(B, A, H, W, self.num_classes, device=raw.device)

        anc_t = torch.tensor(anchors, dtype=torch.float32, device=raw.device)

        for t in targets:
            b_idx = int(t[0].item())
            cls   = int(t[1].item())
            cx, cy, bw, bh = t[2], t[3], t[4], t[5]    # normalised [0,1]
            cx_g = cx * W; cy_g = cy * H
            gx = int(cx_g); gy = int(cy_g)
            if gx >= W: gx = W - 1
            if gy >= H: gy = H - 1

            bw_p = bw * W * stride  # pixel width
            bh_p = bh * H * stride
            iou_anc = torch.min(anc_t[:, 0], torch.tensor(bw_p)) * \
                      torch.min(anc_t[:, 1], torch.tensor(bh_p)) / \
                      (anc_t[:, 0] * anc_t[:, 1] + bw_p * bh_p -
                       torch.min(anc_t[:, 0], torch.tensor(bw_p)) *
                       torch.min(anc_t[:, 1], torch.tensor(bh_p)) + 1e-7)

            # suppress noobj for anchors with iou > threshold
            for ai in range(A):
                if iou_anc[ai] > self.iou_thres:
                    noobj_mask[b_idx, ai, gy, gx] = False

            best_a = int(iou_anc.argmax().item())
            obj_mask  [b_idx, best_a, gy, gx] = True
            noobj_mask[b_idx, best_a, gy, gx] = False
            tx[b_idx, best_a, gy, gx] = cx_g - gx
            ty[b_idx, best_a, gy, gx] = cy_g - gy
            tw[b_idx, best_a, gy, gx] = bw * W * stride / (anc_t[best_a, 0] + 1e-7)
            th[b_idx, best_a, gy, gx] = bh * H * stride / (anc_t[best_a, 1] + 1e-7)
            tc[b_idx, best_a, gy, gx, cls] = 1.0

        return obj_mask, noobj_mask, torch.stack([tx, ty, tw, th], dim=-1), tc

    def forward(
        self, preds: List[torch.Tensor], targets: torch.Tensor
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=preds[0].device, requires_grad=True)
        for raw, ancs, stride in zip(preds, self.anchors, self.strides):
            B, _, H, W = raw.shape
            A = len(ancs)
            raw_r = raw.view(B, A, 5 + self.num_classes, H, W).permute(0, 1, 3, 4, 2)

            obj_m, noobj_m, gt_box, gt_cls = self._build_targets(raw, targets, ancs, stride)

            pred_xy   = torch.sigmoid(raw_r[..., :2])
            pred_wh   = raw_r[..., 2:4]
            pred_conf = raw_r[..., 4]
            pred_cls  = raw_r[..., 5:]

            # bbox loss on positives (CIoU)
            if obj_m.any():
                anc_t = torch.tensor(ancs, dtype=torch.float32, device=raw.device)
                grid_y = torch.arange(H, dtype=torch.float32, device=raw.device)
                grid_x = torch.arange(W, dtype=torch.float32, device=raw.device)
                gy_g, gx_g = torch.meshgrid(grid_y, grid_x, indexing="ij")
                grid = torch.stack([gx_g, gy_g], dim=-1).unsqueeze(0).unsqueeze(0)
                anc_t2 = anc_t.view(1, A, 1, 1, 2)
                pred_boxes_all = torch.cat([
                    (pred_xy + grid) * stride,
                    (torch.sigmoid(pred_wh) * 2) ** 2 * anc_t2,
                ], dim=-1)
                pred_b = pred_boxes_all[obj_m]
                gt_b   = gt_box[obj_m].clone()
                gt_b[..., 0] = (gt_b[..., 0] + 0) * stride  # already relative offset
                gt_b[..., 1] = (gt_b[..., 1] + 0) * stride
                gt_b[..., 2] = gt_b[..., 2]
                gt_b[..., 3] = gt_b[..., 3]
                box_loss = ciou_loss(pred_b, gt_b)
                cls_loss = self.bce(pred_cls[obj_m], gt_cls[obj_m])
            else:
                box_loss = torch.tensor(0.0, device=raw.device)
                cls_loss = torch.tensor(0.0, device=raw.device)

            obj_loss   = self.bce(pred_conf[obj_m],   torch.ones_like(pred_conf[obj_m]))   if obj_m.any()   else torch.tensor(0.0, device=raw.device)
            noobj_loss = self.bce(pred_conf[noobj_m], torch.zeros_like(pred_conf[noobj_m])) if noobj_m.any() else torch.tensor(0.0, device=raw.device)

            total_loss = total_loss + box_loss + cls_loss + obj_loss + 0.5 * noobj_loss

        return total_loss


# ---------------------------------------------------------------------------
# Dataset: Mosaic augmentation
# ---------------------------------------------------------------------------

class MosaicDataset(Dataset):
    """YOLO-format dataset with 4-image mosaic augmentation."""
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        imgsz: int = 640,
        augment: bool = True,
    ):
        self.imgsz   = imgsz
        self.augment = augment
        img_dir = Path(data_root) / split / "images"
        lbl_dir = Path(data_root) / split / "labels"
        self.images = sorted(img_dir.glob("*.*"))
        self.labels = [lbl_dir / (p.stem + ".txt") for p in self.images]

    def __len__(self) -> int:
        return len(self.images)

    def _load(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        img = cv2.imread(str(self.images[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else np.zeros((self.imgsz, self.imgsz, 3), np.uint8)
        lbl_path = self.labels[idx]
        labels = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f.read().strip().splitlines():
                    labels.append(list(map(float, line.split())))
        return img, np.array(labels, dtype=np.float32).reshape(-1, 5)

    def _mosaic(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        s  = self.imgsz
        cx = random.randint(s // 4, 3 * s // 4)
        cy = random.randint(s // 4, 3 * s // 4)
        indices = [idx] + random.sample(range(len(self)), 3)
        mosaic  = np.zeros((s * 2, s * 2, 3), dtype=np.uint8)
        all_labels: List[np.ndarray] = []

        placements = [(0, 0, cx, cy), (cx, 0, s * 2, cy), (0, cy, cx, s * 2), (cx, cy, s * 2, s * 2)]
        for i, (x1, y1, x2, y2) in enumerate(placements):
            img, labels = self._load(indices[i])
            pw, ph = x2 - x1, y2 - y1
            img    = cv2.resize(img, (pw, ph))
            mosaic[y1:y2, x1:x2] = img
            if labels.shape[0]:
                adj = labels.copy()
                adj[:, 1] = (labels[:, 1] * pw + x1) / (s * 2)
                adj[:, 2] = (labels[:, 2] * ph + y1) / (s * 2)
                adj[:, 3] = labels[:, 3] * pw / (s * 2)
                adj[:, 4] = labels[:, 4] * ph / (s * 2)
                all_labels.append(adj)

        mosaic = cv2.resize(mosaic[cy - s // 2: cy + s // 2, cx - s // 2: cx + s // 2], (s, s))
        combined = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0, 5), dtype=np.float32)
        combined[:, 1:] = combined[:, 1:].clip(0.001, 0.999)
        return mosaic, combined

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, labels = self._mosaic(idx) if self.augment else self._load(idx)
        img = cv2.resize(img, (self.imgsz, self.imgsz)) if img.shape[:2] != (self.imgsz, self.imgsz) else img
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img_t, torch.tensor(labels, dtype=torch.float32)


def _collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    imgs, labels = zip(*batch)
    out_labels = []
    for i, lbl in enumerate(labels):
        if lbl.shape[0]:
            bi = torch.full((lbl.shape[0], 1), i)
            out_labels.append(torch.cat([bi, lbl], dim=1))
    return torch.stack(imgs), torch.cat(out_labels, dim=0) if out_labels else torch.zeros((0, 6))


# ---------------------------------------------------------------------------
# mAP computation
# ---------------------------------------------------------------------------

def compute_map(
    model: YOLOv4,
    data_root: str,
    conf: float = 0.25,
    iou: float = 0.5,
    device: torch.device = None,
) -> float:
    """Compute mAP@0.5 on the val split."""
    device = device or torch.device("cpu")
    model.eval()
    ds     = MosaicDataset(data_root, split="val", augment=False)
    loader = DataLoader(ds, batch_size=4, collate_fn=_collate)

    all_tp: List[int] = []
    all_fp: List[int] = []
    all_scores: List[float] = []
    n_gt = 0

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            preds = model(imgs)  # list[(B, N, 5+C)]
            per_image = [preds[0][i] for i in range(preds[0].shape[0])]
            dets = non_max_suppression(per_image, conf, iou)

            for i, det in enumerate(dets):
                gt_mask = targets[:, 0] == i
                gt      = targets[gt_mask]
                n_gt   += gt.shape[0]
                if det is None or det.shape[0] == 0:
                    continue
                matched = set()
                for d in det:
                    x1, y1, x2, y2, sc, _ = d.tolist()
                    tp = 0
                    for gi in range(gt.shape[0]):
                        if gi in matched:
                            continue
                        gcx, gcy, gw, gh = gt[gi, 2:].tolist()
                        gx1 = (gcx - gw / 2) * imgs.shape[3]
                        gy1 = (gcy - gh / 2) * imgs.shape[2]
                        gx2 = (gcx + gw / 2) * imgs.shape[3]
                        gy2 = (gcy + gh / 2) * imgs.shape[2]
                        ix1 = max(x1, gx1); iy1 = max(y1, gy1)
                        ix2 = min(x2, gx2); iy2 = min(y2, gy2)
                        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                        uni   = (x2-x1)*(y2-y1) + (gx2-gx1)*(gy2-gy1) - inter
                        if uni > 0 and inter / uni >= iou:
                            tp = 1; matched.add(gi); break
                    all_tp.append(tp); all_fp.append(1 - tp); all_scores.append(sc)

    if not all_scores:
        return 0.0
    order  = np.argsort(-np.array(all_scores))
    tp_cum = np.cumsum(np.array(all_tp)[order])
    fp_cum = np.cumsum(np.array(all_fp)[order])
    prec   = tp_cum / (tp_cum + fp_cum + 1e-7)
    rec    = tp_cum / (n_gt + 1e-7)
    # VOC 11-point mAP
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p_at_r = prec[rec >= t].max() if (rec >= t).any() else 0.0
        ap    += p_at_r / 11
    return float(ap)


# ---------------------------------------------------------------------------
# Public API — IRDetector
# ---------------------------------------------------------------------------

class IRDetector:
    """Public detector API wrapping YOLOv4."""

    def __init__(
        self,
        weights: Optional[str] = None,
        conf: float = 0.25,
        iou: float = 0.45,
        num_classes: int = NUM_CLASSES,
        device: Optional[str] = None,
    ):
        self.conf        = conf
        self.iou         = iou
        self.num_classes = num_classes
        self.device      = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model       = YOLOv4(num_classes=num_classes).to(self.device)
        if weights and Path(weights).exists():
            self._load_weights(weights)
        self.model.eval()

    def _load_weights(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        state = ckpt.get("model_state", ckpt)
        self.model.load_state_dict(state)

    def detect(self, frame: np.ndarray) -> List[dict]:
        """Run full pipeline on a single frame (H,W,C) uint8. Returns [{bbox,conf,class_id,class_name}]."""
        h, w = frame.shape[:2]
        inp  = cv2.resize(frame, (640, 640))
        if inp.ndim == 2:
            inp = np.stack([inp] * 3, axis=-1)
        elif inp.shape[2] == 1:
            inp = np.concatenate([inp] * 3, axis=-1)
        t = torch.from_numpy(inp).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        t = t.to(self.device)

        with torch.no_grad():
            preds = self.model(t)   # [(1, N, 5+C)]
        dets = non_max_suppression([preds[0][0]], self.conf, self.iou)[0]

        results = []
        if dets is None:
            return results
        sx = w / 640.0; sy = h / 640.0
        for d in dets.cpu().tolist():
            x1, y1, x2, y2, sc, cls_id = d
            results.append({
                "bbox":       [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)],
                "conf":       round(sc, 4),
                "class_id":   int(cls_id),
                "class_name": CLASSES[int(cls_id)] if int(cls_id) < len(CLASSES) else str(int(cls_id)),
            })
        return results

    def crop_roi(self, frame: np.ndarray, bbox: List[int], pad: int = 10) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        return frame[max(0, y1 - pad): min(h, y2 + pad), max(0, x1 - pad): min(w, x2 + pad)]

    def train(
        self,
        data_root: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 8,
        lr: float = 1e-3,
        workers: int = 4,
    ) -> None:
        """Full training loop with cosine LR; saves best checkpoint to data_root/best.pt."""
        self.model.train()
        ds      = MosaicDataset(data_root, split="train", imgsz=imgsz, augment=True)
        loader  = DataLoader(ds, batch_size=batch, shuffle=True,
                             num_workers=workers, collate_fn=_collate, pin_memory=True)
        loss_fn = YOLOv4Loss(num_classes=self.num_classes)
        optim   = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.937, weight_decay=5e-4, nesterov=True)
        sched   = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, eta_min=lr * 0.01)

        best_map = 0.0
        save_path = Path(data_root) / "best.pt"

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            t0 = time.time()
            for imgs, targets in loader:
                imgs    = imgs.to(self.device)
                targets = targets.to(self.device)
                optim.zero_grad()
                preds = self.model(imgs)
                loss  = loss_fn(preds, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optim.step()
                epoch_loss += loss.item()
            sched.step()

            avg_loss = epoch_loss / max(len(loader), 1)
            elapsed  = time.time() - t0
            print(f"[Epoch {epoch:03d}/{epochs}] loss={avg_loss:.4f}  time={elapsed:.1f}s")

            if epoch % 5 == 0 or epoch == epochs:
                mp = compute_map(self.model, data_root, self.conf, 0.5, self.device)
                print(f"  → mAP@0.5 = {mp:.4f}")
                if mp > best_map:
                    best_map = mp
                    self.save(str(save_path))
                    print(f"  → saved best checkpoint ({best_map:.4f})")
                self.model.train()

    def save(self, path: str) -> None:
        torch.save({"model_state": self.model.state_dict()}, path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "IRDetector":
        det = cls(weights=path, **kwargs)
        return det


# ---------------------------------------------------------------------------
# YOLO dataset YAML helper
# ---------------------------------------------------------------------------

def make_yolo_yaml(data_root: str = "data/", out: str = "data/yolo.yaml") -> str:
    """Write a YOLO dataset.yaml from the data/ directory tree."""
    import yaml
    cfg = {
        "path":  str(Path(data_root).resolve()),
        "train": "train",
        "val":   "val",
        "test":  "test",
        "nc":    NUM_CLASSES,
        "names": CLASSES,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"[Module A] YOLO yaml → {out}")
    return out


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    det   = IRDetector()
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    dets  = det.detect(dummy)
    print(f"[Module A] YOLOv4 on {det.device} — detections on dummy frame: {len(dets)}")
