"""Centralised configuration for cadet_atr experiments."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import torch


CLASSES = ["fixed_wing", "rotary_wing", "uav", "vessel", "vehicle_ground", "vehicle_apc"]
NUM_CLASSES = len(CLASSES)


@dataclass
class Config:
    # Paths
    data_root:   str = "data"
    ckpt_dir:    str = "checkpoints"

    # Model
    model_name:  str = "convnext_tiny"
    num_classes: int = NUM_CLASSES
    pretrained:  bool = True

    # Training
    batch_size:  int = 32
    lr:          float = 3e-4
    epochs:      int = 50
    weight_decay: float = 1e-2
    warmup_epochs: int  = 5
    label_smoothing: float = 0.1
    grad_clip:   float = 1.0
    img_size:    int = 224

    # Domain adaptation — DANN
    dann_lambda: float = 1.0
    dann_epochs: int   = 30

    # Fine-tuning
    ft_epochs:       int   = 20
    ft_lr:           float = 1e-4
    ft_head_only_epochs: int = 5

    # Evaluation
    gap_threshold: float = 0.10   # warn if domain gap > 10%

    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )


# Module-level singleton
cfg = Config()
