"""Domain gap measurement and evaluation utilities."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.config import cfg


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, device: Optional[str] = None) -> float:
    """Top-1 accuracy of model on loader."""
    dev = torch.device(device or cfg.device)
    model.eval()
    model.to(dev)
    correct = total = 0
    for imgs, labels in loader:
        imgs   = imgs.to(dev)
        labels = labels.to(dev)
        preds  = model(imgs).argmax(1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)
    return correct / max(total, 1)


def measure_domain_gap(
    model: nn.Module,
    synth_loader: DataLoader,
    real_loader:  Optional[DataLoader] = None,
    device: Optional[str] = None,
) -> dict:
    """
    Measure synthetic accuracy, real accuracy (if data available), and domain gap.

    Returns:
        {
          "synth_acc":   float,
          "real_acc":    float | None,
          "domain_gap":  float | None,
          "gap_warning": bool,
        }
    """
    synth_acc = accuracy(model, synth_loader, device)
    real_acc  = None
    if real_loader is not None:
        real_acc = accuracy(model, real_loader, device)

    gap = (synth_acc - real_acc) if real_acc is not None else None

    result = {
        "synth_acc":   synth_acc,
        "real_acc":    real_acc,
        "domain_gap":  gap,
        "gap_warning": gap is not None and gap > cfg.gap_threshold,
    }

    print(f"[evaluator] Synthetic acc : {synth_acc:.3f}")
    if real_acc is not None:
        print(f"[evaluator] Real acc      : {real_acc:.3f}")
        print(f"[evaluator] Domain gap    : {gap:.3f}"
              + (" ⚠ WARNING: gap > threshold" if result["gap_warning"] else " ✓"))
    else:
        print("[evaluator] No real data loader — skipping real accuracy.")

    return result


def load_model(model: nn.Module, ckpt_path: str, device: Optional[str] = None) -> nn.Module:
    """Load checkpoint state dict into model in-place and return it."""
    dev  = torch.device(device or cfg.device)
    ckpt = torch.load(ckpt_path, map_location=dev)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.to(dev)
    return model
