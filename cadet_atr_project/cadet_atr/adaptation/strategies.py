"""
Domain adaptation strategies for cadet_atr.

Four strategies, in increasing adaptation strength:
  1. histogram        — histogram matching of synthetic → real intensity distribution
  2. domain_random    — BackgroundSwapDataset (aggressive IR augmentation, no real data)
  3. finetune         — supervised fine-tuning on real labels (head_only / full / layer_wise)
  4. dann             — Domain-Adversarial Neural Network (GRL + domain classifier)
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from utils.config import cfg, NUM_CLASSES


# ---------------------------------------------------------------------------
# 1. Histogram matching
# ---------------------------------------------------------------------------

def build_reference_histogram(
    loader: DataLoader,
    n_bins: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """
    Build a per-channel cumulative intensity histogram from a DataLoader.
    Returns (3, n_bins) float32 array (cumulative histogram, normalised to [0,1]).
    Images should be normalised to [0,1] or [-1,1]; works with either.
    """
    hist = np.zeros((3, n_bins), dtype=np.float64)
    total = 0

    for imgs, _ in loader:
        imgs_np = imgs.cpu().numpy()
        # De-normalise [-1,1] → [0,1] if needed
        if imgs_np.min() < 0:
            imgs_np = (imgs_np + 1.0) / 2.0
        for c in range(3):
            ch = (imgs_np[:, c] * (n_bins - 1)).astype(int).clip(0, n_bins - 1)
            np.add.at(hist[c], ch.ravel(), 1)
        total += imgs_np.shape[0] * imgs_np.shape[2] * imgs_np.shape[3]

    cdf = hist.cumsum(axis=1) / max(total, 1)
    return cdf.astype(np.float32)


def apply_histogram_matching(
    images: torch.Tensor,
    src_cdf: np.ndarray,
    tgt_cdf: np.ndarray,
    n_bins: int = 256,
) -> torch.Tensor:
    """
    Match the intensity histogram of `images` (normalised to [-1,1]) to
    the target CDF `tgt_cdf`, given `src_cdf` as the current distribution.
    Returns a tensor of the same shape, normalised to [-1,1].
    """
    imgs_np = images.cpu().numpy()
    was_neg = imgs_np.min() < 0
    if was_neg:
        imgs_np = (imgs_np + 1.0) / 2.0

    out = imgs_np.copy()
    for c in range(3):
        # Build lookup: for each bin b, find t such that tgt_cdf[t] ≥ src_cdf[b]
        lut = np.searchsorted(tgt_cdf[c], src_cdf[c]).astype(np.float32) / (n_bins - 1)
        lut = lut.clip(0.0, 1.0)
        ch  = (imgs_np[:, c] * (n_bins - 1)).astype(int).clip(0, n_bins - 1)
        out[:, c] = lut[ch]

    result = torch.from_numpy(out)
    if was_neg:
        result = result * 2.0 - 1.0
    return result.to(images.device)


# ---------------------------------------------------------------------------
# 2. Domain randomisation / BackgroundSwapDataset
# ---------------------------------------------------------------------------

class BackgroundSwapDataset(Dataset):
    """
    Wraps an existing dataset and randomly composites targets onto
    background patches drawn from a background pool.

    If `bg_pool` is None the original image is used (falls back to
    aggressive IR augmentation only).
    """

    def __init__(
        self,
        base_dataset: Dataset,
        bg_pool: Optional[Dataset] = None,
        swap_prob: float = 0.5,
    ):
        self.base      = base_dataset
        self.bg_pool   = bg_pool
        self.swap_prob = swap_prob

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, label = self.base[idx]
        if self.bg_pool is None or torch.rand(1).item() > self.swap_prob:
            return img, label

        bg_idx = torch.randint(len(self.bg_pool), (1,)).item()
        bg, _  = self.bg_pool[bg_idx]

        # Resize bg to match img, blend with random alpha
        if bg.shape != img.shape:
            bg = F.interpolate(bg.unsqueeze(0), img.shape[-2:], mode="bilinear",
                               align_corners=False).squeeze(0)
        alpha = torch.rand(1).item() * 0.4        # 0–40% background
        composited = img * (1 - alpha) + bg * alpha
        return composited, label


# ---------------------------------------------------------------------------
# 3. Fine-tuning
# ---------------------------------------------------------------------------

class RealDataFinetuner:
    """
    Supervised fine-tuning on real (labelled) data.

    Three modes:
      head_only   — freeze backbone, train classifier head only (fastest)
      full        — unfreeze all layers with 10× lower LR on backbone
      layer_wise  — progressively unfreeze from head to backbone
    """

    def __init__(self, model: nn.Module, device: Optional[str] = None):
        self.model  = model
        self.device = torch.device(device or cfg.device)
        self.model.to(self.device)

    def _freeze_backbone(self, freeze: bool) -> None:
        for name, param in self.model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = not freeze

    def finetune(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        mode: str = "full",
        ckpt_path: str = "checkpoints/finetune_best.pt",
        verbose: bool = True,
    ) -> str:
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        if mode == "head_only":
            self._freeze_backbone(True)
            params = [p for p in self.model.parameters() if p.requires_grad]
            opt    = AdamW(params, lr=cfg.ft_lr)
            sched  = CosineAnnealingLR(opt, T_max=cfg.ft_head_only_epochs, eta_min=cfg.ft_lr * 0.01)
            epochs = cfg.ft_head_only_epochs

        elif mode == "full":
            self._freeze_backbone(False)
            backbone_params = [p for n, p in self.model.named_parameters() if "classifier" not in n]
            head_params     = [p for n, p in self.model.named_parameters() if "classifier"     in n]
            opt    = AdamW([
                {"params": backbone_params, "lr": cfg.ft_lr / 10},
                {"params": head_params,     "lr": cfg.ft_lr},
            ], weight_decay=cfg.weight_decay)
            sched  = CosineAnnealingLR(opt, T_max=cfg.ft_epochs, eta_min=cfg.ft_lr * 0.001)
            epochs = cfg.ft_epochs

        elif mode == "layer_wise":
            # Stage 1: head only; Stage 2: full fine-tune
            self.finetune(train_loader, val_loader, "head_only",
                          ckpt_path.replace(".pt", "_stage1.pt"), verbose)
            self.finetune(train_loader, val_loader, "full",
                          ckpt_path, verbose)
            return ckpt_path

        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose from head_only / full / layer_wise")

        best_acc = 0.0
        for epoch in range(1, epochs + 1):
            self.model.train()
            for imgs, labels in train_loader:
                imgs   = imgs.to(self.device)
                labels = labels.to(self.device)
                opt.zero_grad()
                loss = criterion(self.model(imgs), labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                opt.step()
            sched.step()

            self.model.eval()
            with torch.no_grad():
                correct = total = 0
                for imgs, labels in val_loader:
                    imgs   = imgs.to(self.device)
                    labels = labels.to(self.device)
                    correct += (self.model(imgs).argmax(1) == labels).sum().item()
                    total   += imgs.size(0)
            val_acc = correct / max(total, 1)
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({"epoch": epoch, "model_state": self.model.state_dict(),
                            "val_acc": best_acc, "mode": mode}, ckpt_path)
            if verbose:
                print(f"[finetune/{mode}] epoch={epoch}/{epochs}  "
                      f"val_acc={val_acc:.3f}  best={best_acc:.3f}")

        return ckpt_path


# ---------------------------------------------------------------------------
# 4. DANN — Domain-Adversarial Neural Network
# ---------------------------------------------------------------------------

class _GradientReversalFn(torch.autograd.Function):
    """Reverses gradient during backward pass (multiplied by -lambda)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lam * grad_output, None


class GRL(nn.Module):
    def __init__(self, lam: float = 1.0):
        super().__init__()
        self.lam = lam

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFn.apply(x, self.lam)


class DANNModel(nn.Module):
    """
    ConvNeXt backbone + task head + GRL + domain classifier.

    forward(x, return_domain=False) → class logits
    forward(x, return_domain=True)  → (class logits, domain logits)
    """

    def __init__(self, backbone: nn.Module, num_classes: int = NUM_CLASSES, lam: float = 1.0):
        super().__init__()
        # backbone = model.features; pool + flatten handled below
        self.backbone = backbone

        # Infer feature dim from a dummy pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat  = backbone(dummy)
            feat  = feat.mean([-2, -1])
            feat_dim = feat.shape[1]

        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.task_head  = nn.Linear(feat_dim, num_classes)
        self.grl        = GRL(lam)
        self.domain_clf = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),   # 0=synthetic, 1=real
        )

    def set_lambda(self, lam: float) -> None:
        self.grl.lam = lam

    def forward(self, x: torch.Tensor, return_domain: bool = False):
        feat  = self.backbone(x)
        feat  = self.pool(feat).flatten(1)
        cls   = self.task_head(feat)
        if not return_domain:
            return cls
        rev   = self.grl(feat)
        dom   = self.domain_clf(rev)
        return cls, dom


class DANNTrainer:
    """
    Train DANNModel with both source (synthetic) and target (real, unlabelled) data.

    lambda is annealed from 0 → dann_lambda following Ganin et al. (2016).
    """

    def __init__(self, dann_model: DANNModel, device: Optional[str] = None):
        self.model  = dann_model
        self.device = torch.device(device or cfg.device)
        self.model.to(self.device)

    def train(
        self,
        src_loader:  DataLoader,
        tgt_loader:  DataLoader,
        val_loader:  DataLoader,
        ckpt_path:   str = "checkpoints/dann_best.pt",
        verbose:     bool = True,
    ) -> str:
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        cls_crit = nn.CrossEntropyLoss(label_smoothing=0.1)
        dom_crit = nn.CrossEntropyLoss()

        opt   = AdamW(self.model.parameters(), lr=cfg.ft_lr, weight_decay=cfg.weight_decay)
        sched = CosineAnnealingLR(opt, T_max=cfg.dann_epochs, eta_min=cfg.ft_lr * 0.01)

        best_acc = 0.0
        tgt_iter = iter(self._cycle(tgt_loader))

        for epoch in range(1, cfg.dann_epochs + 1):
            # Anneal lambda: 0 → cfg.dann_lambda over training
            p     = epoch / cfg.dann_epochs
            lam   = cfg.dann_lambda * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)
            self.model.set_lambda(lam)
            self.model.train()

            for src_imgs, src_labels in src_loader:
                src_imgs   = src_imgs.to(self.device)
                src_labels = src_labels.to(self.device)

                try:
                    tgt_imgs, _ = next(tgt_iter)
                except StopIteration:
                    tgt_iter = iter(self._cycle(tgt_loader))
                    tgt_imgs, _ = next(tgt_iter)
                tgt_imgs = tgt_imgs.to(self.device)

                # Domain labels: 0=source, 1=target
                src_dom = torch.zeros(src_imgs.size(0), dtype=torch.long, device=self.device)
                tgt_dom = torch.ones( tgt_imgs.size(0), dtype=torch.long, device=self.device)

                # Source: class loss + domain loss
                cls_out, dom_src = self.model(src_imgs, return_domain=True)
                loss_cls = cls_crit(cls_out, src_labels)
                loss_dom_src = dom_crit(dom_src, src_dom)

                # Target: domain loss only
                _, dom_tgt = self.model(tgt_imgs, return_domain=True)
                loss_dom_tgt = dom_crit(dom_tgt, tgt_dom)

                loss = loss_cls + 0.5 * (loss_dom_src + loss_dom_tgt)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                opt.step()

            sched.step()

            # Validation (class accuracy on synthetic val)
            self.model.eval()
            with torch.no_grad():
                correct = total = 0
                for imgs, labels in val_loader:
                    imgs   = imgs.to(self.device)
                    labels = labels.to(self.device)
                    preds  = self.model(imgs, return_domain=False).argmax(1)
                    correct += (preds == labels).sum().item()
                    total   += imgs.size(0)
            val_acc = correct / max(total, 1)
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({"epoch": epoch, "model_state": self.model.state_dict(),
                            "val_acc": best_acc}, ckpt_path)
            if verbose:
                print(f"[DANN] epoch={epoch}/{cfg.dann_epochs}  "
                      f"lam={lam:.3f}  val_acc={val_acc:.3f}  best={best_acc:.3f}")

        return ckpt_path

    @staticmethod
    def _cycle(loader: DataLoader) -> Iterator:
        while True:
            yield from loader
