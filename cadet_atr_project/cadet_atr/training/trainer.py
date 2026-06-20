"""Training loop for cadet_atr baseline and adaptation."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from utils.config import cfg


class Trainer:
    """Standard supervised trainer with warmup-cosine LR and gradient clipping."""

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        lr: Optional[float] = None,
        epochs: Optional[int] = None,
        weight_decay: Optional[float] = None,
        warmup_epochs: Optional[int] = None,
        grad_clip: Optional[float] = None,
        label_smoothing: Optional[float] = None,
    ):
        self.model         = model
        self.device        = torch.device(device or cfg.device)
        self.lr            = lr          or cfg.lr
        self.epochs        = epochs      or cfg.epochs
        self.weight_decay  = weight_decay  or cfg.weight_decay
        self.warmup_epochs = warmup_epochs or cfg.warmup_epochs
        self.grad_clip     = grad_clip     or cfg.grad_clip
        self.label_smooth  = label_smoothing if label_smoothing is not None else cfg.label_smoothing

        self.model.to(self.device)

    def _build_optimizer_scheduler(self):
        opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        total = self.epochs

        def _lr_lambda(epoch: int) -> float:
            if epoch < self.warmup_epochs:
                return epoch / max(self.warmup_epochs, 1)
            progress = (epoch - self.warmup_epochs) / max(total - self.warmup_epochs, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        sched = LambdaLR(opt, _lr_lambda)
        return opt, sched

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
        self.model.eval()
        correct = total = 0
        loss_sum = 0.0
        for imgs, labels in loader:
            imgs   = imgs.to(self.device)
            labels = labels.to(self.device)
            out    = self.model(imgs)
            loss_sum += criterion(out, labels).item() * imgs.size(0)
            correct  += (out.argmax(1) == labels).sum().item()
            total    += imgs.size(0)
        return loss_sum / max(total, 1), correct / max(total, 1)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        ckpt_path:    str = "checkpoints/baseline_best.pt",
        verbose:      bool = True,
    ) -> str:
        """Train and save best checkpoint. Returns path to saved checkpoint."""
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smooth)
        opt, sched = self._build_optimizer_scheduler()

        best_acc = 0.0
        scaler   = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            t0 = time.perf_counter()
            train_loss = 0.0
            correct = total = 0

            for imgs, labels in train_loader:
                imgs   = imgs.to(self.device)
                labels = labels.to(self.device)
                opt.zero_grad()

                with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                    out  = self.model(imgs)
                    loss = criterion(out, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                scaler.step(opt)
                scaler.update()

                train_loss += loss.item() * imgs.size(0)
                correct    += (out.argmax(1) == labels).sum().item()
                total      += imgs.size(0)

            sched.step()
            val_loss, val_acc = self._evaluate(val_loader, criterion)
            elapsed = time.perf_counter() - t0

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({"epoch": epoch, "model_state": self.model.state_dict(),
                            "val_acc": best_acc}, ckpt_path)

            if verbose:
                tr_acc = correct / max(total, 1)
                print(f"[Epoch {epoch:03d}/{self.epochs}]  "
                      f"train_loss={train_loss/total:.4f}  train_acc={tr_acc:.3f}  "
                      f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
                      f"best={best_acc:.3f}  ({elapsed:.1f}s)")

        return ckpt_path
