"""
Hard-negative mining for confusable target classes.

The KCI (2025) paper's confusion matrix (Table 5) shows MiG-19 and MiG-21 are most
often misclassified as F-16 — the three fighters share a very similar IR silhouette.
The Kaggle GPU run (2026-07-04, 43-class taxonomy, single ConvNeXt_tiny, 93.12% test
accuracy) reproduced that exact bleed (Su27/MiG21 -> F16) and surfaced a second,
larger cluster: armored ground vehicles. GROUND-domain accuracy (85.2%) trailed
AIR (95.7%) and NAVAL (99.75%) specifically because of BMP2/Bradley/K21 IFV bleed and
general T72/T90/Leopard2 MBT bleed — both are silhouette-similar tracked/wheeled
vehicle families, the same underlying failure mode as the fighter-jet confusion.
Standard cross-entropy training over the full dataset treats every sample equally, so
these rare, hard-to-separate examples get drowned out by easy ones. This module:

  1. mines "hard negatives" — samples from a confusable class group that the model
     either misclassifies or is only barely confident about (small top1/top2 margin)
  2. oversamples them into a fine-tuning subset (HardNegativeDataset)
  3. runs a short, low-LR additional fine-tuning pass concentrated on that subset
     (finetune_on_hard_negatives), on top of an already-trained checkpoint

This is a post-hoc addition to the normal 300-epoch schedule in module_b_classifier,
not a replacement for it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Set

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

_reats_root = str(Path(__file__).parent.parent)
if _reats_root not in sys.path:
    sys.path.insert(0, _reats_root)

from config import CLASSES, NUM_CLASSES

# Confusable groups — visually similar silhouettes prone to cross-misclassification.
# F-16 / MiG-19 / MiG-21: called out explicitly by the professor (highest confusion
# rate in the paper's confusion matrix); confirmed on the 43-class taxonomy by the
# 2026-07-04 Kaggle run's own confusion matrix (Su27/MiG21 -> F16 bleed).
# IFV/APC and MBT: two distinct armored-vehicle clusters the same Kaggle run
# surfaced (kept separate — the report describes them as two clusters, not one
# six-way confusion, and IFVs/APCs aren't usually mistaken for MBTs).
# Extend this list as new confusions surface from future confusion-matrix runs.
CONFUSABLE_GROUPS: List[Set[str]] = [
    {"F16", "MiG19", "MiG21"},
    {"BMP2", "Bradley", "K21"},
    {"T72", "T90", "Leopard2"},
]


def _confusable_class_indices(groups: List[Set[str]] = CONFUSABLE_GROUPS) -> Set[int]:
    names = set().union(*groups) if groups else set()
    return {CLASSES.index(n) for n in names if n in CLASSES}


def _labels_without_loading(dataset: Dataset) -> Optional[List[int]]:
    """Per-sample labels without decoding any image, when the dataset exposes them
    (ImageFolder: .targets / .samples). Returns None if only __getitem__ can tell."""
    targets = getattr(dataset, "targets", None)
    if targets is not None:
        return [int(t) for t in targets]
    samples = getattr(dataset, "samples", None)
    if samples is not None:
        return [int(lbl) for _, lbl in samples]
    return None


@torch.no_grad()
def mine_hard_negatives(
    model: nn.Module,
    dataset: Dataset,
    device: str,
    confusable_ids: Optional[Set[int]] = None,
    margin_thresh: float = 0.15,
    batch_size: int = 64,
) -> List[int]:
    """Return dataset indices that are hard negatives.

    A sample (restricted to `confusable_ids` classes, default: CONFUSABLE_GROUPS) is
    a hard negative if the model misclassifies it, or if its softmax top1/top2 margin
    is below `margin_thresh` (the model is unsure, even when it happens to get it right).

    When the dataset exposes labels without image loading (ImageFolder), only the
    confusable-class subset is ever decoded and forward-passed — for 3 confusable
    classes out of 43 that skips ~93% of the inference work.
    """
    if confusable_ids is None:
        confusable_ids = _confusable_class_indices()

    model.eval()
    model.to(device)

    # index_map[j] = original dataset index of the j-th sample in iteration order
    labels_all = _labels_without_loading(dataset) if confusable_ids else None
    if labels_all is not None:
        index_map = [i for i, l in enumerate(labels_all) if l in confusable_ids]
        if not index_map:
            return []
        loader = DataLoader(
            torch.utils.data.Subset(dataset, index_map),
            batch_size=batch_size, shuffle=False,
        )
    else:
        index_map = list(range(len(dataset)))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    hard_indices: List[int] = []
    offset = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        out = model(imgs)
        probs = torch.softmax(out, dim=-1) if not hasattr(model, "models") else out
        top2 = probs.topk(min(2, probs.size(-1)), dim=-1)
        top1_idx = top2.indices[:, 0]
        margin = top2.values[:, 0] - top2.values[:, -1]

        for i in range(imgs.size(0)):
            true_label = int(labels[i])
            if confusable_ids and true_label not in confusable_ids:
                continue
            misclassified = int(top1_idx[i]) != true_label
            low_margin = float(margin[i]) < margin_thresh
            if misclassified or low_margin:
                hard_indices.append(index_map[offset + i])
        offset += imgs.size(0)

    return hard_indices


class HardNegativeDataset(Dataset):
    """Wraps `base_dataset`, oversampling `hard_indices` — the professor's requirement to
    "extract data with a high probability of misidentification and perform additional
    training." Regular samples keep their normal rate so the fine-tune pass does not
    catastrophically forget other classes; only hard negatives repeat `oversample_factor`
    times."""

    def __init__(self, base_dataset: Dataset, hard_indices: List[int], oversample_factor: int = 4):
        self.base_dataset = base_dataset
        self._index_map = list(range(len(base_dataset))) + hard_indices * max(oversample_factor - 1, 0)

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int):
        return self.base_dataset[self._index_map[idx]]


def finetune_on_hard_negatives(
    model: nn.Module,
    base_train_dataset: Dataset,
    val_loader: DataLoader,
    device: str,
    arch: str = "convnext_tiny",
    hard_indices: Optional[List[int]] = None,
    oversample_factor: int = 4,
    epochs: int = 15,
    lr: float = 1e-5,
    batch_size: int = 64,
    ckpt_path: str = "checkpoints/hard_negative_finetuned.pth",
) -> float:
    """Short, low-LR fine-tuning pass over an oversampled hard-negative subset, run on
    top of an already fully-trained checkpoint. Returns the best validation accuracy
    reached during the pass."""
    from modules.module_b_classifier import LabelSmoothingCrossEntropy, evaluate
    from torch.optim import AdamW

    if hard_indices is None:
        hard_indices = mine_hard_negatives(model, base_train_dataset, device)
    if not hard_indices:
        print("[hard_negative_mining] No hard negatives found — skipping fine-tune.")
        return 0.0

    hn_dataset = HardNegativeDataset(base_train_dataset, hard_indices, oversample_factor)
    hn_loader = DataLoader(hn_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = LabelSmoothingCrossEntropy(0.1)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in hn_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            correct    += (out.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"[hard-neg epoch {epoch:2d}] train_loss={total_loss/len(hn_loader):.4f} "
            f"train_acc={correct/max(total, 1):.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"state_dict": model.state_dict(), "epoch": epoch,
                 "best_val_acc": best_val_acc, "arch": arch},
                ckpt_path,
            )

    return best_val_acc


def main() -> None:
    import argparse
    from modules.module_b_classifier import build_model, build_loaders, CONFIG, ARCHITECTURES

    parser = argparse.ArgumentParser(description="Hard-negative mining + fine-tune pass")
    parser.add_argument("--checkpoint", required=True, help="Base checkpoint to fine-tune")
    parser.add_argument("--arch", default="convnext_tiny", choices=ARCHITECTURES)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--oversample", type=int, default=4)
    parser.add_argument("--out", default="checkpoints/hard_negative_finetuned.pth")
    args = parser.parse_args()

    device = CONFIG["device"]
    model = build_model(args.arch, NUM_CLASSES, pretrained=False).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    state = (ckpt.get("ema_state_dict") or ckpt.get("state_dict") or ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)

    train_loader, val_loader, _ = build_loaders(CONFIG)
    hard_idx = mine_hard_negatives(model, train_loader.dataset, device)
    print(
        f"[hard_negative_mining] {len(hard_idx)} hard negatives found out of "
        f"{len(train_loader.dataset)} training samples "
        f"({len(hard_idx) / max(len(train_loader.dataset), 1):.1%})"
    )

    best_acc = finetune_on_hard_negatives(
        model, train_loader.dataset, val_loader, device,
        arch=args.arch, hard_indices=hard_idx,
        oversample_factor=args.oversample, epochs=args.epochs, lr=args.lr,
        ckpt_path=args.out,
    )
    print(f"[hard_negative_mining] best val acc after fine-tune: {best_acc:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
