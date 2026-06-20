"""Visualisation utilities: t-SNE, GradCAM overlay, feature extraction."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.config import CLASSES, cfg


@torch.no_grad()
def _extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[str] = None,
    max_samples: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract penultimate-layer features and labels from loader. Returns (feats, labels)."""
    dev = torch.device(device or cfg.device)
    model.eval()
    model.to(dev)

    feats_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    # Hook on the AdaptiveAvgPool → flatten stage
    acts: list[torch.Tensor] = []

    def _hook(m, inp, out):
        acts.append(out.detach().cpu())

    hook = model.avgpool.register_forward_hook(_hook) if hasattr(model, "avgpool") else None

    for imgs, labels in loader:
        if sum(len(f) for f in feats_list) >= max_samples:
            break
        imgs = imgs.to(dev)
        acts.clear()
        model(imgs)

        if acts:
            f = acts[0].flatten(1).numpy()
        else:
            # Fallback: use the output logits
            out = model(imgs).cpu().numpy()
            f   = out
        feats_list.append(f)
        labels_list.append(labels.numpy())

    if hook:
        hook.remove()

    return np.vstack(feats_list), np.concatenate(labels_list)


def plot_tsne(
    model: nn.Module,
    loader: DataLoader,
    title: str = "Feature t-SNE",
    save_path: Optional[str] = None,
    device: Optional[str] = None,
):
    """2D t-SNE of backbone features, coloured by class. Requires matplotlib + sklearn."""
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        print("[visualise] Install matplotlib and scikit-learn for t-SNE plots.")
        return

    feats, labels = _extract_features(model, loader, device)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
    emb  = tsne.fit_transform(feats)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    for i, cls_name in enumerate(CLASSES):
        mask = labels == i
        if mask.any():
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=[cmap(i)], label=cls_name, alpha=0.6, s=12)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title(title)
    ax.axis("off")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualise] t-SNE saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def grad_cam_overlay(
    model: nn.Module,
    image_tensor: torch.Tensor,
    class_idx: Optional[int] = None,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Returns a GradCAM heatmap overlaid on the input image.

    image_tensor: (1, 3, H, W) normalised to [-1, 1]
    Returns: (H, W, 3) uint8 BGR overlay.
    """
    import cv2

    dev = torch.device(device or cfg.device)
    model.eval()
    model.to(dev)

    acts: list[torch.Tensor] = []
    grads: list[torch.Tensor] = []

    target_layer = model.features[-1][-1]

    def fwd_hook(m, inp, out):
        acts.append(out)

    def bwd_hook(m, gin, gout):
        grads.append(gout[0].detach())

    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    x   = image_tensor.to(dev)
    out = model(x)
    idx = class_idx if class_idx is not None else out.argmax(1).item()
    model.zero_grad()
    out[0, idx].backward()

    fh.remove(); bh.remove()

    if not acts or not grads:
        return np.zeros((224, 224, 3), dtype=np.uint8)

    alpha = grads[0].mean(dim=[-2, -1], keepdim=True)
    cam   = (acts[0] * alpha).sum(dim=1).squeeze(0)
    cam   = torch.relu(cam).detach().cpu().numpy()
    if cam.max() > 0:
        cam /= cam.max()

    H, W = image_tensor.shape[-2:]
    cam_up = cv2.resize(cam, (W, H))
    heatmap = cv2.applyColorMap((cam_up * 255).astype(np.uint8), cv2.COLORMAP_JET)

    img_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_np = ((img_np + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    return cv2.addWeighted(img_bgr, 0.55, heatmap, 0.45, 0)
