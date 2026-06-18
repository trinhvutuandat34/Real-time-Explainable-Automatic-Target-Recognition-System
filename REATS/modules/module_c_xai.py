"""
MODULE C — XAI Engine
Explainability: Grad-CAM + SHAP + MC Dropout uncertainty.
Target: Faithfulness Deletion/Insertion AUC ≥ 0.80
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class MCDropoutWrapper(nn.Module):
    """Bayesian approximation via MC Dropout — n_samples forward passes at inference."""

    def __init__(self, model: nn.Module, n_samples: int = 20):
        super().__init__()
        self.model     = model
        self.n_samples = n_samples

    def _enable_dropout(self):
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict:
        """
        Returns:
            mean_probs:  (B, C) mean softmax probs across samples
            uncertainty: (B,)  predictive entropy (high → OOD warning)
            all_probs:   (n_samples, B, C)
        """
        self.model.eval()
        self._enable_dropout()
        probs_list = [
            torch.softmax(self.model(x), dim=-1) for _ in range(self.n_samples)
        ]
        all_probs  = torch.stack(probs_list)           # (S, B, C)
        mean_probs = all_probs.mean(dim=0)              # (B, C)
        entropy    = -(mean_probs * mean_probs.clamp(min=1e-8).log()).sum(dim=-1)
        return {"mean_probs": mean_probs, "uncertainty": entropy, "all_probs": all_probs}


class GradCAMExplainer:
    """Grad-CAM heatmaps via pytorch-grad-cam. Defaults to last ConvNeXt stage."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        self._Target = ClassifierOutputTarget
        layer        = target_layer if target_layer is not None else model.features[-1][-1]
        self.cam     = GradCAM(model=model, target_layers=[layer])

    def explain(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """Return (H, W) heatmap in [0, 1] for x: (1, C, H, W)."""
        return self.cam(input_tensor=x, targets=[self._Target(class_idx)])[0]


class SHAPExplainer:
    """SHAP DeepExplainer — pixel-level attribution for ConvNeXt."""

    def __init__(self, model: nn.Module, background: torch.Tensor):
        import shap
        self.explainer = shap.DeepExplainer(model, background)

    def explain(self, x: torch.Tensor) -> np.ndarray:
        """Return (B, C, H, W) SHAP values."""
        return np.array(self.explainer.shap_values(x))


def faithfulness_deletion_auc(
    model:     nn.Module,
    x:         torch.Tensor,
    saliency:  np.ndarray,
    class_idx: int,
    steps:     int = 20,
    device:    str = "cpu",
) -> float:
    """
    Mask top-saliency pixels progressively; measure class-prob drop.
    Normalised AUC. Target ≥ 0.80.
    """
    model.eval().to(device)
    x     = x.clone().to(device)
    order = np.argsort(saliency.flatten())[::-1]
    step  = len(order) // steps
    scores = []

    with torch.no_grad():
        for i in range(steps + 1):
            x_flat = x.view(x.shape[0], x.shape[1], -1).clone()
            if i > 0:
                x_flat[:, :, order[: i * step]] = 0
            prob = torch.softmax(model(x_flat.view(x.shape)), dim=-1)[0, class_idx].item()
            scores.append(prob)

    baseline = scores[0]
    return float(np.trapz(scores, dx=1.0 / steps) / baseline) if baseline > 0 else 0.0


def faithfulness_insertion_auc(
    model:     nn.Module,
    x:         torch.Tensor,
    saliency:  np.ndarray,
    class_idx: int,
    steps:     int = 20,
    device:    str = "cpu",
) -> float:
    """
    Reveal top-saliency pixels progressively; measure class-prob rise.
    Normalised AUC. Target ≥ 0.80.
    """
    model.eval().to(device)
    x      = x.clone().to(device)
    order  = np.argsort(saliency.flatten())[::-1]
    step   = len(order) // steps
    blank  = torch.zeros_like(x)
    scores = []

    with torch.no_grad():
        for i in range(steps + 1):
            revealed = blank.clone()
            src_flat = x.view(x.shape[0], x.shape[1], -1)
            rev_flat = revealed.view(x.shape[0], x.shape[1], -1)
            if i > 0:
                rev_flat[:, :, order[: i * step]] = src_flat[:, :, order[: i * step]]
            prob = torch.softmax(model(rev_flat.view(x.shape)), dim=-1)[0, class_idx].item()
            scores.append(prob)

    final = scores[-1]
    return float(np.trapz(scores, dx=1.0 / steps) / final) if final > 0 else 0.0


def explain_prediction(
    model:      nn.Module,
    x:          torch.Tensor,
    class_idx:  int,
    background: Optional[torch.Tensor] = None,
    device:     str = "cpu",
) -> dict:
    """Full XAI report: Grad-CAM, MC Dropout, Faithfulness AUC, optional SHAP."""
    model.to(device)
    x = x.to(device)

    heatmap  = GradCAMExplainer(model).explain(x, class_idx)
    mc_out   = MCDropoutWrapper(model)(x)
    del_auc  = faithfulness_deletion_auc(model, x, heatmap, class_idx, device=device)
    ins_auc  = faithfulness_insertion_auc(model, x, heatmap, class_idx, device=device)

    result = {
        "heatmap":                     heatmap,
        "uncertainty":                 mc_out["uncertainty"].cpu().numpy(),
        "mean_probs":                  mc_out["mean_probs"].cpu().numpy(),
        "faithfulness_deletion_auc":   del_auc,
        "faithfulness_insertion_auc":  ins_auc,
    }
    if background is not None:
        result["shap_values"] = SHAPExplainer(model, background.to(device)).explain(x)
    return result


if __name__ == "__main__":
    print("[Module C] XAI Engine — Grad-CAM + SHAP + MC Dropout")
    print("[Module C] Target: Faithfulness Deletion/Insertion AUC ≥ 0.80")
