"""
MODULE C — XAI Engine
Explainability: Grad-CAM + SHAP + MC Dropout uncertainty.
Target: Faithfulness Deletion/Insertion AUC ≥ 0.80
"""

import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable


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
        """Returns mean_probs (B,C), uncertainty (B,), all_probs (S,B,C)."""
        self.model.eval()
        self._enable_dropout()
        probs_list = [torch.softmax(self.model(x), dim=-1) for _ in range(self.n_samples)]
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


class GradCAMPlusPlusExplainer:
    """Grad-CAM++ — better localisation for multiple object instances."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        self._Target = ClassifierOutputTarget
        layer        = target_layer if target_layer is not None else model.features[-1][-1]
        self.cam     = GradCAMPlusPlus(model=model, target_layers=[layer])

    def explain(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """Return (H, W) heatmap in [0, 1] for x: (1, C, H, W)."""
        return self.cam(input_tensor=x, targets=[self._Target(class_idx)])[0]


class EigenCAMExplainer:
    """EigenCAM — gradient-free CAM, robust when gradients are unstable."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        from pytorch_grad_cam import EigenCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        self._Target = ClassifierOutputTarget
        layer        = target_layer if target_layer is not None else model.features[-1][-1]
        self.cam     = EigenCAM(model=model, target_layers=[layer])

    def explain(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """Return (H, W) heatmap in [0, 1] for x: (1, C, H, W)."""
        return self.cam(input_tensor=x, targets=[self._Target(class_idx)])[0]


class SHAPExplainer:
    """SHAP DeepExplainer — pixel-level attribution for ConvNeXt."""

    def __init__(self, model: nn.Module, background: torch.Tensor):
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP not installed. Run: pip install shap")
        self.explainer = shap.DeepExplainer(model, background)

    def explain(self, x: torch.Tensor) -> np.ndarray:
        """Return (B, C, H, W) SHAP values."""
        return np.array(self.explainer.shap_values(x))


class LIMEExplainer:
    """LIME superpixel explanations — requires pip install lime."""

    def __init__(self, model: nn.Module, transform: Callable, device: str = "cpu"):
        self.model     = model
        self.transform = transform
        self.device    = device

    def explain(
        self,
        img_np:       np.ndarray,
        class_idx:    int,
        num_samples:  int = 500,
        num_features: int = 20,
    ) -> np.ndarray:
        """Return (H, W) superpixel importance map in [0, 1]."""
        try:
            from lime import lime_image
        except ImportError:
            raise ImportError("LIME not installed. Run: pip install lime")

        def _predict(imgs: np.ndarray) -> np.ndarray:
            self.model.eval()
            tensors = torch.stack([self.transform(img) for img in imgs]).to(self.device)
            with torch.no_grad():
                return torch.softmax(self.model(tensors), dim=-1).cpu().numpy()

        exp      = lime_image.LimeImageExplainer()
        result   = exp.explain_instance(img_np.astype(np.double), _predict,
                                        top_labels=class_idx + 1, num_samples=num_samples)
        _, mask  = result.get_image_and_mask(class_idx, positive_only=False,
                                             num_features=num_features, hide_rest=False)
        heatmap  = mask.astype(np.float32)
        mn, mx   = heatmap.min(), heatmap.max()
        return (heatmap - mn) / (mx - mn) if mx > mn else heatmap


def overlay_heatmap(
    img_np:   np.ndarray,
    heatmap:  np.ndarray,
    alpha:    float = 0.5,
    colormap: int   = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Return BGR overlay of heatmap on (H, W, 3) image."""
    h, w         = img_np.shape[:2]
    heat_uint8   = np.uint8(255 * np.clip(heatmap, 0, 1))
    heat_resized = cv2.resize(heat_uint8, (w, h))
    colored      = cv2.applyColorMap(heat_resized, colormap)
    base_bgr     = img_np[..., ::-1].copy() if img_np.shape[2] == 3 else img_np
    return cv2.addWeighted(base_bgr, 1 - alpha, colored, alpha, 0)


def save_explanation(
    img_np:     np.ndarray,
    heatmap:    np.ndarray,
    save_path:  str,
    class_name: str,
    confidence: float,
) -> None:
    """Save side-by-side (original | heatmap overlay) PNG with class/confidence title."""
    overlay = overlay_heatmap(img_np, heatmap)[..., ::-1]  # BGR -> RGB
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img_np);  axes[0].set_title("Original");        axes[0].axis("off")
    axes[1].imshow(overlay); axes[1].set_title("Heatmap Overlay"); axes[1].axis("off")
    fig.suptitle(f"{class_name}  |  conf={confidence:.3f}", fontsize=13)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def faithfulness_deletion_auc(
    model:     nn.Module,
    x:         torch.Tensor,
    saliency:  np.ndarray,
    class_idx: int,
    steps:     int = 20,
    device:    str = "cpu",
) -> float:
    """Mask top-saliency pixels progressively; AUC of class-prob drop. Target ≥ 0.80."""
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
    """Reveal top-saliency pixels progressively; AUC of class-prob rise. Target ≥ 0.80."""
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


def _get_cam_explainer(
    model: nn.Module, method: str, target_layer: Optional[nn.Module] = None
) -> Any:
    """Return the correct CAM explainer instance for the given method string."""
    m = method.lower()
    if m == "gradcam":       return GradCAMExplainer(model, target_layer)
    if m in ("gradcam++", "gradcampp"): return GradCAMPlusPlusExplainer(model, target_layer)
    if m == "eigencam":      return EigenCAMExplainer(model, target_layer)
    raise ValueError(f"Unknown method '{method}'. Choose: gradcam | gradcam++ | eigencam")


def explain_prediction(
    model:        nn.Module,
    x:            torch.Tensor,
    class_idx:    int,
    background:   Optional[torch.Tensor] = None,
    device:       str = "cpu",
    method:       str = "gradcam",
    target_layer: Optional[nn.Module] = None,
    save_path:    Optional[str] = None,
    class_name:   str = "target",
) -> Dict[str, Any]:
    """Full XAI report: CAM heatmap, MC Dropout uncertainty, Faithfulness AUC, optional SHAP."""
    model.to(device)
    x = x.to(device)

    heatmap    = _get_cam_explainer(model, method, target_layer).explain(x, class_idx)
    mc_out     = MCDropoutWrapper(model)(x)
    mean_probs = mc_out["mean_probs"].cpu().numpy()
    del_auc    = faithfulness_deletion_auc(model, x, heatmap, class_idx, device=device)
    ins_auc    = faithfulness_insertion_auc(model, x, heatmap, class_idx, device=device)

    result: Dict[str, Any] = {
        "heatmap":                    heatmap,
        "uncertainty":                mc_out["uncertainty"].cpu().numpy(),
        "mean_probs":                 mean_probs,
        "faithfulness_deletion_auc":  del_auc,
        "faithfulness_insertion_auc": ins_auc,
        "method":                     method,
    }
    if background is not None:
        result["shap_values"] = SHAPExplainer(model, background.to(device)).explain(x)
    if save_path is not None:
        img_np = np.clip(x[0].cpu().permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)
        save_explanation(img_np, heatmap, save_path, class_name, float(mean_probs[0, class_idx]))
    return result


def batch_explain(
    model:     nn.Module,
    loader:    Any,
    n_samples: int,
    device:    str,
    method:    str = "gradcam",
    save_dir:  Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run explain_prediction on first n_samples from loader; optionally save PNGs."""
    results: List[Dict[str, Any]] = []
    seen = 0
    for batch_x, batch_y in loader:
        for i in range(batch_x.shape[0]):
            if seen >= n_samples:
                break
            cls = int(batch_y[i].item())
            sp  = str(Path(save_dir) / f"sample_{seen:04d}_cls{cls}.png") if save_dir else None
            res = explain_prediction(model, batch_x[i:i+1], cls, device=device,
                                     method=method, save_path=sp, class_name=str(cls))
            res["class_idx"] = cls
            results.append(res)
            seen += 1
        if seen >= n_samples:
            break
    return results


def _bin_calibration(
    confs: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> Dict[str, Any]:
    """Shared binning logic for calibration_plot and mc_dropout_calibration_plot."""
    bins                           = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_cnts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            bin_accs.append(0.0); bin_confs.append(float((lo + hi) / 2)); bin_cnts.append(0)
        else:
            bin_accs.append(float(correct[mask].mean()))
            bin_confs.append(float(confs[mask].mean()))
            bin_cnts.append(int(mask.sum()))
    ece = float(sum(abs(bin_accs[k] - bin_confs[k]) * bin_cnts[k]
                    for k in range(n_bins)) / max(len(confs), 1))
    return {"bin_accs": bin_accs, "bin_confs": bin_confs, "ece": ece,
            "_bins": bins, "_cnts": bin_cnts}


def _save_reliability_diagram(data: Dict[str, Any], title: str, save_path: str) -> None:
    """Save reliability diagram PNG from binned calibration data."""
    bins = data["_bins"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.bar(bins[:-1], data["bin_accs"], width=1 / (len(bins) - 1), align="edge",
           alpha=0.6, label="Accuracy")
    ax.plot(data["bin_confs"], data["bin_accs"], "ro-", label=f"ECE={data['ece']:.4f}")
    ax.set(xlabel="Confidence", ylabel="Accuracy", title=title)
    ax.legend()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def calibration_plot(
    model:     nn.Module,
    loader:    Any,
    device:    str,
    n_bins:    int = 15,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Reliability diagram (softmax confidence); returns {bin_accs, bin_confs, ece}."""
    model.eval().to(device)
    all_confs, all_correct = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            conf, pred = torch.softmax(model(x), dim=-1).max(dim=-1)
            all_confs.extend(conf.cpu().tolist())
            all_correct.extend((pred == y).long().cpu().tolist())
    data = _bin_calibration(np.array(all_confs), np.array(all_correct), n_bins)
    if save_path:
        _save_reliability_diagram(data, "Reliability Diagram", save_path)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def mc_dropout_calibration_plot(
    mc_wrapper: MCDropoutWrapper,
    loader:     Any,
    device:     str,
    save_path:  Optional[str] = None,
) -> Dict[str, Any]:
    """Reliability diagram using MC Dropout mean probabilities; returns {bin_accs, bin_confs, ece}."""
    mc_wrapper.model.to(device)
    all_confs, all_correct = [], []
    for x, y in loader:
        out       = mc_wrapper(x.to(device))
        conf, pred = out["mean_probs"].max(dim=-1)
        all_confs.extend(conf.cpu().tolist())
        all_correct.extend((pred.cpu() == y).long().tolist())
    data = _bin_calibration(np.array(all_confs), np.array(all_correct))
    if save_path:
        _save_reliability_diagram(data, "MC Dropout Reliability Diagram", save_path)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def entropy_threshold(
    mc_wrapper: MCDropoutWrapper,
    loader:     Any,
    device:     str,
    target_fpr: float = 0.05,
) -> float:
    """Return predictive-entropy threshold tuned to target_fpr on the validation set."""
    mc_wrapper.model.to(device)
    entropies: List[float] = []
    for x, _ in loader:
        entropies.extend(mc_wrapper(x.to(device))["uncertainty"].cpu().tolist())
    arr = np.sort(entropies)
    idx = min(int(np.floor((1.0 - target_fpr) * len(arr))), len(arr) - 1)
    return float(arr[idx])


if __name__ == "__main__":
    print("[Module C] XAI Engine — Grad-CAM / Grad-CAM++ / EigenCAM + LIME + SHAP + MC Dropout")
    print("[Module C] Target: Faithfulness Deletion/Insertion AUC ≥ 0.80")
