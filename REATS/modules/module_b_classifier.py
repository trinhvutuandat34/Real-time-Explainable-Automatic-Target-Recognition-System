"""
MODULE B — ATR Classifier
Heterogeneous 6-architecture Averaging Ensemble: ConvNeXt_tiny, ResNeXt50_32x4d,
ViT_b_16, Swin_T, VGG16, ResNet18 — one model per architecture, not 6 seeds of one.

Training modes:
  FULL (default): 300 epochs, validation from epoch 225 → ~6h/model, ~24h/ensemble on T4
  FAST: 75 epochs, validation from epoch 10 → ~1.5-2h/model, ~12h/ensemble on T4
        Reduces final accuracy by ~1-2% but sufficient for model comparison.
        Enable: CONFIG['enable_fast_train'] = True

Target (full training): Accuracy ≥ 92%, ECE ≤ 0.05
Paper: Do et al. (2025), JKSCI Vol.30 No.1
Hyperparams: AdamW lr=1e-4, batch=128, epochs=300 (or 75 in fast mode), checkpoint from epoch 225 (or 10)
"""

import copy
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import kornia.augmentation as K
from sklearn.metrics import precision_recall_fscore_support

# mlflow is experiment-tracking only — optional. Importing this module (done by
# the dashboard, smoke tests, metrics_report, and inference) must not hard-fail
# when it's absent (e.g. a Kaggle image without it). Fall back to a no-op shim
# so train_full_pipeline's mlflow.* calls run unchanged, just without logging.
try:
    import mlflow
except ImportError:
    import contextlib as _contextlib

    class _NoMlflow:
        def set_experiment(self, *a, **k): pass
        def start_run(self, *a, **k): return _contextlib.nullcontext()
        def log_params(self, *a, **k): pass
        def log_param(self, *a, **k): pass
        def log_metrics(self, *a, **k): pass

    mlflow = _NoMlflow()
    print("[module_b] mlflow not installed — training will run without "
          "experiment logging (pip install mlflow to enable).")

import sys as _sys
from pathlib import Path as _Path
_reats_root = str(_Path(__file__).parent.parent)
if _reats_root not in _sys.path:
    _sys.path.insert(0, _reats_root)
from config import CLASSES, NUM_CLASSES
from modules.augmentation_viewpoint import MultiViewpointAugmentor

CONFIG = {
    "data_root":        "data/",
    "num_classes":      NUM_CLASSES,
    "img_size":         224,
    "batch_size":       128,
    "lr":               1e-4,
    "epochs":           300,
    "best_epoch_start": 225,
    "device":           "cuda" if torch.cuda.is_available() else "cpu",
    "classes":          CLASSES,
    "warmup_epochs":    10,
    "min_lr":           1e-6,
    "weight_decay":     1e-2,
    "grad_clip":        1.0,
    "label_smoothing":  0.1,
    "ema_decay":        0.9999,
    # Fast training mode: set enable_fast_train=True (or pass the config through
    # make_fast_config) to run ~1.5-2h per model instead of ~6h — ~12h for the
    # 6-model ensemble instead of ~24h. Trades ~1-2% accuracy for quota compatibility.
    "enable_fast_train": False,
}


def make_fast_config(cfg: dict) -> dict:
    """Return a copy of `cfg` tuned for a short, quota-friendly run (~1.5-2h/model on T4
    vs ~6h full training). Idempotent — safe to apply more than once.

    Besides cutting epochs, this drops `ema_decay` from 0.9999 to 0.999. That matters:
    the EMA time constant is 1/(1-decay) steps, so 0.9999 (10000 steps) never converges
    inside a 75-epoch (~4300-step) schedule — the EMA weights that get validated *and*
    saved (dashboard loads `ema_state_dict` first) would stay ~65% initialization. 0.999
    (1000-step constant) converges well within the short schedule.
    """
    fast_cfg = cfg.copy()
    fast_cfg.update({
        "enable_fast_train": True,    # so train_full_pipeline picks the fast aug path too
        "epochs":            75,      # reduced from 300
        "best_epoch_start":  10,      # start validation/checkpointing much earlier
        "warmup_epochs":     3,       # faster warmup
        "ema_decay":         0.999,   # converges within the short schedule (see docstring)
    })
    return fast_cfg


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

class _NormalizedEqualize(nn.Module):
    """RandomEqualize for tensors normalized with mean=0.5, std=0.5.
    Kornia's equalize requires [0,1] input, so we denorm→equalize→renorm."""
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self._eq = K.RandomEqualize(p=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x
        x01 = (x * 0.5 + 0.5).clamp(0.0, 1.0)
        x01 = self._eq(x01)
        return (x01 - 0.5) / 0.5


class KorniaAugmentPipeline(nn.Module):
    """IR augmentation from Table 1 of the paper — each transform p=0.5.
    Optionally reduced complexity (fewer transforms, lower probability) for fast training."""
    def __init__(self, full: bool = True):
        super().__init__()
        if full:
            # Full paper augmentation
            self.aug = nn.Sequential(
                K.RandomResizedCrop((224, 224), p=0.5),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
                K.RandomRotation(degrees=180, p=0.5),
                K.RandomAffine(degrees=15, translate=(0.1, 0.1), p=0.5),
                K.RandomPerspective(distortion_scale=0.3, p=0.5),
                K.RandomBrightness(brightness=(0.7, 1.3), p=0.5),
                K.RandomContrast(contrast=(0.7, 1.3), p=0.5),
                _NormalizedEqualize(p=0.5),
                K.RandomGaussianNoise(mean=0.0, std=0.05, p=0.5),
            )
        else:
            # Lightweight aug for fast training — fewer transforms, lower p
            self.aug = nn.Sequential(
                K.RandomResizedCrop((224, 224), p=0.3),
                K.RandomHorizontalFlip(p=0.3),
                K.RandomRotation(degrees=90, p=0.3),
                K.RandomBrightness(brightness=(0.8, 1.2), p=0.3),
                K.RandomContrast(contrast=(0.8, 1.2), p=0.3),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.aug(x)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_convnext(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """ConvNeXt_tiny with ImageNet weights; final Linear replaced for num_classes."""
    weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model   = models.convnext_tiny(weights=weights)
    in_feat = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_feat, num_classes)
    return model


def build_resnext50(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """ResNeXt50_32x4d with ImageNet weights; final Linear replaced for num_classes."""
    weights = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1 if pretrained else None
    model   = models.resnext50_32x4d(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_vit_b_16(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """ViT_b_16 with ImageNet weights; classification head replaced for num_classes."""
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model   = models.vit_b_16(weights=weights)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model


def build_swin_t(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Swin_T with ImageNet weights; final Linear replaced for num_classes."""
    weights = models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
    model   = models.swin_t(weights=weights)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model


def build_vgg16(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """VGG16 with ImageNet weights; final Linear replaced for num_classes."""
    weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    model   = models.vgg16(weights=weights)
    in_feat = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_feat, num_classes)
    return model


def build_resnet18(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """ResNet18 with ImageNet weights; final Linear replaced for num_classes."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model   = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# The 6 heterogeneous architectures required by Do et al. (2025) / professor's spec —
# fuses local CNN features (ResNet/VGG/ResNeXt/ConvNeXt) with global Transformer
# features (ViT/Swin), unlike averaging 6 seeds of a single architecture.
ARCHITECTURES: List[str] = ["convnext_tiny", "resnext50", "vit_b_16", "swin_t", "vgg16", "resnet18"]

_MODEL_BUILDERS = {
    "convnext_tiny": build_convnext,
    "resnext50":     build_resnext50,
    "vit_b_16":      build_vit_b_16,
    "swin_t":        build_swin_t,
    "vgg16":         build_vgg16,
    "resnet18":      build_resnet18,
}


def build_model(arch: str, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Dispatch to the architecture-specific builder — `arch` must be a key of ARCHITECTURES."""
    if arch not in _MODEL_BUILDERS:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from {ARCHITECTURES}")
    return _MODEL_BUILDERS[arch](num_classes=num_classes, pretrained=pretrained)


class EnsembleClassifier(nn.Module):
    """Softmax averaging over N models — architecture-agnostic, so it works for both
    the legacy homogeneous ensemble and the paper's heterogeneous 6-architecture ensemble."""
    def __init__(self, models_list: list):
        super().__init__()
        self.models = nn.ModuleList(models_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = torch.stack(
            [torch.softmax(m(x), dim=-1) for m in self.models], dim=0
        )
        return probs.mean(dim=0)


def preprocess_roi(roi: np.ndarray, img_size: int = 224, device: str = "cpu") -> torch.Tensor:
    """BGR or grayscale uint8 ROI → normalized (1, 3, img_size, img_size) float tensor.

    Matches the eval transform in build_loaders (resize → grayscale×3 → [-1, 1])
    without a PIL round-trip — pure cv2 + tensor ops, so it is cheap enough for the
    per-detection real-time path in Module D.
    """
    import cv2
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    gray = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(gray.astype(np.float32) / 255.0).sub_(0.5).div_(0.5)
    return t.unsqueeze(0).expand(3, -1, -1).contiguous().unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def build_loaders(cfg: dict = CONFIG) -> Tuple[DataLoader, DataLoader, DataLoader]:
    base_tf = transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    root = Path(cfg["data_root"])
    # pin_memory only helps host→CUDA copies; on CPU-only ("base" Kaggle) it is
    # a no-op that just prints a warning, so gate it on CUDA availability.
    # persistent_workers keeps the 4 workers alive across epochs instead of
    # respawning them each epoch — noticeable with the short epochs of fast mode.
    num_workers = 4
    kw   = dict(batch_size=cfg["batch_size"], num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=num_workers > 0)
    return (
        DataLoader(datasets.ImageFolder(root / "train", transform=base_tf), shuffle=True,  **kw),
        DataLoader(datasets.ImageFolder(root / "val",   transform=base_tf), shuffle=False, **kw),
        DataLoader(datasets.ImageFolder(root / "test",  transform=base_tf), shuffle=False, **kw),
    )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing; reduces overconfidence."""
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        return ((1.0 - self.smoothing) * nll + self.smoothing * smooth_loss).mean()


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class ModelEMA:
    """Maintains exponential moving average of model weights."""
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module) -> None:
        """Update shadow weights with current model parameters."""
        with torch.no_grad():
            for s_p, m_p in zip(self.shadow.parameters(), model.parameters()):
                s_p.data.mul_(self.decay).add_(m_p.data, alpha=1.0 - self.decay)

    @property
    def module(self) -> nn.Module:
        """Returns the EMA model."""
        return self.shadow


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    """Linear warmup then cosine decay to min_lr."""
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int) -> None:
        """Update LR for epoch (1-indexed)."""
        if epoch <= self.warmup_epochs:
            scale = epoch / max(self.warmup_epochs, 1)
        else:
            progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = self.min_lr + (base_lr - self.min_lr) * scale


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    aug: nn.Module,
    device: str,
    scaler: Optional[GradScaler] = None,
    ema: Optional[ModelEMA] = None,
    grad_clip: float = 1.0,
) -> Tuple[float, float]:
    """Single training epoch; supports AMP and EMA."""
    model.train()
    total_loss, correct = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        imgs = aug(imgs)
        optimizer.zero_grad()
        if scaler is not None:
            with autocast("cuda"):
                out  = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if ema is not None:
            ema.update(model)
        total_loss += loss.item()
        correct    += (out.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """Evaluate loss and accuracy."""
    model.eval()
    total_loss, correct = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out  = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item()
        correct    += (out.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


def ece_from_arrays(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """ECE from per-sample max-confidence and correctness arrays (vectorised binning)."""
    conf    = np.asarray(conf,    dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    n = len(conf)
    if n == 0:
        return 0.0
    bins = np.linspace(0, 1, n_bins + 1)
    # Match the historical (lo, hi] binning: right-inclusive edges
    bin_idx = np.clip(np.searchsorted(bins, conf, side="left") - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        cnt  = int(mask.sum())
        if cnt == 0:
            continue
        ece += abs(conf[mask].mean() - correct[mask].mean()) * cnt / n
    return float(ece)


def compute_ece(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    n_bins: int = 15,
    is_probs: bool = False,
) -> float:
    """Expected Calibration Error — target ≤ 0.05."""
    all_conf, all_correct = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            probs = out if is_probs else torch.softmax(out, dim=-1)
            conf, pred = probs.max(dim=-1)
            all_conf.append(conf.cpu().numpy())
            all_correct.append((pred == labels).cpu().numpy())

    if not all_conf:
        return 0.0
    return ece_from_arrays(np.concatenate(all_conf), np.concatenate(all_correct), n_bins)


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

class TemperatureScaler(nn.Module):
    """Post-hoc calibration by learning a single temperature T."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x) / self.temperature

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits / T."""
        with torch.no_grad():
            return self.forward(x)

    def fit(self, val_loader: DataLoader, device: str) -> float:
        """Optimise T on val set with NLL; returns final temperature."""
        self.model.eval()
        self.to(device)
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        nll_loss  = nn.CrossEntropyLoss()
        logits_list, labels_list = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits_list.append(self.model(imgs))
                labels_list.append(labels)
        all_logits = torch.cat(logits_list)
        all_labels = torch.cat(labels_list)

        def _eval():
            optimizer.zero_grad()
            loss = nll_loss(all_logits / self.temperature, all_labels)
            loss.backward()
            return loss

        optimizer.step(_eval)
        return float(self.temperature.item())

    def scaled_ece(self, val_loader: DataLoader, device: str, n_bins: int = 15) -> float:
        """ECE after temperature scaling."""
        return compute_ece(self, val_loader, device, n_bins=n_bins, is_probs=False)


# ---------------------------------------------------------------------------
# Per-class metrics
# ---------------------------------------------------------------------------

def per_class_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """Per-class precision, recall, F1 + macro averages."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            out  = model(imgs)
            probs = torch.softmax(out, dim=-1) if out.shape[-1] > 1 else out
            all_preds.extend(probs.argmax(1).cpu().tolist())
            all_labels.extend(labels.tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(CLASSES))), zero_division=0
    )
    metrics: Dict[str, float] = {}
    for i, cls in enumerate(CLASSES):
        metrics[f"{cls}_precision"] = float(precision[i])
        metrics[f"{cls}_recall"]    = float(recall[i])
        metrics[f"{cls}_f1"]        = float(f1[i])
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    metrics["macro_precision"] = float(precision.mean())
    metrics["macro_recall"]    = float(recall.mean())
    metrics["macro_f1"]        = float(f1.mean())
    metrics["accuracy"]        = correct / len(all_labels)
    return metrics


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------

def train_full_pipeline(
    cfg: dict,
    seed: int = 42,
    ckpt_path: str = "checkpoints/convnext_best.pth",
    arch: str = "convnext_tiny",
) -> Tuple[float, str]:
    """Train a single model (architecture `arch`) end-to-end with AMP, EMA, and warmup-cosine LR.

    If cfg['enable_fast_train']=True, uses reduced epochs (75 instead of 300) and earlier
    validation (from epoch 10 instead of 225) to fit within quota (~1.5-2h on T4).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Apply fast-training config overrides if enabled (idempotent)
    if cfg.get("enable_fast_train", False):
        cfg = make_fast_config(cfg)

    device = cfg["device"]
    # Inputs are a fixed 224×224 shape, so let cuDNN auto-tune the fastest conv
    # algorithms once and reuse them — a meaningful speedup on the CNN-heavy
    # architectures (ConvNeXt/ResNeXt/VGG/ResNet), free of any accuracy cost.
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    Path("checkpoints").mkdir(exist_ok=True)

    train_loader, val_loader, _ = build_loaders(cfg)
    model        = build_model(arch, cfg["num_classes"]).to(device)

    # Use lightweight augmentation if fast training is enabled
    use_full_aug = not cfg.get("enable_fast_train", False)
    aug_pipeline = nn.Sequential(
        (MultiViewpointAugmentor() if use_full_aug else nn.Identity()).to(device),
        KorniaAugmentPipeline(full=use_full_aug).to(device),
    )

    criterion    = LabelSmoothingCrossEntropy(cfg.get("label_smoothing", 0.1))
    optimizer    = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-2))
    scaler       = GradScaler("cuda") if device == "cuda" else None
    ema          = ModelEMA(model, decay=cfg.get("ema_decay", 0.9999))
    scheduler    = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=cfg.get("warmup_epochs", 10),
        total_epochs=cfg["epochs"],
        min_lr=cfg.get("min_lr", 1e-6),
    )

    best_val_acc = 0.0
    mode_suffix = "_fast" if cfg.get("enable_fast_train", False) else ""
    mlflow.set_experiment(f"REATS-Baseline{mode_suffix}")
    with mlflow.start_run(run_name=f"{arch}_seed{seed}"):
        mlflow.log_params({k: v for k, v in cfg.items() if not isinstance(v, list)})
        mlflow.log_param("seed", seed)
        mlflow.log_param("arch", arch)
        mlflow.log_param("fast_train", cfg.get("enable_fast_train", False))
        for epoch in range(1, cfg["epochs"] + 1):
            scheduler.step(epoch)
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, aug_pipeline, device,
                scaler=scaler, ema=ema, grad_clip=cfg.get("grad_clip", 1.0),
            )
            mlflow.log_metrics({"train_loss": tr_loss, "train_acc": tr_acc}, step=epoch)

            if epoch >= cfg["best_epoch_start"]:
                val_loss, val_acc = evaluate(ema.module, val_loader, criterion, device)
                mlflow.log_metrics({"val_loss": val_loss, "val_acc": val_acc}, step=epoch)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(
                        {
                            "state_dict":     model.state_dict(),
                            "epoch":          epoch,
                            "best_val_acc":   best_val_acc,
                            "ema_state_dict": ema.module.state_dict(),
                            "arch":           arch,
                        },
                        ckpt_path,
                    )
                    print(f"[Epoch {epoch}] New best: {val_acc:.4f} → {ckpt_path}")

            if epoch % (5 if cfg.get("enable_fast_train", False) else 10) == 0:
                print(f"[Epoch {epoch:3d}] loss={tr_loss:.4f} acc={tr_acc:.4f}")

    return best_val_acc, ckpt_path


def train_ensemble(
    cfg: dict,
    architectures: Optional[List[str]] = None,
    ckpt_dir: str = "checkpoints/",
) -> List[str]:
    """Train one model per architecture in `architectures` (default: the 6 heterogeneous
    architectures from Do et al. 2025 — ConvNeXt_tiny, ResNeXt50, ViT_b_16, Swin_T, VGG16,
    ResNet18), each with a distinct seed. A heterogeneous ensemble fuses local CNN features
    with global Transformer features; it is not 6 seeds of one architecture.

    If cfg['enable_fast_train']=True, trains 6 models in ~12 hours instead of ~24.
    """
    architectures = architectures or ARCHITECTURES
    Path(ckpt_dir).mkdir(exist_ok=True)
    paths: List[str] = []
    fast_note = " (fast mode)" if cfg.get("enable_fast_train", False) else ""
    for i, arch in enumerate(architectures):
        seed = 42 + i
        path = str(Path(ckpt_dir) / f"{arch}_{i}.pth")
        print(f"\n=== Training model {i}/{len(architectures)}: {arch} (seed={seed}){fast_note} ===")
        _, saved = train_full_pipeline(cfg, seed=seed, ckpt_path=path, arch=arch)
        paths.append(saved)
    return paths


def load_ensemble(
    ckpt_paths: List[str],
    num_classes: int = NUM_CLASSES,
    device: str = "cpu",
) -> EnsembleClassifier:
    """Load N checkpoints into EnsembleClassifier. Each checkpoint's architecture is read
    from its saved 'arch' field, so a heterogeneous ensemble loads correctly; checkpoints
    saved before this field existed default to convnext_tiny for backward compatibility."""
    loaded: List[nn.Module] = []
    for path in ckpt_paths:
        ckpt = torch.load(path, map_location=device)
        arch = ckpt.get("arch", "convnext_tiny")
        m = build_model(arch, num_classes, pretrained=False).to(device)
        state = ckpt.get("ema_state_dict", ckpt.get("state_dict", ckpt))
        m.load_state_dict(state)
        m.eval()
        loaded.append(m)
    return EnsembleClassifier(loaded).to(device)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import sys
    device = CONFIG["device"]
    print(f"[Module B] Device: {device}")

    # Check for --fast flag. Just flip the flag on; train_full_pipeline applies
    # make_fast_config internally, so the CLI and the notebook (which sets
    # CONFIG['enable_fast_train']=True) take the exact same code path.
    fast_mode = "--fast" in sys.argv
    if fast_mode:
        print("[Module B] Using FAST training mode (75 epochs, reduced augmentation)")
    cfg = {**CONFIG, "enable_fast_train": True} if fast_mode else CONFIG

    best_val_acc, ckpt_path = train_full_pipeline(cfg)
    status = "PASS" if best_val_acc >= 0.92 else ("CLOSE" if best_val_acc >= 0.90 else "FAIL — consider ensemble/longer training")
    print(f"\n[Module B] Best val accuracy: {best_val_acc:.4f} — {status}")

    _, val_loader, _ = build_loaders(cfg)
    model = build_convnext(cfg["num_classes"], pretrained=False).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("ema_state_dict", ckpt["state_dict"]))
    model.eval()

    metrics = per_class_metrics(model, val_loader, device)
    print("\n[Module B] Per-class metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    ece = compute_ece(model, val_loader, device)
    print(f"\n[Module B] ECE: {ece:.4f} ({'PASS' if ece <= 0.05 else 'FAIL'})")


if __name__ == "__main__":
    main()
