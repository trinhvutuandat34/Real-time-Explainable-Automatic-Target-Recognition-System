"""
MODULE B — ATR Classifier
ConvNeXt_tiny + Averaging Ensemble (6 models).
Target: Accuracy ≥ 92%, ECE ≤ 0.05
Paper: Do et al. (2025), JKSCI Vol.30 No.1
Hyperparams: AdamW lr=1e-4, batch=128, epochs=300, checkpoint from epoch 225
"""

import copy
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import kornia.augmentation as K
import mlflow
from sklearn.metrics import precision_recall_fscore_support

CLASSES = ["F16", "LYNX", "MiG19", "MiG21", "PKG", "PTG"]

CONFIG = {
    "data_root":        "data/",
    "num_classes":      6,
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
}


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
    """IR augmentation from Table 1 of the paper — each transform p=0.5."""
    def __init__(self):
        super().__init__()
        self.aug = nn.Sequential(
            K.RandomResizedCrop((224, 224), p=0.5),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomRotation(degrees=30, p=0.5),
            K.RandomAffine(degrees=15, translate=(0.1, 0.1), p=0.5),
            K.RandomPerspective(distortion_scale=0.3, p=0.5),
            K.RandomBrightness(brightness=(0.7, 1.3), p=0.5),
            K.RandomContrast(contrast=(0.7, 1.3), p=0.5),
            _NormalizedEqualize(p=0.5),
            K.RandomGaussianNoise(mean=0.0, std=0.05, p=0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.aug(x)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_convnext(num_classes: int = 6, pretrained: bool = True) -> nn.Module:
    """ConvNeXt_tiny with ImageNet weights; final Linear replaced for num_classes."""
    weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model   = models.convnext_tiny(weights=weights)
    in_feat = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_feat, num_classes)
    return model


class EnsembleClassifier(nn.Module):
    """Softmax averaging over N ConvNeXt models — paper uses 6."""
    def __init__(self, models_list: list):
        super().__init__()
        self.models = nn.ModuleList(models_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = torch.stack(
            [torch.softmax(m(x), dim=-1) for m in self.models], dim=0
        )
        return probs.mean(dim=0)


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
    kw   = dict(batch_size=cfg["batch_size"], num_workers=4, pin_memory=True)
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
            with autocast():
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
            all_conf.extend(conf.cpu().tolist())
            all_correct.extend((pred == labels).cpu().tolist())

    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    n    = len(all_conf)
    for lo, hi in zip(bins[:-1], bins[1:]):
        idx = [i for i, c in enumerate(all_conf) if lo < c <= hi]
        if not idx:
            continue
        avg_conf = np.mean([all_conf[i]    for i in idx])
        avg_acc  = np.mean([all_correct[i] for i in idx])
        ece += abs(avg_conf - avg_acc) * len(idx) / n
    return float(ece)


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
) -> Tuple[float, str]:
    """Train a single ConvNeXt end-to-end with AMP, EMA, and warmup-cosine LR."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = cfg["device"]
    Path("checkpoints").mkdir(exist_ok=True)

    train_loader, val_loader, _ = build_loaders(cfg)
    model        = build_convnext(cfg["num_classes"]).to(device)
    aug_pipeline = KorniaAugmentPipeline().to(device)
    criterion    = LabelSmoothingCrossEntropy(cfg.get("label_smoothing", 0.1))
    optimizer    = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-2))
    scaler       = GradScaler() if device == "cuda" else None
    ema          = ModelEMA(model, decay=cfg.get("ema_decay", 0.9999))
    scheduler    = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=cfg.get("warmup_epochs", 10),
        total_epochs=cfg["epochs"],
        min_lr=cfg.get("min_lr", 1e-6),
    )

    best_val_acc = 0.0
    mlflow.set_experiment("REATS-Baseline")
    with mlflow.start_run(run_name=f"ConvNeXt_seed{seed}"):
        mlflow.log_params({k: v for k, v in cfg.items() if not isinstance(v, list)})
        mlflow.log_param("seed", seed)
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
                        },
                        ckpt_path,
                    )
                    print(f"[Epoch {epoch}] New best: {val_acc:.4f} → {ckpt_path}")

            if epoch % 10 == 0:
                print(f"[Epoch {epoch:3d}] loss={tr_loss:.4f} acc={tr_acc:.4f}")

    return best_val_acc, ckpt_path


def train_ensemble(
    cfg: dict,
    n_models: int = 6,
    ckpt_dir: str = "checkpoints/",
) -> List[str]:
    """Train n_models independently with seeds 42..42+n_models-1."""
    Path(ckpt_dir).mkdir(exist_ok=True)
    paths: List[str] = []
    for i in range(n_models):
        seed = 42 + i
        path = str(Path(ckpt_dir) / f"convnext_{i}.pth")
        print(f"\n=== Training model {i} (seed={seed}) ===")
        _, saved = train_full_pipeline(cfg, seed=seed, ckpt_path=path)
        paths.append(saved)
    return paths


def load_ensemble(
    ckpt_paths: List[str],
    num_classes: int = 6,
    device: str = "cpu",
) -> EnsembleClassifier:
    """Load N checkpoints into EnsembleClassifier."""
    loaded: List[nn.Module] = []
    for path in ckpt_paths:
        m = build_convnext(num_classes, pretrained=False).to(device)
        ckpt = torch.load(path, map_location=device)
        state = ckpt.get("ema_state_dict", ckpt.get("state_dict", ckpt))
        m.load_state_dict(state)
        m.eval()
        loaded.append(m)
    return EnsembleClassifier(loaded).to(device)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    device = CONFIG["device"]
    print(f"[Module B] Device: {device}")

    best_val_acc, ckpt_path = train_full_pipeline(CONFIG)
    status = "PASS" if best_val_acc >= 0.92 else "FAIL — debug needed"
    print(f"\n[Module B] Best val accuracy: {best_val_acc:.4f} — {status}")

    _, val_loader, _ = build_loaders(CONFIG)
    model = build_convnext(CONFIG["num_classes"], pretrained=False).to(device)
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
