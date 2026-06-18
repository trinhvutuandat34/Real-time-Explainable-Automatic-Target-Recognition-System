"""
MODULE B — ATR Classifier
ConvNeXt_tiny + Averaging Ensemble (6 models).
Target: Accuracy ≥ 92%, ECE ≤ 0.05
Paper: Do et al. (2025), JKSCI Vol.30 No.1
Hyperparams: AdamW lr=1e-4, batch=128, epochs=300, checkpoint from epoch 225
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import kornia.augmentation as K
import mlflow
from pathlib import Path

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
}


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
            K.RandomEqualize(p=0.5),
            K.RandomGaussianNoise(mean=0.0, std=0.05, p=0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.aug(x)


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


def build_loaders(cfg: dict = CONFIG):
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


def train_one_epoch(model, loader, optimizer, criterion, aug, device):
    model.train()
    total_loss, correct = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        imgs = aug(imgs)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (out.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out  = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item()
        correct    += (out.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


def compute_ece(model, loader, device, n_bins: int = 15) -> float:
    """Expected Calibration Error — target ≤ 0.05."""
    all_conf, all_correct = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            probs = torch.softmax(model(imgs), dim=-1)
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


def main():
    device = CONFIG["device"]
    print(f"[Module B] Device: {device}")

    train_loader, val_loader, _ = build_loaders(CONFIG)
    model        = build_convnext(CONFIG["num_classes"]).to(device)
    optimizer    = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    criterion    = nn.CrossEntropyLoss()
    aug_pipeline = KorniaAugmentPipeline().to(device)
    best_val_acc = 0.0
    Path("checkpoints").mkdir(exist_ok=True)

    mlflow.set_experiment("REATS-Baseline")
    with mlflow.start_run(run_name="ConvNeXt_tiny_baseline"):
        mlflow.log_params(CONFIG)
        for epoch in range(1, CONFIG["epochs"] + 1):
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, aug_pipeline, device)
            mlflow.log_metrics({"train_loss": tr_loss, "train_acc": tr_acc}, step=epoch)

            if epoch >= CONFIG["best_epoch_start"]:
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                mlflow.log_metrics({"val_loss": val_loss, "val_acc": val_acc}, step=epoch)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), "checkpoints/convnext_best.pth")
                    print(f"[Epoch {epoch}] New best: {val_acc:.4f}")

            if epoch % 10 == 0:
                print(f"[Epoch {epoch:3d}] loss={tr_loss:.4f} acc={tr_acc:.4f}")

    status = "PASS" if best_val_acc >= 0.90 else "FAIL — debug needed"
    print(f"\n[Module B] Best val accuracy: {best_val_acc:.4f} — {status}")


if __name__ == "__main__":
    main()
