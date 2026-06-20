"""Kornia-based IR augmentation for cadet_atr training."""

from __future__ import annotations

import torch
import torch.nn as nn
import kornia.augmentation as K


class IRSyntheticAugmentation(nn.Module):
    """
    Augmentation for synthetic IR images during training.
    Simulates common sensor and atmospheric artefacts seen in real FLIR data.
    """

    def __init__(self, img_size: int = 224):
        super().__init__()
        self.aug = nn.Sequential(
            K.RandomResizedCrop((img_size, img_size), scale=(0.7, 1.0), p=0.6),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.3),
            K.RandomRotation(degrees=180, p=0.5),
            K.RandomBrightness(brightness=(0.6, 1.4), p=0.5),
            K.RandomContrast(contrast=(0.6, 1.4), p=0.5),
            K.RandomGaussianNoise(mean=0.0, std=0.04, p=0.4),
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.5, 2.0), p=0.3),
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.aug(x)


class IRRealAugmentation(nn.Module):
    """
    Lighter augmentation for real (scarce) FLIR images during fine-tuning.
    Avoids aggressive crops that lose contextual IR signatures.
    """

    def __init__(self, img_size: int = 224):
        super().__init__()
        self.aug = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomRotation(degrees=30, p=0.4),
            K.RandomBrightness(brightness=(0.8, 1.2), p=0.4),
            K.RandomContrast(contrast=(0.8, 1.2), p=0.4),
            K.RandomGaussianNoise(mean=0.0, std=0.02, p=0.3),
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.aug(x)
