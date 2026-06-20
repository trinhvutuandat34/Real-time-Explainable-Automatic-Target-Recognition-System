"""ConvNeXt model factory for cadet_atr."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from utils.config import NUM_CLASSES


def build_model(
    model_name: str = "convnext_tiny",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """Return a ConvNeXt model with the final linear replaced for num_classes."""
    weights_map = {
        "convnext_tiny":   models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
        "convnext_small":  models.ConvNeXt_Small_Weights.IMAGENET1K_V1,
        "convnext_base":   models.ConvNeXt_Base_Weights.IMAGENET1K_V1,
    }
    builder_map = {
        "convnext_tiny":  models.convnext_tiny,
        "convnext_small": models.convnext_small,
        "convnext_base":  models.convnext_base,
    }
    if model_name not in builder_map:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(builder_map)}")

    weights = weights_map[model_name] if pretrained else None
    model   = builder_map[model_name](weights=weights)
    in_feat = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_feat, num_classes)
    return model


def get_backbone(model: nn.Module) -> nn.Module:
    """Return the feature-extraction backbone (everything except the classifier)."""
    return model.features


def get_feature_dim(model: nn.Module) -> int:
    """Return the feature dimensionality before the classifier head."""
    return model.classifier[2].in_features
