"""Model and transform helpers shared by training, evaluation, and inference.

This module keeps model construction and preprocessing in one place so that
training, offline evaluation, CLI prediction, and the FastAPI service all use
the same image size, normalization, class order, and checkpoint format.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    """Build the preprocessing pipeline for either training or evaluation.

    Random augmentation is applied only when ``train=True``. Validation, test,
    and inference use deterministic preprocessing so reported metrics and API
    predictions are repeatable.
    """
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                # Lightweight augmentation improves robustness without changing
                # the label semantics of common garbage-object photos.
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create a MobileNetV2 classifier with a project-specific output layer.

    The ImageNet-pretrained backbone provides reusable visual features, while
    replacing the final linear layer adapts the network to the 12 garbage
    categories used by this project.
    """
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def get_class_names(data_dir: Path) -> list[str]:
    """Read class names from the processed training directory."""
    dataset = ImageFolder(data_dir / "train")
    return dataset.classes


def save_checkpoint(
    output_path: Path,
    model: nn.Module,
    class_names: list[str],
    image_size: int,
    metrics: dict[str, float],
) -> None:
    """Persist model weights and metadata needed for later inference."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "image_size": image_size,
            "architecture": "mobilenet_v2",
            "metrics": metrics,
        },
        output_path,
    )


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    """Load a saved checkpoint and return a ready-to-evaluate model."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    model = build_model(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint
