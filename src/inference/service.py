"""Reusable model service for API inference."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from src.training.modeling import build_transforms, load_checkpoint


class ImageClassifier:
    def __init__(self, checkpoint_path: Path, device: torch.device | None = None) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.checkpoint = load_checkpoint(checkpoint_path, self.device)
        self.class_names: list[str] = self.checkpoint["class_names"]
        self.image_size = int(self.checkpoint.get("image_size", 224))
        self.transform = build_transforms(self.image_size, train=False)

    def predict_path(self, image_path: Path, top_k: int = 3) -> tuple[list[dict[str, float | str]], float]:
        start_time = time.perf_counter()
        try:
            with Image.open(image_path) as image:
                tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Invalid image file: {image_path}") from exc

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)
            values, indices = torch.topk(probabilities, k=min(top_k, len(self.class_names)))

        predictions = [
            {
                "class_name": self.class_names[index.item()],
                "probability": round(float(value.item()), 4),
            }
            for value, index in zip(values, indices)
        ]
        inference_ms = (time.perf_counter() - start_time) * 1000
        return predictions, round(inference_ms, 2)
