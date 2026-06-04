"""Run Top-K prediction for a single image using a saved checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from src.training.modeling import build_transforms, load_checkpoint


def predict_image(
    image_path: Path,
    checkpoint_path: Path,
    top_k: int = 3,
    device: torch.device | None = None,
) -> list[dict[str, float | str]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    class_names = checkpoint["class_names"]
    image_size = int(checkpoint.get("image_size", 224))
    transform = build_transforms(image_size, train=False)

    with Image.open(image_path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        values, indices = torch.topk(probabilities, k=min(top_k, len(class_names)))

    return [
        {
            "class_name": class_names[index.item()],
            "probability": round(float(value.item()), 4),
        }
        for value, index in zip(values, indices)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/best_mobilenet_v2.pth"),
    )
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    predictions = predict_image(args.image_path, args.checkpoint, args.top_k)
    for prediction in predictions:
        print(f"{prediction['class_name']}: {prediction['probability']:.4f}")


if __name__ == "__main__":
    main()
