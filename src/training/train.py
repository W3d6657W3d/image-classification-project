"""Train a MobileNetV2 baseline on the processed garbage dataset.

The script uses the balanced processed split, applies augmentation only to the
training loader, tracks validation metrics each epoch, and saves the checkpoint
with the best validation macro F1 score.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.training.metrics import classification_metrics, confusion_matrix
from src.training.modeling import (
    build_model,
    build_transforms,
    save_checkpoint,
)


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> tuple[float, dict[str, float]]:
    """Run one training or evaluation epoch and return loss plus metrics.

    Passing an optimizer enables gradient updates; passing ``None`` switches the
    model into evaluation mode and disables backward propagation. The same
    metric path is used for train and validation so the numbers are comparable.
    """
    is_training = optimizer is not None
    model.train(is_training)

    running_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    for batch_index, (inputs, labels) in enumerate(dataloader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break

        inputs = inputs.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            if is_training:
                loss.backward()
                optimizer.step()

        # Weight batch loss by batch size so a smaller final batch does not
        # distort the epoch average.
        running_loss += loss.item() * inputs.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())

    matrix = confusion_matrix(y_true, y_pred, num_classes=len(dataloader.dataset.classes))
    metrics = classification_metrics(matrix)
    average_loss = running_loss / max(len(y_true), 1)

    return average_loss, {
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
    }


def main() -> None:
    """Parse training arguments, train the model, and write artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/garbage_classification"),
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("models/best_mobilenet_v2.pth"),
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=Path("reports/training_history.json"),
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = ImageFolder(
        args.data_dir / "train",
        transform=build_transforms(args.image_size, train=True),
    )
    val_dataset = ImageFolder(
        args.data_dir / "val",
        transform=build_transforms(args.image_size, train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(
        num_classes=len(train_dataset.classes),
        pretrained=not args.no_pretrained,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_f1 = -1.0
    history: list[dict[str, object]] = []
    start_time = time.time()

    print(f"Device: {device}")
    print(f"Classes: {train_dataset.classes}")
    print(f"Train images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
        )
        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            max_batches=args.max_val_batches,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)

        print(
            "Epoch "
            f"{epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_f1={train_metrics['macro_f1']:.4f} | "
            f"val_loss={val_loss:.4f} val_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            # Model selection is based on the validation split only. The test
            # split stays untouched until the separate evaluation script runs.
            save_checkpoint(
                args.output_model,
                model,
                train_dataset.classes,
                args.image_size,
                {
                    "best_val_macro_f1": best_val_f1,
                    "best_val_accuracy": val_metrics["accuracy"],
                    "epoch": epoch,
                },
            )
            print(f"Saved best model: {args.output_model}")

    args.history_path.parent.mkdir(parents=True, exist_ok=True)
    args.history_path.write_text(
        json.dumps(
            {
                "device": str(device),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "image_size": args.image_size,
                "learning_rate": args.learning_rate,
                "pretrained": not args.no_pretrained,
                "elapsed_seconds": round(time.time() - start_time, 2),
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote training history: {args.history_path}")


if __name__ == "__main__":
    main()
