"""Evaluate a saved checkpoint on the test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.training.metrics import (
    classification_metrics,
    confusion_matrix,
    save_confusion_matrix_plot,
    save_metrics_csv,
)
from src.training.modeling import build_transforms, load_checkpoint


def write_report(
    output_path: Path,
    checkpoint_path: Path,
    metrics: dict[str, object],
    class_names: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    per_class = metrics["per_class"]

    lines = [
        "# Model Evaluation Report",
        "",
        "## Checkpoint",
        "",
        f"`{checkpoint_path}`",
        "",
        "## Overall Metrics",
        "",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Macro precision: {metrics['macro_precision']:.4f}",
        f"- Macro recall: {metrics['macro_recall']:.4f}",
        f"- Macro F1: {metrics['macro_f1']:.4f}",
        "",
        "## Per-Class Metrics",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for index, class_name in enumerate(class_names):
        lines.append(
            f"| {class_name} | "
            f"{per_class['precision'][index]:.4f} | "
            f"{per_class['recall'][index]:.4f} | "
            f"{per_class['f1'][index]:.4f} | "
            f"{int(per_class['support'][index])} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "This report is generated from the held-out test split. The test set is",
            "kept separate from training and validation during model selection.",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/garbage_classification"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/best_mobilenet_v2.pth"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/model_report.md"),
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("reports/test_metrics.csv"),
    )
    parser.add_argument(
        "--confusion-matrix",
        type=Path,
        default=Path("reports/figures/confusion_matrix.png"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    class_names = checkpoint["class_names"]
    image_size = int(checkpoint.get("image_size", 224))

    test_dataset = ImageFolder(
        args.data_dir / "test",
        transform=build_transforms(image_size, train=False),
    )
    if test_dataset.classes != class_names:
        raise ValueError(
            "Test dataset classes do not match checkpoint classes: "
            f"{test_dataset.classes} != {class_names}"
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    y_true: list[int] = []
    y_pred: list[int] = []

    model.eval()
    with torch.no_grad():
        for batch_index, (inputs, labels) in enumerate(test_loader, start=1):
            if args.max_batches is not None and batch_index > args.max_batches:
                break
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(predictions.cpu().tolist())

    matrix = confusion_matrix(y_true, y_pred, num_classes=len(class_names))
    metrics = classification_metrics(matrix)
    save_metrics_csv(class_names, metrics, args.metrics_csv)
    save_confusion_matrix_plot(matrix, class_names, args.confusion_matrix)
    write_report(args.report_path, args.checkpoint, metrics, class_names)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro precision: {metrics['macro_precision']:.4f}")
    print(f"Macro recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Wrote: {args.report_path}")
    print(f"Wrote: {args.metrics_csv}")
    print(f"Wrote: {args.confusion_matrix}")


if __name__ == "__main__":
    main()
