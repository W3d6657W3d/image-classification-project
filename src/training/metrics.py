"""Small metric helpers for classification evaluation."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    num_classes: int,
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label, pred_label] += 1
    return matrix


def classification_metrics(matrix: np.ndarray) -> dict[str, object]:
    true_positive = np.diag(matrix).astype(float)
    false_positive = matrix.sum(axis=0) - true_positive
    false_negative = matrix.sum(axis=1) - true_positive
    support = matrix.sum(axis=1).astype(float)

    precision = np.divide(
        true_positive,
        true_positive + false_positive,
        out=np.zeros_like(true_positive),
        where=(true_positive + false_positive) != 0,
    )
    recall = np.divide(
        true_positive,
        true_positive + false_negative,
        out=np.zeros_like(true_positive),
        where=(true_positive + false_negative) != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    total = matrix.sum()
    accuracy = float(true_positive.sum() / total) if total else 0.0

    return {
        "accuracy": accuracy,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        },
    }


def save_metrics_csv(
    class_names: list[str],
    metrics: dict[str, object],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    per_class = metrics["per_class"]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["class_name", "precision", "recall", "f1", "support"],
        )
        writer.writeheader()
        for index, class_name in enumerate(class_names):
            writer.writerow(
                {
                    "class_name": class_name,
                    "precision": round(float(per_class["precision"][index]), 4),
                    "recall": round(float(per_class["recall"][index]), 4),
                    "f1": round(float(per_class["f1"][index]), 4),
                    "support": int(per_class["support"][index]),
                }
            )


def save_confusion_matrix_plot(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=axis)

    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = matrix.max() / 2 if matrix.size and matrix.max() else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(
                col,
                row,
                int(matrix[row, col]),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
