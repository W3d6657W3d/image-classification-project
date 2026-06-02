"""Create a balanced train/validation/test split for image classification."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def copy_split(class_name: str, split_name: str, paths: list[Path], output_dir: Path) -> None:
    target_dir = output_dir / split_name / class_name
    target_dir.mkdir(parents=True, exist_ok=True)

    for index, source_path in enumerate(paths, start=1):
        suffix = source_path.suffix.lower()
        target_path = target_dir / f"{class_name}_{index:04d}{suffix}"
        shutil.copy2(source_path, target_path)


def write_split_summary(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["class_name", "selected", "train", "val", "test"],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_split_distribution_plot(rows: list[dict[str, object]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    class_names = [str(row["class_name"]) for row in rows]
    train_counts = [int(row["train"]) for row in rows]
    val_counts = [int(row["val"]) for row in rows]
    test_counts = [int(row["test"]) for row in rows]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_positions = range(len(class_names))

    plt.figure(figsize=(12, 6))
    plt.bar(x_positions, train_counts, label="train", color="#4C78A8")
    plt.bar(x_positions, val_counts, bottom=train_counts, label="val", color="#F58518")
    val_bottom = [train + val for train, val in zip(train_counts, val_counts)]
    plt.bar(x_positions, test_counts, bottom=val_bottom, label="test", color="#54A24B")
    plt.xticks(list(x_positions), class_names, rotation=35, ha="right")
    plt.ylabel("Image count")
    plt.title("Processed Split Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/garbage_classification"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/garbage_classification"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("reports/figures"),
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
    )
    parser.add_argument("--max-per-class", type=int, default=600)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train, validation, and test ratios must sum to 1.0")

    if not args.raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {args.raw_dir}")

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    rng = random.Random(args.seed)
    class_dirs = sorted(path for path in args.raw_dir.iterdir() if path.is_dir())
    rows: list[dict[str, object]] = []

    for class_dir in class_dirs:
        image_paths = list_images(class_dir)
        rng.shuffle(image_paths)
        selected_paths = image_paths[: args.max_per_class]

        selected_count = len(selected_paths)
        train_count = int(selected_count * args.train_ratio)
        val_count = int(selected_count * args.val_ratio)
        test_count = selected_count - train_count - val_count

        train_paths = selected_paths[:train_count]
        val_paths = selected_paths[train_count : train_count + val_count]
        test_paths = selected_paths[train_count + val_count :]

        copy_split(class_dir.name, "train", train_paths, args.output_dir)
        copy_split(class_dir.name, "val", val_paths, args.output_dir)
        copy_split(class_dir.name, "test", test_paths, args.output_dir)

        rows.append(
            {
                "class_name": class_dir.name,
                "selected": selected_count,
                "train": len(train_paths),
                "val": len(val_paths),
                "test": len(test_paths),
            }
        )

    rows = sorted(rows, key=lambda row: str(row["class_name"]))
    write_split_summary(rows, args.reports_dir / "split_summary.csv")
    save_split_distribution_plot(rows, args.figures_dir / "split_distribution.png")

    total_selected = sum(int(row["selected"]) for row in rows)
    total_train = sum(int(row["train"]) for row in rows)
    total_val = sum(int(row["val"]) for row in rows)
    total_test = sum(int(row["test"]) for row in rows)

    print(f"Classes: {len(rows)}")
    print(f"Selected images: {total_selected}")
    print(f"Train images: {total_train}")
    print(f"Validation images: {total_val}")
    print(f"Test images: {total_test}")
    print(f"Wrote: {args.output_dir}")
    print(f"Wrote: {args.reports_dir / 'split_summary.csv'}")
    print(f"Wrote: {args.figures_dir / 'split_distribution.png'}")


if __name__ == "__main__":
    main()
