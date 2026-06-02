"""Analyze raw image classification data and create basic EDA figures."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def check_image(path: Path) -> tuple[bool, tuple[int, int] | None]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return True, image.size
    except (OSError, UnidentifiedImageError):
        return False, None


def write_distribution_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "class_name",
                "image_count",
                "valid_count",
                "invalid_count",
                "percentage",
                "min_width",
                "min_height",
                "max_width",
                "max_height",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_distribution_plot(rows: list[dict[str, object]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    class_names = [str(row["class_name"]) for row in rows]
    counts = [int(row["image_count"]) for row in rows]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, counts, color="#4C78A8")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Image count")
    plt.title("Raw Class Distribution")
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_sample_grid(
    class_to_images: dict[str, list[Path]],
    output_path: Path,
    samples_per_class: int,
    seed: int,
) -> None:
    import matplotlib.pyplot as plt

    rng = random.Random(seed)
    class_names = sorted(class_to_images)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        len(class_names),
        samples_per_class,
        figsize=(samples_per_class * 2.0, len(class_names) * 1.8),
    )

    for row_index, class_name in enumerate(class_names):
        images = class_to_images[class_name]
        sample_paths = rng.sample(images, k=min(samples_per_class, len(images)))
        for col_index in range(samples_per_class):
            axis = axes[row_index][col_index]
            axis.axis("off")
            if col_index >= len(sample_paths):
                continue
            with Image.open(sample_paths[col_index]) as image:
                axis.imshow(image.convert("RGB"))
            if col_index == 0:
                axis.set_ylabel(class_name, rotation=0, ha="right", va="center")

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
        "--figures-dir",
        type=Path,
        default=Path("reports/figures"),
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
    )
    parser.add_argument("--samples-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {args.raw_dir}")

    class_dirs = sorted(path for path in args.raw_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class directories found in: {args.raw_dir}")

    class_to_images: dict[str, list[Path]] = {}
    rows: list[dict[str, object]] = []

    for class_dir in class_dirs:
        image_paths = list_images(class_dir)
        class_to_images[class_dir.name] = image_paths

    total_images = sum(len(paths) for paths in class_to_images.values())
    if total_images == 0:
        raise ValueError("No image files found.")

    for class_name, image_paths in class_to_images.items():
        valid_count = 0
        invalid_count = 0
        widths: list[int] = []
        heights: list[int] = []

        for image_path in image_paths:
            is_valid, size = check_image(image_path)
            if is_valid and size is not None:
                valid_count += 1
                widths.append(size[0])
                heights.append(size[1])
            else:
                invalid_count += 1

        rows.append(
            {
                "class_name": class_name,
                "image_count": len(image_paths),
                "valid_count": valid_count,
                "invalid_count": invalid_count,
                "percentage": round(len(image_paths) / total_images * 100, 2),
                "min_width": min(widths) if widths else "",
                "min_height": min(heights) if heights else "",
                "max_width": max(widths) if widths else "",
                "max_height": max(heights) if heights else "",
            }
        )

    rows = sorted(rows, key=lambda row: str(row["class_name"]))

    write_distribution_csv(rows, args.reports_dir / "class_distribution.csv")
    save_distribution_plot(rows, args.figures_dir / "class_distribution.png")
    save_sample_grid(
        class_to_images,
        args.figures_dir / "sample_grid.png",
        args.samples_per_class,
        args.seed,
    )

    largest = max(int(row["image_count"]) for row in rows)
    smallest = min(int(row["image_count"]) for row in rows)
    imbalance_ratio = largest / smallest

    print(f"Classes: {len(rows)}")
    print(f"Total images: {total_images}")
    print(f"Smallest class: {smallest}")
    print(f"Largest class: {largest}")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}")
    print(f"Wrote: {args.reports_dir / 'class_distribution.csv'}")
    print(f"Wrote: {args.figures_dir / 'class_distribution.png'}")
    print(f"Wrote: {args.figures_dir / 'sample_grid.png'}")


if __name__ == "__main__":
    main()
