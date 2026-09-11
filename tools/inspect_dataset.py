
"""Inspect the class distribution of an image-folder dataset."""

import argparse
from pathlib import Path

from config import DEFAULT_IMAGE_DATASET_DIR, EMOTION_CLASSES


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def count_images(directory):
    return sum(
        1
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def inspect_split(directory):
    total = 0
    print(f"\n{directory}")
    for class_name in EMOTION_CLASSES:
        class_dir = directory / class_name
        count = count_images(class_dir) if class_dir.is_dir() else 0
        total += count
        print(f"  {class_name:8} {count:5}")
    print(f"  {'total':8} {total:5}")
    return total


def main():
    parser = argparse.ArgumentParser(description="统计图像分类数据集")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_IMAGE_DATASET_DIR)
    args = parser.parse_args()

    if not args.dataset_dir.is_dir():
        raise SystemExit(f"数据集目录不存在: {args.dataset_dir}")

    counts = {split: inspect_split(args.dataset_dir / split) for split in ("train", "val")}
    print(f"\nAll images: {sum(counts.values())}")


if __name__ == "__main__":
    main()
