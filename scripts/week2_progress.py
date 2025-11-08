"""Quick dataset inspector for Week 2 progress.

Usage:
    python scripts\week2_progress.py

This script counts images in each class folder under `dataset_sample/` and
prints a simple summary. It writes `output_week2_counts.json` in the repo root.
"""
from pathlib import Path
from collections import OrderedDict
import json
import sys


def main():
    repo_root = Path(__file__).resolve().parents[1]
    dataset_dir = repo_root / "dataset_sample"

    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    counts = OrderedDict()
    total = 0

    for class_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
        imgs = [p for p in class_dir.rglob('*') if p.is_file()]
        count = len(imgs)
        counts[class_dir.name] = count
        total += count

    if total == 0:
        print("No images found in dataset_sample/ subfolders.")
        sys.exit(1)

    # Print summary
    for cls, c in counts.items():
        pct = (c / total) * 100 if total else 0
        print(f"{cls}: {c} images ({pct:.1f}%)")

    print(f"Total images: {total}")

    out = {
        "total_images": total,
        "counts": counts,
    }

    out_file = repo_root / "output_week2_counts.json"
    # Convert OrderedDict to normal dict for JSON
    with out_file.open('w', encoding='utf-8') as f:
        json.dump({"total_images": total, "counts": dict(counts)}, f, indent=2)

    print(f"Counts written to {out_file}")


if __name__ == '__main__':
    main()
