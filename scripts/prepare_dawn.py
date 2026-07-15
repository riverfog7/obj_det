from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
SPLITS = (
    "val",
    "test",
    "train",
    "train",
    "train",
    "train",
    "train",
    "train",
    "train",
    "train",
)


def prepare_dawn(root: Path) -> dict[str, int]:
    image_dir = root / "images"
    label_dir = root / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Expected DAWN images/ and labels/ under {root}")

    groups: dict[str, list[Path]] = defaultdict(list)
    for image_path in sorted(image_dir.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing DAWN label: {label_path}")
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            groups[digest].append(image_path)

    if not groups:
        raise ValueError(f"No DAWN images found in {image_dir}")

    split_root = root / "splits"
    shutil.rmtree(split_root, ignore_errors=True)
    for split in {"train", "val", "test"}:
        (split_root / split / "images").mkdir(parents=True)
        (split_root / split / "labels").mkdir()

    counts: Counter[str] = Counter()
    for index, digest in enumerate(sorted(groups)):
        split = SPLITS[index % len(SPLITS)]
        for image_path in groups[digest]:
            label_path = label_dir / f"{image_path.stem}.txt"
            os.link(image_path, split_root / split / "images" / image_path.name)
            os.link(label_path, split_root / split / "labels" / label_path.name)
            counts[split] += 1

    return {split: counts[split] for split in ("train", "val", "test")}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create deterministic DAWN train/val/test hard-link splits."
    )
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    counts = prepare_dawn(args.root)
    print(
        "Prepared DAWN splits: "
        + ", ".join(f"{split}={count}" for split, count in counts.items())
    )


if __name__ == "__main__":
    main()
