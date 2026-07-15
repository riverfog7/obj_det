from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


EXPECTED_IMAGES = 15_000
EXPECTED_ANNOTATIONS = 97_942
CLIP_DURATION_NS = 60_000_000_000
V3_IMAGE_PATTERN = re.compile(
    r"^(?P<timestamp>\d+)_jpg\.rf\.(?![0-9a-f]{32}\.jpg$).+\.jpg$"
)
CATEGORIES = (
    "biker",
    "car",
    "pedestrian",
    "trafficLight",
    "trafficLight-Green",
    "trafficLight-GreenLeft",
    "trafficLight-Red",
    "trafficLight-RedLeft",
    "trafficLight-Yellow",
    "trafficLight-YellowLeft",
    "truck",
)
OUTPUT_DIRS = {"train": "train", "val": "valid", "test": "test"}


def split_for_timestamp(timestamp: int) -> str:
    bucket = (timestamp // CLIP_DURATION_NS) % 10
    if bucket == 0:
        return "val"
    if bucket == 1:
        return "test"
    return "train"


def prepare_udacity(
    root: Path,
    *,
    expected_images: int = EXPECTED_IMAGES,
    expected_annotations: int = EXPECTED_ANNOTATIONS,
) -> dict[str, int]:
    export_dir = root / "data" / "export"
    csv_path = export_dir / "_annotations.csv"
    if not export_dir.is_dir() or not csv_path.is_file():
        raise FileNotFoundError(
            f"Expected Udacity data/export/_annotations.csv under {root}"
        )

    images: dict[str, tuple[Path, int]] = {}
    for image_path in sorted(export_dir.glob("*.jpg")):
        match = V3_IMAGE_PATTERN.fullmatch(image_path.name)
        if match:
            images[image_path.name] = (image_path, int(match.group("timestamp")))

    if len(images) != expected_images:
        raise ValueError(
            f"Expected {expected_images} Udacity v3 images, found {len(images)}"
        )

    annotations_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "filename",
            "width",
            "height",
            "class",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
        }
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Unexpected Udacity CSV columns: {reader.fieldnames}")

        for row in reader:
            filename = row.get("filename", "").strip()
            if not filename or filename not in images:
                continue

            width = int(row["width"])
            height = int(row["height"])
            xmin = int(row["xmin"])
            ymin = int(row["ymin"])
            xmax = int(row["xmax"])
            ymax = int(row["ymax"])
            category = row["class"].strip()
            if category not in CATEGORIES:
                raise ValueError(f"Unknown Udacity category: {category!r}")
            if width <= 0 or height <= 0 or xmax <= xmin or ymax <= ymin:
                raise ValueError(f"Invalid Udacity box for {filename}: {row}")
            if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
                raise ValueError(f"Out-of-bounds Udacity box for {filename}: {row}")

            annotations_by_image[filename].append(
                {
                    "category": category,
                    "width": width,
                    "height": height,
                    "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                }
            )

    annotation_count = sum(len(items) for items in annotations_by_image.values())
    if annotation_count != expected_annotations:
        raise ValueError(
            f"Expected {expected_annotations} Udacity v3 annotations, "
            f"found {annotation_count}"
        )

    category_ids = {name: index for index, name in enumerate(CATEGORIES, start=1)}
    split_images: dict[str, list[dict[str, Any]]] = defaultdict(list)
    split_annotations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    annotation_id = 1

    stage = Path(tempfile.mkdtemp(prefix=".udacity-prepared-", dir=root))
    try:
        for directory in OUTPUT_DIRS.values():
            (stage / directory).mkdir()

        for image_id, (filename, (image_path, timestamp)) in enumerate(
            sorted(images.items()), start=1
        ):
            split = split_for_timestamp(timestamp)
            output_dir = stage / OUTPUT_DIRS[split]
            os.link(image_path, output_dir / filename)

            image_annotations = annotations_by_image.get(filename, [])
            if image_annotations:
                width = image_annotations[0]["width"]
                height = image_annotations[0]["height"]
            else:
                with Image.open(image_path) as image:
                    width, height = image.size
            split_images[split].append(
                {
                    "id": image_id,
                    "file_name": filename,
                    "width": width,
                    "height": height,
                }
            )
            counts[split] += 1

            for item in image_annotations:
                bbox = item["bbox"]
                split_annotations[split].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_ids[item["category"]],
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

        categories = [
            {"id": category_id, "name": name}
            for name, category_id in category_ids.items()
        ]
        for split, directory in OUTPUT_DIRS.items():
            payload = {
                "info": {
                    "source": "Kaggle sshikamaru/udacity-self-driving-car-dataset",
                    "split_policy": "80/10/10 by one-minute timestamp block",
                },
                "images": split_images[split],
                "annotations": split_annotations[split],
                "categories": categories,
            }
            (stage / directory / "_annotations.coco.json").write_text(
                json.dumps(payload, separators=(",", ":")), encoding="utf-8"
            )

        for directory in OUTPUT_DIRS.values():
            destination = root / directory
            shutil.rmtree(destination, ignore_errors=True)
            (stage / directory).replace(destination)
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    return {
        "train": counts["train"],
        "val": counts["val"],
        "test": counts["test"],
        "annotations": annotation_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the Kaggle Udacity v3 TensorFlow export as COCO splits."
    )
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    counts = prepare_udacity(args.root)
    print(
        "Prepared Udacity splits: "
        + ", ".join(f"{name}={count}" for name, count in counts.items())
    )


if __name__ == "__main__":
    main()
