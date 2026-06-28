from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from PIL import Image

from obj_det.datasets.models import ImageRecord, ObjectAnnotation

from .base import BaseSourceDataset


logger = logging.getLogger(__name__)


CLASS_NAMES = {
    1: "Bicycle",
    2: "Boat",
    3: "Bottle",
    4: "Bus",
    5: "Car",
    6: "Cat",
    7: "Chair",
    8: "Cup",
    9: "Dog",
    10: "Motorbike",
    11: "People",
    12: "Table",
}

CLASS_IDS = {name: idx for idx, name in CLASS_NAMES.items()}

LIGHTING = {
    1: "low",
    2: "ambient",
    3: "object",
    4: "single",
    5: "weak",
    6: "strong",
    7: "screen",
    8: "window",
    9: "shadow",
    10: "twilight",
}

INDOOR_OUTDOOR = {
    1: "indoor",
    2: "outdoor",
}

SPLITS = {
    1: "train",
    2: "val",
    3: "test",
}


class ExDarkSourceDataset(BaseSourceDataset):
    REQUIRED_PATH_KEYS = ("images", "annotations", "imageclasslist")

    def _iter_records(self, split: str) -> Iterator[ImageRecord]:
        self.verify_paths(split, keys=self.REQUIRED_PATH_KEYS)

        images_dir = self.path(split, "images")
        annotations_dir = self.path(split, "annotations")

        for (
            file_name,
            image_class_id,
            lighting_id,
            indoor_outdoor_id,
            row_split,
        ) in self._image_list(split):
            if row_split != split:
                continue

            image_class = CLASS_NAMES[image_class_id]
            expected_image_path = images_dir / image_class / file_name
            expected_annotation_path = annotations_dir / image_class / f"{file_name}.txt"

            image_path = self._resolve_file(expected_image_path)
            if image_path is None:
                logger.warning(
                    "Skipping ExDark image with missing file: dataset=%s split=%s image=%s",
                    self.key,
                    split,
                    expected_image_path,
                )
                continue

            annotation_path = self._resolve_file(expected_annotation_path)
            if annotation_path is None:
                logger.warning(
                    "Skipping ExDark image with missing annotation: "
                    "dataset=%s split=%s image=%s annotation=%s",
                    self.key,
                    split,
                    image_path,
                    expected_annotation_path,
                )
                continue

            with Image.open(image_path) as image:
                width, height = image.size

            objects = self._parse_annotations(
                annotation_path=annotation_path,
                image_width=width,
                image_height=height,
            )

            yield self.make_record(
                split=split,
                source_id=f"{image_class}/{file_name}",
                image_path=image_path,
                width=width,
                height=height,
                objects=objects,
                meta={
                    "source_file_name": file_name,
                    "source_annotation_file": str(annotation_path),
                    "source_annotation_format": "exdark",
                    "image_class_id": image_class_id,
                    "image_class": image_class,
                    "lighting_id": lighting_id,
                    "lighting": LIGHTING[lighting_id],
                    "indoor_outdoor_id": indoor_outdoor_id,
                    "indoor_outdoor": INDOOR_OUTDOOR[indoor_outdoor_id],
                },
            )

    def _resolve_file(self, path: Path) -> Path | None:
        if path.exists():
            return path

        if not path.parent.exists():
            return None

        wanted = path.name.lower()
        for candidate in path.parent.iterdir():
            if candidate.name.lower() == wanted:
                return candidate

        return None

    def _image_list(self, split: str) -> list[tuple[str, int, int, int, str]]:
        path = self.path(split, "imageclasslist")
        rows: list[tuple[str, int, int, int, str]] = []

        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            line = line.strip()
            if not line or line.startswith("Name"):
                continue

            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"Malformed ExDark imageclasslist row {path}:{line_number}: {line!r}"
                )

            file_name = parts[0]
            image_class_id = int(parts[1])
            lighting_id = int(parts[2])
            indoor_outdoor_id = int(parts[3])
            split_id = int(parts[4])

            if image_class_id not in CLASS_NAMES:
                raise ValueError(
                    f"Unknown ExDark class id {image_class_id}: {path}:{line_number}"
                )
            if lighting_id not in LIGHTING:
                raise ValueError(
                    f"Unknown ExDark lighting id {lighting_id}: {path}:{line_number}"
                )
            if indoor_outdoor_id not in INDOOR_OUTDOOR:
                raise ValueError(
                    f"Unknown ExDark indoor/outdoor id {indoor_outdoor_id}: {path}:{line_number}"
                )
            if split_id not in SPLITS:
                raise ValueError(f"Unknown ExDark split id {split_id}: {path}:{line_number}")

            rows.append(
                (
                    file_name,
                    image_class_id,
                    lighting_id,
                    indoor_outdoor_id,
                    SPLITS[split_id],
                )
            )

        return rows

    def _parse_annotations(
        self,
        *,
        annotation_path: Path,
        image_width: int,
        image_height: int,
    ) -> list[ObjectAnnotation]:
        objects: list[ObjectAnnotation] = []

        for line_number, line in enumerate(
            annotation_path.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            line = line.strip()
            if not line or line.startswith("%"):
                continue

            parts = line.split()
            if len(parts) < 5:
                logger.warning(
                    "Skipping malformed ExDark annotation row: %s:%s %r",
                    annotation_path,
                    line_number,
                    line,
                )
                continue

            native_label = parts[0]
            if native_label not in CLASS_IDS:
                logger.warning(
                    "Skipping ExDark row with unknown label: %s:%s label=%s",
                    annotation_path,
                    line_number,
                    native_label,
                )
                continue

            try:
                bbox_xywh = [float(value) for value in parts[1:5]]
            except ValueError:
                logger.warning(
                    "Skipping malformed ExDark annotation row: %s:%s %r",
                    annotation_path,
                    line_number,
                    line,
                )
                continue

            obj = self.make_object(
                bbox_xywh=bbox_xywh,
                image_width=image_width,
                image_height=image_height,
                native_label=native_label,
                native_label_id=CLASS_IDS[native_label],
                meta={"source_line_number": line_number},
            )

            if obj is None:
                logger.warning(
                    "Skipping invalid ExDark annotation row: %s:%s bbox=%s",
                    annotation_path,
                    line_number,
                    bbox_xywh,
                )
                continue

            objects.append(obj)

        return objects
