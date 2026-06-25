from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from PIL import Image

from obj_det.datasets.models import ImageRecord, ObjectAnnotation

from .base import BaseSourceDataset


logger = logging.getLogger(__name__)


VISDRONE_CATEGORIES = {
    0: "ignored_region",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}


class VisDroneDetSourceDataset(BaseSourceDataset):
    """
    Raw VisDrone DET txt annotations -> canonical ImageRecord adapter.

    Required split path keys:
        images: image directory
        annotations: annotation txt directory
    """

    REQUIRED_PATH_KEYS = ("images", "annotations")

    def _iter_records(self, split: str) -> Iterator[ImageRecord]:
        self.verify_paths(split, keys=self.REQUIRED_PATH_KEYS)

        images_dir = self.path(split, "images")
        annotations_dir = self.path(split, "annotations")

        for image_path in sorted(images_dir.glob("*.jpg")):
            annotation_path = annotations_dir / f"{image_path.stem}.txt"
            if not annotation_path.exists():
                raise FileNotFoundError(
                    f"Missing VisDrone annotation for dataset='{self.key}', "
                    f"split='{split}', image='{image_path}': {annotation_path}"
                )

            with Image.open(image_path) as image:
                width, height = image.size

            objects: list[ObjectAnnotation] = []
            for line_number, line in enumerate(annotation_path.read_text().splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    x, y, w, h, score, category_id, truncation, occlusion = (
                        self._parse_annotation_line(
                            line=line,
                            annotation_path=annotation_path,
                            line_number=line_number,
                        )
                    )
                except ValueError:
                    logger.warning(
                        "Skipping malformed VisDrone annotation: dataset=%s "
                        "split=%s file=%s line=%s text=%s",
                        self.key,
                        split,
                        annotation_path,
                        line_number,
                        line,
                    )
                    continue

                native_label = VISDRONE_CATEGORIES[category_id]

                obj = self.make_object(
                    bbox_xywh=(x, y, w, h),
                    image_width=width,
                    image_height=height,
                    native_label=native_label,
                    native_label_id=category_id,
                    ignore=(score == 0 or category_id in {0, 11}),
                    meta={
                        "score": score,
                        "truncation": truncation,
                        "occlusion": occlusion,
                        "source_line_number": line_number,
                    },
                )
                if obj is not None:
                    objects.append(obj)

            if not objects:
                logger.warning(
                    "VisDrone image has no valid objects after filtering: "
                    "dataset=%s split=%s image=%s annotation=%s",
                    self.key,
                    split,
                    image_path,
                    annotation_path,
                )

            yield self.make_record(
                split=split,
                source_id=image_path.stem,
                image_path=image_path,
                width=width,
                height=height,
                objects=objects,
                meta={
                    "source_file_name": image_path.name,
                    "source_annotation_file": annotation_path.name,
                    "source_annotation_format": "visdrone_det",
                },
            )

    def _parse_annotation_line(
        self,
        *,
        line: str,
        annotation_path: Path,
        line_number: int,
    ) -> tuple[float, float, float, float, int, int, int, int]:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 8:
            raise ValueError(
                f"Malformed VisDrone annotation line in {annotation_path}:{line_number}. "
                f"Expected 8 comma-separated fields, got {len(fields)}: {line!r}"
            )

        try:
            x, y, w, h = map(float, fields[:4])
            score = int(fields[4])
            category_id = int(fields[5])
            truncation = int(fields[6])
            occlusion = int(fields[7])
        except ValueError as exc:
            raise ValueError(
                f"Malformed VisDrone annotation values in {annotation_path}:{line_number}: "
                f"{line!r}"
            ) from exc

        if category_id not in VISDRONE_CATEGORIES:
            raise ValueError(
                f"Unknown VisDrone category_id={category_id} in "
                f"{annotation_path}:{line_number}"
            )

        return x, y, w, h, score, category_id, truncation, occlusion
