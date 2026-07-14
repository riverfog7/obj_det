from __future__ import annotations

import json
import logging
from typing import Iterator

from PIL import Image

from obj_det.datasets.models import ImageRecord, ObjectAnnotation

from .base import BaseSourceDataset


logger = logging.getLogger(__name__)


class CityscapesSourceDataset(BaseSourceDataset):
    """Cityscapes fine polygon annotations converted to detection boxes."""

    REQUIRED_PATH_KEYS = ("images", "annotations")
    ANNOTATION_SUFFIX = "_gtFine_polygons.json"

    def _iter_records(self, split: str) -> Iterator[ImageRecord]:
        self.verify_paths(split, keys=self.REQUIRED_PATH_KEYS)
        images_dir = self.path(split, "images")
        annotations_dir = self.path(split, "annotations")

        for annotation_path in sorted(annotations_dir.rglob(f"*{self.ANNOTATION_SUFFIX}")):
            relative_path = annotation_path.relative_to(annotations_dir)
            source_id = annotation_path.name.removesuffix(self.ANNOTATION_SUFFIX)
            image_path = (
                images_dir
                / relative_path.parent
                / f"{source_id}_leftImg8bit.png"
            )
            if not image_path.exists():
                raise FileNotFoundError(
                    f"Missing Cityscapes image for dataset='{self.key}', "
                    f"split='{split}', annotation='{annotation_path}': {image_path}"
                )

            with annotation_path.open("r", encoding="utf-8") as file:
                annotation = json.load(file)
            with Image.open(image_path) as image:
                width, height = image.size

            objects = self._objects(annotation.get("objects", []), width=width, height=height)
            yield self.make_record(
                split=split,
                source_id=f"{relative_path.parent}/{source_id}",
                image_path=image_path,
                width=width,
                height=height,
                objects=objects,
                meta={
                    "source_file_name": image_path.name,
                    "source_annotation_file": str(annotation_path),
                    "source_annotation_format": "cityscapes_polygons",
                    "city": relative_path.parent.name,
                },
            )

    def _objects(
        self,
        raw_objects: list[dict],
        *,
        width: int,
        height: int,
    ) -> list[ObjectAnnotation]:
        objects: list[ObjectAnnotation] = []
        for raw_object in raw_objects:
            if raw_object.get("deleted", False):
                continue

            source_label = str(raw_object.get("label", "")).strip()
            if not source_label:
                raise ValueError("Cityscapes object has no label")
            iscrowd = source_label.endswith("group")
            label = source_label.removesuffix("group") if iscrowd else source_label

            polygon = raw_object.get("polygon", [])
            if len(polygon) < 3 or any(len(point) < 2 for point in polygon):
                logger.warning("Skipping malformed Cityscapes polygon for label=%s", source_label)
                continue
            xs = [float(point[0]) for point in polygon]
            ys = [float(point[1]) for point in polygon]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)

            obj = self.make_object(
                bbox_xywh=(xmin, ymin, xmax - xmin, ymax - ymin),
                image_width=width,
                image_height=height,
                native_label=label,
                iscrowd=iscrowd,
                meta={"source_label": source_label},
            )
            if obj is None:
                if label not in self.cfg.ignore_labels:
                    logger.warning(
                        "Skipping invalid Cityscapes polygon for label=%s", source_label
                    )
                continue
            objects.append(obj)
        return objects
