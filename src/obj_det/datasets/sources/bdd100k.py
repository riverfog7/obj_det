from __future__ import annotations

import json
import logging
from typing import Any, Iterator

from PIL import Image

from obj_det.datasets.models import ImageRecord, ObjectAnnotation

from .base import BaseSourceDataset


logger = logging.getLogger(__name__)


class Bdd100kSourceDataset(BaseSourceDataset):
    """BDD100K detection labels in Scalabel JSON format."""

    REQUIRED_PATH_KEYS = ("images", "annotations")

    def _iter_records(self, split: str) -> Iterator[ImageRecord]:
        self.verify_paths(split, keys=self.REQUIRED_PATH_KEYS)
        images_dir = self.path(split, "images")
        annotations_path = self.path(split, "annotations")

        with annotations_path.open("r", encoding="utf-8") as file:
            frames = json.load(file)
        if not isinstance(frames, list):
            raise ValueError(f"BDD100K annotations must be a JSON list: {annotations_path}")

        for frame in frames:
            filename = str(frame.get("name", "")).strip()
            if not filename:
                raise ValueError("BDD100K frame has no name")
            image_path = images_dir / filename
            if not image_path.exists():
                raise FileNotFoundError(
                    f"Missing BDD100K image for dataset='{self.key}', "
                    f"split='{split}', file_name='{filename}': {image_path}"
                )

            with Image.open(image_path) as image:
                width, height = image.size

            attributes = frame.get("attributes") or {}
            objects = self._objects(frame.get("labels") or [], width=width, height=height)
            yield self.make_record(
                split=split,
                source_id=filename,
                image_path=image_path,
                width=width,
                height=height,
                objects=objects,
                condition=str(attributes.get("weather") or self.cfg.default_condition),
                domain="road",
                meta={
                    "source_file_name": filename,
                    "source_annotation_format": "bdd100k_scalabel",
                    "scene": attributes.get("scene"),
                    "timeofday": attributes.get("timeofday"),
                    "timestamp": frame.get("timestamp"),
                },
            )

    def _objects(
        self,
        labels: list[dict[str, Any]],
        *,
        width: int,
        height: int,
    ) -> list[ObjectAnnotation]:
        objects: list[ObjectAnnotation] = []
        for label in labels:
            box = label.get("box2d")
            if not box:
                continue

            category = str(label.get("category", "")).strip()
            if not category:
                raise ValueError("BDD100K box annotation has no category")
            x1 = float(box["x1"])
            y1 = float(box["y1"])
            x2 = float(box["x2"])
            y2 = float(box["y2"])

            obj = self.make_object(
                bbox_xywh=(x1, y1, x2 - x1, y2 - y1),
                image_width=width,
                image_height=height,
                native_label=category,
                native_label_id=label.get("id"),
                meta={
                    "attributes": label.get("attributes") or {},
                    "manual_shape": label.get("manualShape"),
                    "manual_attributes": label.get("manualAttributes"),
                },
            )
            if obj is None:
                logger.warning("Skipping invalid BDD100K box for category=%s", category)
                continue
            objects.append(obj)
        return objects
