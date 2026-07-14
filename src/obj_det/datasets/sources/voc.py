from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

from PIL import Image

from obj_det.datasets.models import ImageRecord, ObjectAnnotation

from .base import BaseSourceDataset


logger = logging.getLogger(__name__)


class PascalVocSourceDataset(BaseSourceDataset):
    """Pascal VOC XML annotations to canonical image records."""

    REQUIRED_PATH_KEYS = ("images", "annotations", "image_set")

    def _iter_records(self, split: str) -> Iterator[ImageRecord]:
        self.verify_paths(split, keys=self.REQUIRED_PATH_KEYS)

        images_dir = self.path(split, "images")
        annotations_dir = self.path(split, "annotations")
        image_set_path = self.path(split, "image_set")

        for source_id in self._image_ids(image_set_path):
            annotation_path = annotations_dir / f"{source_id}.xml"
            if not annotation_path.exists():
                raise FileNotFoundError(
                    f"Missing Pascal VOC annotation for dataset='{self.key}', "
                    f"split='{split}', image_id='{source_id}': {annotation_path}"
                )

            root = ET.parse(annotation_path).getroot()
            filename = root.findtext("filename") or f"{source_id}.jpg"
            image_path = images_dir / filename
            if not image_path.exists():
                raise FileNotFoundError(
                    f"Missing Pascal VOC image for dataset='{self.key}', "
                    f"split='{split}', image_id='{source_id}': {image_path}"
                )

            with Image.open(image_path) as image:
                width, height = image.size

            objects = self._objects(root, width=width, height=height)
            yield self.make_record(
                split=split,
                source_id=source_id,
                image_path=image_path,
                width=width,
                height=height,
                objects=objects,
                meta={
                    "source_file_name": filename,
                    "source_annotation_file": str(annotation_path),
                    "source_annotation_format": "pascal_voc",
                },
            )

    @staticmethod
    def _image_ids(path: Path) -> list[str]:
        return [line.strip().split()[0] for line in path.read_text().splitlines() if line.strip()]

    def _objects(self, root: ET.Element, *, width: int, height: int) -> list[ObjectAnnotation]:
        objects: list[ObjectAnnotation] = []
        for element in root.findall("object"):
            label = self._required_text(element, "name")
            box = element.find("bndbox")
            if box is None:
                raise ValueError(f"Pascal VOC object '{label}' has no bndbox")

            xmin = float(self._required_text(box, "xmin"))
            ymin = float(self._required_text(box, "ymin"))
            xmax = float(self._required_text(box, "xmax"))
            ymax = float(self._required_text(box, "ymax"))
            difficult = int(element.findtext("difficult", "0")) != 0

            obj = self.make_object(
                bbox_xywh=(xmin - 1.0, ymin - 1.0, xmax - xmin + 1.0, ymax - ymin + 1.0),
                image_width=width,
                image_height=height,
                native_label=label,
                ignore=difficult,
                meta={
                    "difficult": difficult,
                    "truncated": int(element.findtext("truncated", "0")) != 0,
                    "pose": element.findtext("pose", "Unspecified"),
                },
            )
            if obj is None:
                logger.warning("Skipping invalid Pascal VOC box for label=%s", label)
                continue
            objects.append(obj)
        return objects

    @staticmethod
    def _required_text(element: ET.Element, name: str) -> str:
        value = element.findtext(name)
        if value is None or not value.strip():
            raise ValueError(f"Pascal VOC element is missing required field '{name}'")
        return value.strip()
