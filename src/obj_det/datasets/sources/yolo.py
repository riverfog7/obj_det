from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import yaml
from PIL import Image

from obj_det.datasets.models import ImageRecord, ObjectAnnotation

from .base import BaseSourceDataset


logger = logging.getLogger(__name__)


class YoloSourceDataset(BaseSourceDataset):
    """
    Raw YOLO detection/segmentation-label dataset -> canonical ImageRecord adapter.

    Expected split path keys:
        images: image directory
        labels: YOLO txt label directory

    Supported label rows:
        class_id cx cy w h
        class_id x1 y1 x2 y2 ... xn yn

    Polygon rows are converted to enclosing detection boxes.
    """

    REQUIRED_PATH_KEYS = ("images", "labels")
    IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, cfg):
        super().__init__(cfg)
        self._class_names: dict[int, str] | None = None

    def _iter_records(self, split: str) -> Iterator[ImageRecord]:
        self.verify_paths(split, keys=self.REQUIRED_PATH_KEYS)

        image_dir = self.path(split, "images")
        label_dir = self.path(split, "labels")
        class_names = self._load_class_names()

        for image_path in sorted(image_dir.iterdir()):
            if (
                not image_path.is_file()
                or image_path.suffix.lower() not in self.IMAGE_SUFFIXES
            ):
                continue

            label_path = label_dir / f"{image_path.stem}.txt"
            with Image.open(image_path) as image:
                width, height = image.size
            objects: list[ObjectAnnotation] = []

            if not label_path.exists():
                logger.warning(
                    "Missing YOLO label file: dataset=%s split=%s image=%s label=%s",
                    self.key,
                    split,
                    image_path,
                    label_path,
                )
            else:
                for line_number, line in enumerate(
                    label_path.read_text(encoding="utf-8", errors="replace").splitlines(),
                    start=1,
                ):
                    line = line.strip()
                    if not line:
                        continue

                    obj = self._line_to_object(
                        line=line,
                        line_number=line_number,
                        label_path=label_path,
                        class_names=class_names,
                        image_width=width,
                        image_height=height,
                    )
                    if obj is not None:
                        objects.append(obj)

            yield self.make_record(
                split=split,
                source_id=image_path.stem,
                image_path=image_path,
                width=width,
                height=height,
                objects=objects,
                meta={
                    "source_file_name": image_path.name,
                    "source_label_file": str(label_path),
                    "source_annotation_format": "yolo",
                },
            )

    def _load_class_names(self) -> dict[int, str]:
        if self._class_names is not None:
            return self._class_names

        data_yaml = self.resolve_path(Path("data.yaml"))
        if not data_yaml.exists():
            raise FileNotFoundError(f"Missing YOLO data.yaml: {data_yaml}")

        data = yaml.safe_load(data_yaml.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"Expected mapping in YOLO data.yaml: {data_yaml}")

        names = data.get("names")
        if isinstance(names, list):
            class_names = {idx: str(name).strip() for idx, name in enumerate(names)}
        elif isinstance(names, dict):
            class_names = {int(idx): str(name).strip() for idx, name in names.items()}
        else:
            raise ValueError(f"YOLO data.yaml must contain list/dict 'names': {data_yaml}")

        if not class_names or any(not name for name in class_names.values()):
            raise ValueError(f"YOLO data.yaml contains empty class names: {data_yaml}")

        nc = data.get("nc")
        if nc is not None and int(nc) != len(class_names):
            raise ValueError(
                f"YOLO data.yaml nc={nc} disagrees with {len(class_names)} names: {data_yaml}"
            )

        self._class_names = class_names
        return class_names

    def _line_to_object(
        self,
        *,
        line: str,
        line_number: int,
        label_path: Path,
        class_names: dict[int, str],
        image_width: int,
        image_height: int,
    ) -> ObjectAnnotation | None:
        parts = line.split()

        try:
            class_id = int(parts[0])
        except (IndexError, ValueError):
            logger.warning(
                "Skipping malformed YOLO label row: %s:%s %r",
                label_path,
                line_number,
                line,
            )
            return None

        native_label = class_names.get(class_id)
        if native_label is None:
            self._increment_import_stat("source_objects_seen")
            self._increment_import_stat("dropped_unknown_labels")
            logger.warning(
                "Skipping YOLO label with unknown class id: %s:%s class_id=%s",
                label_path,
                line_number,
                class_id,
            )
            return None

        try:
            values = [float(value) for value in parts[1:]]
        except ValueError:
            logger.warning(
                "Skipping malformed YOLO label row: %s:%s %r",
                label_path,
                line_number,
                line,
            )
            return None

        source_shape: str
        if len(values) == 4:
            cx, cy, w, h = values
            bbox_xywh = (
                (cx - w / 2.0) * image_width,
                (cy - h / 2.0) * image_height,
                w * image_width,
                h * image_height,
            )
            source_shape = "box"
        elif len(values) > 4 and len(values) % 2 == 0:
            xs = values[0::2]
            ys = values[1::2]
            x1 = min(xs) * image_width
            y1 = min(ys) * image_height
            x2 = max(xs) * image_width
            y2 = max(ys) * image_height
            bbox_xywh = (x1, y1, x2 - x1, y2 - y1)
            source_shape = "polygon"
        else:
            logger.warning(
                "Skipping malformed YOLO label row: %s:%s %r",
                label_path,
                line_number,
                line,
            )
            return None

        obj = self.make_object(
            bbox_xywh=bbox_xywh,
            image_width=image_width,
            image_height=image_height,
            native_label=native_label,
            native_label_id=class_id,
            meta={
                "source_line_number": line_number,
                "source_shape": source_shape,
            },
        )

        if obj is None:
            logger.warning(
                "Skipping invalid YOLO label row: %s:%s class_id=%s bbox=%s",
                label_path,
                line_number,
                class_id,
                bbox_xywh,
            )

        return obj
