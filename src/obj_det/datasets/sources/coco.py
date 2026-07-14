from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator

from obj_det.datasets.models import ImageRecord, ObjectAnnotation

from .base import BaseSourceDataset


logger = logging.getLogger(__name__)


class _CocoModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class _CocoImage(_CocoModel):
    id: int | str
    file_name: str
    width: int | None = None
    height: int | None = None


class _CocoAnnotation(_CocoModel):
    image_id: int | str
    category_id: int | str
    bbox: list[float]
    id: int | str | None = None
    area: float | None = None
    iscrowd: bool = False
    ignore: bool = False


class _CocoCategory(_CocoModel):
    id: int | str
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("COCO category name cannot be empty")
        return value


class _CocoFile(_CocoModel):
    images: list[_CocoImage] = Field(default_factory=list)
    annotations: list[_CocoAnnotation] = Field(default_factory=list)
    categories: list[_CocoCategory] = Field(default_factory=list)

    def categories_by_id(self) -> dict[str, str]:
        return {str(category.id): category.name for category in self.categories}

    def annotations_by_image_id(self) -> dict[str, list[_CocoAnnotation]]:
        grouped: dict[str, list[_CocoAnnotation]] = {}
        for annotation in self.annotations:
            grouped.setdefault(str(annotation.image_id), []).append(annotation)
        return grouped


class CocoSourceDataset(BaseSourceDataset):
    """
    Raw COCO-style object detection dataset -> canonical ImageRecord adapter.

    Required split path keys:
        images: image directory
        annotations: COCO JSON annotation file
    """

    REQUIRED_PATH_KEYS = ("images", "annotations")

    def _iter_records(self, split: str) -> Iterator[ImageRecord]:
        self.verify_paths(split, keys=self.REQUIRED_PATH_KEYS)

        images_dir = self.path(split, "images")
        annotations_path = self.path(split, "annotations")

        with annotations_path.open("r", encoding="utf-8") as f:
            coco = _CocoFile.model_validate(json.load(f))

        categories_by_id = coco.categories_by_id()
        annotations_by_image_id = coco.annotations_by_image_id()
        condition_part = self.split_cfg(split).meta.get("condition_from_file_name_part")

        for image_info in coco.images:
            source_image_id = image_info.id
            source_file_name = image_info.file_name
            raw_image_path = Path(source_file_name)
            image_path = (
                raw_image_path
                if raw_image_path.is_absolute()
                else images_dir / raw_image_path
            )

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Missing COCO image for dataset='{self.key}', split='{split}', "
                    f"file_name='{source_file_name}': {image_path}"
                )

            with Image.open(image_path) as image:
                width, height = image.size

            if (
                image_info.width is not None
                and image_info.height is not None
                and image_info.width > 0
                and image_info.height > 0
                and (image_info.width, image_info.height) != (width, height)
            ):
                logger.warning(
                    "Skipping COCO image with mismatched dimensions: "
                    "dataset=%s split=%s image_id=%s file_name=%s "
                    "json_size=%sx%s actual_size=%sx%s",
                    self.key,
                    split,
                    source_image_id,
                    source_file_name,
                    image_info.width,
                    image_info.height,
                    width,
                    height,
                )
                continue

            objects: list[ObjectAnnotation] = []
            for annotation in annotations_by_image_id.get(str(source_image_id), []):
                category_key = str(annotation.category_id)

                if category_key not in categories_by_id:
                    raise KeyError(
                        f"Unknown COCO category_id={annotation.category_id}. "
                        f"Known category ids: {sorted(categories_by_id.keys())}"
                    )

                obj = self.make_object(
                    bbox_xywh=annotation.bbox,
                    image_width=width,
                    image_height=height,
                    native_label=categories_by_id[category_key],
                    native_label_id=annotation.category_id,
                    ignore=annotation.ignore,
                    iscrowd=annotation.iscrowd,
                    meta={
                        "source_annotation_id": annotation.id,
                        "source_area": annotation.area,
                    },
                )
                if obj is None:
                    logger.warning(
                        "Skipping invalid COCO annotation: dataset=%s split=%s "
                        "image_id=%s file_name=%s annotation_id=%s bbox=%s",
                        self.key,
                        split,
                        source_image_id,
                        source_file_name,
                        annotation.id,
                        annotation.bbox,
                    )
                    continue

                objects.append(obj)

            condition = None
            if condition_part is not None:
                try:
                    condition = raw_image_path.parts[int(condition_part)]
                except (IndexError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Cannot derive condition from COCO file_name={source_file_name!r} "
                        f"using path part {condition_part!r}"
                    ) from exc

            yield self.make_record(
                split=split,
                source_id=source_image_id,
                image_path=image_path,
                width=width,
                height=height,
                objects=objects,
                condition=condition,
                meta={
                    "source_image_id": source_image_id,
                    "source_file_name": source_file_name,
                    "source_annotation_format": "coco",
                },
            )
