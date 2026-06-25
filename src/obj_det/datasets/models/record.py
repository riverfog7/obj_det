from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from .annotation import ObjectAnnotation
from .base import SchemaModel
from .types import LabelMode


class ImageRecord(SchemaModel):
    """
    One canonical dataset row.

    One ImageRecord = one image + all object annotations for that image.
    """

    image_id: str
    dataset: str
    split: str

    image_path: Path
    width: int = Field(gt=0)
    height: int = Field(gt=0)

    objects: list[ObjectAnnotation] = Field(default_factory=list)

    condition: str = "unknown"  # haze, fog, rain, snow, clear, low_light, etc.
    domain: str = "unknown"  # aerial, road, general, remote_sensing, etc.
    is_synthetic: bool | None = None

    # Image-level leftovers:
    # source_image_id, source_file_name, condition_level, sensor, tags, sequence_id, etc.
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("image_id", "dataset", "split")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("field cannot be empty")

        return value

    @field_validator("condition", "domain")
    @classmethod
    def normalize_metadata_string(cls, value: str) -> str:
        value = value.strip().lower()

        return value if value else "unknown"

    def valid_objects(
        self,
        mode: LabelMode = "native",
        include_crowd: bool = True,
    ) -> list[ObjectAnnotation]:
        valid: list[ObjectAnnotation] = []

        for obj in self.objects:
            if obj.ignore:
                continue

            if obj.iscrowd and not include_crowd:
                continue

            if obj.get_label(mode) is None:
                continue

            valid.append(obj)

        return valid

    def labels(self, mode: LabelMode = "native") -> list[str]:
        labels: list[str] = []

        for obj in self.valid_objects(mode=mode):
            label = obj.get_label(mode)
            if label is not None:
                labels.append(label)

        return labels

    def assert_valid_geometry(self) -> None:
        issues: list[str] = []

        for idx, obj in enumerate(self.objects):
            if not obj.bbox.within_image(self.width, self.height):
                issues.append(
                    f"object[{idx}] bbox={obj.bbox.xywh()} exceeds image bounds "
                    f"{self.width}x{self.height}"
                )

        if issues:
            joined = "\n".join(issues)
            raise ValueError(f"Invalid geometry for image_id={self.image_id}:\n{joined}")
