from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator

from .base import SchemaModel
from .bbox import BBox
from .types import LabelMode


class ObjectAnnotation(SchemaModel):
    """
    One object annotation inside one image.
    """

    bbox: BBox

    # Original dataset label.
    native_label: str
    native_label_id: int | str | None = None

    # Harmonized label for cross-dataset experiments.
    # Example:
    #   pedestrian -> person
    #   people -> person
    #   van -> car
    #
    # If None, this object is ignored in meta-label mode.
    meta_label: str | None = None

    ignore: bool = False
    iscrowd: bool = False

    # Object-specific leftovers:
    # source_annotation_id, occlusion, truncation, track_id, etc.
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("native_label")
    @classmethod
    def validate_native_label(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("native_label cannot be empty")

        return value

    def get_label(self, mode: LabelMode = "native") -> str | None:
        if mode == "native":
            return self.native_label
        if mode == "meta":
            return self.meta_label

        raise ValueError(f"Unknown label mode: {mode}")
