from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


BBoxPolicy = Literal["strict", "clip", "drop"]


class SourceConfigModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )


class SourceSplitConfig(SourceConfigModel):
    """
    Split-level source configuration.

    `paths` is intentionally generic.

    Example keys:
        images
        annotations
        labels
        metadata
        depth
        masks

    The adapter decides which keys it needs.
    """

    paths: dict[str, Path] = Field(default_factory=dict)
    output_split: str | None = None

    condition: str | None = None
    domain: str | None = None
    is_synthetic: bool | None = None

    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("output_split")
    @classmethod
    def validate_output_split(cls, value: str | None) -> str | None:
        if value is None:
            return None

        value = value.strip()
        if not value:
            raise ValueError("output_split cannot be empty")

        return value


class SourceDatasetConfig(SourceConfigModel):
    """
    Dataset-level source configuration.
    """

    key: str
    root: Path
    source_format: str

    splits: dict[str, SourceSplitConfig]

    default_condition: str = "unknown"
    default_domain: str = "unknown"
    default_is_synthetic: bool | None = None

    # Native label -> harmonized label.
    # Missing native labels get meta_label=None.
    class_map: dict[str, str | None] = Field(default_factory=dict)

    # Labels to completely drop during import.
    ignore_labels: set[str] = Field(default_factory=set)

    # YOLO class id -> native label. If absent, YOLO sources read root/data.yaml.
    class_names: dict[int, str] | None = None

    bbox_policy: BBoxPolicy = "strict"

    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        value = value.strip().lower()
        if not value:
            raise ValueError("dataset key cannot be empty")
        return value

    @field_validator("source_format")
    @classmethod
    def validate_source_format(cls, value: str) -> str:
        value = value.strip().lower()
        if not value:
            raise ValueError("source_format cannot be empty")
        return value

    @field_validator("default_condition", "default_domain")
    @classmethod
    def normalize_default_string(cls, value: str) -> str:
        value = value.strip().lower()
        return value if value else "unknown"

    @field_validator("class_map")
    @classmethod
    def normalize_class_map(cls, value: dict[str, str | None]) -> dict[str, str | None]:
        normalized: dict[str, str | None] = {}

        for native_label, meta_label in value.items():
            native_label = native_label.strip()
            if not native_label:
                raise ValueError("class_map labels cannot be empty")

            if meta_label is not None:
                meta_label = meta_label.strip() or None

            normalized[native_label] = meta_label

        return normalized

    @field_validator("ignore_labels")
    @classmethod
    def normalize_ignore_labels(cls, value: set[str]) -> set[str]:
        normalized: set[str] = set()

        for label in value:
            label = label.strip()
            if not label:
                raise ValueError("ignore_labels cannot contain empty labels")
            normalized.add(label)

        return normalized

    @field_validator("class_names", mode="before")
    @classmethod
    def normalize_class_names(cls, value: Any) -> dict[int, str] | None:
        if value is None:
            return None

        if isinstance(value, list):
            items = enumerate(value)
        elif isinstance(value, dict):
            items = value.items()
        else:
            raise TypeError("class_names must be a list or mapping")

        normalized: dict[int, str] = {}
        for raw_id, raw_name in items:
            class_id = int(raw_id)
            if class_id < 0:
                raise ValueError("class_names ids must be non-negative")

            name = str(raw_name).strip()
            if not name:
                raise ValueError("class_names cannot contain empty labels")

            normalized[class_id] = name

        if not normalized:
            raise ValueError("class_names cannot be empty")

        return normalized
