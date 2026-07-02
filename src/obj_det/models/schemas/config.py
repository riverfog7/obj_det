from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from .base import ModelSchema


LabelMode = Literal["native", "meta"]
BackendName = Literal["hf_trainer", "ultralytics", "torchvision"]
ProtocolName = Literal["controlled", "equal_hpo", "ecosystem"]


def validate_class_list(value: list[str]) -> list[str]:
    cleaned = [item.strip() for item in value]
    if not cleaned or any(not item for item in cleaned):
        raise ValueError("classes must contain non-empty labels")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("classes cannot contain duplicates")
    return cleaned


class ModelConfig(ModelSchema):
    key: str
    backend: BackendName
    model_name_or_path: str | Path
    weights: str | Path | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        value = value.strip().lower()
        if not value:
            raise ValueError("model key cannot be empty")
        return value


class TrainConfig(ModelSchema):
    run_key: str
    classes: list[str]
    label_mode: LabelMode = "meta"
    output_dir: Path
    protocol: ProtocolName = "controlled"
    image_size: int = Field(default=640, gt=0)
    max_epochs: int | None = Field(default=50, gt=0)
    max_steps: int | None = Field(default=None, gt=0)
    effective_batch_size: int = Field(default=16, gt=0)
    per_device_batch_size: int | None = Field(default=None, gt=0)
    gradient_accumulation_steps: int | None = Field(default=None, gt=0)
    seed: int = 0
    augmentation_policy: str = "basic"
    eval_metric: str = "map_50_95"
    eval_every_epochs: int = Field(default=1, gt=0)
    amp: bool = True
    hparams: dict[str, Any] = Field(default_factory=dict)
    backend_params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("run_key")
    @classmethod
    def validate_run_key(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("run_key cannot be empty")
        return value

    @field_validator("classes")
    @classmethod
    def validate_classes(cls, value: list[str]) -> list[str]:
        return validate_class_list(value)


class PredictConfig(ModelSchema):
    classes: list[str]
    label_mode: LabelMode = "meta"
    batch_size: int = Field(default=8, gt=0)
    image_size: int = Field(default=640, gt=0)
    conf_threshold: float = Field(default=0.001, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    augmentation_policy: str = "none"
    backend_params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("classes")
    @classmethod
    def validate_classes(cls, value: list[str]) -> list[str]:
        return validate_class_list(value)


class EvalConfig(ModelSchema):
    classes: list[str]
    label_mode: LabelMode = "meta"
    batch_size: int = Field(default=8, gt=0)
    image_size: int = Field(default=640, gt=0)
    conf_threshold: float = Field(default=0.001, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    compute_per_class: bool = True
    compute_per_condition: bool = True
    compute_per_domain: bool = True
    compute_per_size: bool = True
    primary_metric: str = "map_50_95"
    backend_params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("classes")
    @classmethod
    def validate_classes(cls, value: list[str]) -> list[str]:
        return validate_class_list(value)
