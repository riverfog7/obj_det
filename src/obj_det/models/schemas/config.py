from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from .base import ModelSchema


LabelMode = Literal["native", "meta"]
BackendName = Literal["hf_trainer", "ultralytics", "torchvision"]
ProtocolName = Literal["controlled", "equal_hpo"]


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


class PreprocessConfig(ModelSchema):
    image_size: int = Field(gt=0)


class AugmentationConfig(ModelSchema):
    policy: Literal["none", "basic", "weather", "provider"] = "none"
    horizontal_flip_p: float = Field(default=0.0, ge=0.0, le=1.0)
    color_jitter_strength: float = Field(default=0.0, ge=0.0)
    color_jitter_p: float = Field(default=0.0, ge=0.0, le=1.0)


class DataLoaderConfig(ModelSchema):
    num_workers: int = Field(default=0, ge=0)
    pin_memory: bool = True
    persistent_workers: bool | None = None
    prefetch_factor: int | None = Field(default=None, gt=0)
    predecode_images: bool = False
    include_samples_in_batch: bool = False
    decode_backend: Literal["pil", "opencv"] = "pil"
    profile_every_n: int | None = Field(default=None, gt=0)


class EvalStrategyConfig(ModelSchema):
    enabled: bool = False
    every_epochs: int = Field(default=1, gt=0)


class OptimizerConfig(ModelSchema):
    name: Literal["adamw"] = "adamw"
    weight_decay: float = Field(default=1.0e-4, ge=0.0)
    beta1: float = Field(default=0.9, ge=0.0, lt=1.0)
    beta2: float = Field(default=0.999, ge=0.0, lt=1.0)
    epsilon: float = Field(default=1.0e-8, gt=0.0)


class SchedulerConfig(ModelSchema):
    name: Literal["warmup_cosine"] = "warmup_cosine"
    warmup_epochs: float = Field(default=1.0, ge=0.0)
    total_epochs: int = Field(default=50, gt=0)
    min_lr_ratio: float = Field(default=0.01, ge=0.0, le=1.0)


class CheckpointConfig(ModelSchema):
    save_every_epochs: int = Field(default=1, gt=0)
    save_best: bool = True
    save_last: bool = True
    keep_all_epoch_checkpoints: bool = True


class EarlyStoppingConfig(ModelSchema):
    enabled: bool = True
    metric: str = "map_50_95"
    mode: Literal["max"] = "max"
    min_epochs: int = Field(default=10, gt=0)
    patience: int = Field(default=8, gt=0)
    min_delta: float = Field(default=0.001, ge=0.0)
    restore_best: bool = True


class TrainConfig(ModelSchema):
    run_key: str
    classes: list[str]
    label_mode: LabelMode = "meta"
    output_dir: Path
    protocol: ProtocolName = "controlled"
    preprocess: PreprocessConfig
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    loader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)
    eval_strategy: EvalStrategyConfig = Field(default_factory=EvalStrategyConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    max_epochs: int | None = Field(default=50, gt=0)
    max_steps: int | None = Field(default=None, gt=0)
    batch_size: int = Field(default=16, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    logging_steps: int = Field(default=100, gt=0)
    seed: int = 0
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
    preprocess: PreprocessConfig
    conf_threshold: float = Field(default=0.001, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_detections_per_image: int = Field(default=300, gt=0)
    backend_params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("classes")
    @classmethod
    def validate_classes(cls, value: list[str]) -> list[str]:
        return validate_class_list(value)


class EvalConfig(ModelSchema):
    classes: list[str]
    label_mode: LabelMode = "meta"
    batch_size: int = Field(default=8, gt=0)
    preprocess: PreprocessConfig
    conf_threshold: float = Field(default=0.001, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_detections_per_image: int = Field(default=300, gt=0)
    compute_per_class: bool = False
    compute_per_condition: bool = False
    compute_per_domain: bool = False
    compute_per_size: bool = False
    primary_metric: str = "map_50_95"
    backend_params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("classes")
    @classmethod
    def validate_classes(cls, value: list[str]) -> list[str]:
        return validate_class_list(value)
