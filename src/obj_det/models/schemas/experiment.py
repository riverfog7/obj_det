from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator

from .base import ModelSchema
from .config import EvalConfig, ModelConfig, PredictConfig, TrainConfig, validate_class_list
from .tuning import SearchSpace, TuningConfig


class DatasetRef(ModelSchema):
    path: Path
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"


class FinalConfig(ModelSchema):
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])
    output_dir: Path | None = None
    evaluate_val: bool = True

    @field_validator("seeds")
    @classmethod
    def validate_seeds(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("final seeds cannot be empty")
        if len(set(value)) != len(value):
            raise ValueError("final seeds cannot contain duplicates")
        return value


class ExperimentConfig(ModelSchema):
    dataset: DatasetRef
    classes: list[str]

    model: ModelConfig | None = None
    model_file: Path | None = None

    train: TrainConfig
    eval: EvalConfig
    predict: PredictConfig | None = None

    tuning: TuningConfig | None = None
    search_space: SearchSpace | None = None
    search_space_file: Path | None = None

    final: FinalConfig = Field(default_factory=FinalConfig)

    @model_validator(mode="before")
    @classmethod
    def fill_shared_config(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        classes = data.get("classes")
        if classes is None:
            return data

        data = dict(data)
        train = dict(data.get("train") or {})
        eval_cfg = dict(data.get("eval") or {})
        predict = dict(data["predict"]) if data.get("predict") is not None else None

        train.setdefault("classes", classes)
        eval_cfg.setdefault("classes", classes)

        if predict is not None:
            predict.setdefault("classes", classes)

        if "label_mode" in train:
            eval_cfg.setdefault("label_mode", train["label_mode"])
            if predict is not None:
                predict.setdefault("label_mode", train["label_mode"])

        if "image_size" in train:
            eval_cfg.setdefault("image_size", train["image_size"])
            if predict is not None:
                predict.setdefault("image_size", train["image_size"])

        data["train"] = train
        data["eval"] = eval_cfg
        if predict is not None:
            data["predict"] = predict

        return data

    @field_validator("classes")
    @classmethod
    def validate_classes(cls, value: list[str]) -> list[str]:
        return validate_class_list(value)

    @model_validator(mode="after")
    def validate_file_choices(self) -> "ExperimentConfig":
        if self.model is not None and self.model_file is not None:
            raise ValueError("Use either model or model_file, not both")
        if self.model is None and self.model_file is None:
            raise ValueError("Either model or model_file is required")
        if self.search_space is not None and self.search_space_file is not None:
            raise ValueError("Use either search_space or search_space_file, not both")
        return self
