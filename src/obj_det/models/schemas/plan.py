from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, PrivateAttr, field_validator, model_validator

from .base import ModelSchema
from .config import LabelMode, ProtocolName, validate_class_list


class DatasetRefConfig(ModelSchema):
    key: str
    path: Path
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    default_class_space: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        value = value.strip().lower()
        if not value:
            raise ValueError("dataset key cannot be empty")
        return value


class ClassSpaceConfig(ModelSchema):
    classes: list[str]
    label_mode: LabelMode = "meta"

    @field_validator("classes")
    @classmethod
    def validate_classes(cls, value: list[str]) -> list[str]:
        return validate_class_list(value)


class RecipeConfig(ModelSchema):
    protocol: ProtocolName
    augmentation_file: Path | None = None
    search_space_file: Path | None = None
    train: dict[str, Any] = Field(default_factory=dict)
    eval: dict[str, Any] = Field(default_factory=dict)
    predict: dict[str, Any] = Field(default_factory=dict)
    tuning: dict[str, Any] = Field(default_factory=dict)
    final: dict[str, Any] = Field(default_factory=dict)
    logging: dict[str, Any] = Field(default_factory=dict)


class ModelGroupConfig(ModelSchema):
    models: list[Path]

    @field_validator("models")
    @classmethod
    def validate_models(cls, value: list[Path]) -> list[Path]:
        if not value:
            raise ValueError("model group cannot be empty")
        return value


class RunTemplateConfig(ModelSchema):
    run_key: str = "{model_key}_{dataset_key}_{protocol}"
    output_dir: str = "runs/{model_key}/{dataset_key}/{protocol}"
    tuning_study_name: str = "{model_key}_{dataset_key}_{protocol}"
    tuning_output_dir: str = "runs/hpo/{model_key}_{dataset_key}_{protocol}"
    final_output_dir: str = "runs/{model_key}/{dataset_key}/{protocol}/final"
    wandb_project: str = "{dataset_key}_{protocol}"


class ExperimentPlanConfig(ModelSchema):
    key: str
    dataset_file: Path
    class_space_file: Path
    recipe_file: Path
    model_files: list[Path] = Field(default_factory=list)
    model_group_file: Path | None = None
    run_template: RunTemplateConfig = Field(default_factory=RunTemplateConfig)
    backend_defaults: dict[str, dict[str, Any]] = Field(default_factory=dict)
    model_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    _base_dir: Path = PrivateAttr(default=Path("."))

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        value = value.strip().lower()
        if not value:
            raise ValueError("plan key cannot be empty")
        return value

    @model_validator(mode="after")
    def validate_model_sources(self) -> "ExperimentPlanConfig":
        if not self.model_files and self.model_group_file is None:
            raise ValueError("plan must define model_files, model_group_file, or both")
        return self
