from __future__ import annotations

from typing import Any

from pydantic import Field

from .base import ModelSchema


class EvalResult(ModelSchema):
    model_key: str
    dataset_key: str | None = None
    split: str | None = None
    primary_metric: str
    primary_metric_value: float
    metrics: dict[str, float] = Field(default_factory=dict)
    per_class: dict[str, dict[str, float]] = Field(default_factory=dict)
    per_condition: dict[str, dict[str, float]] = Field(default_factory=dict)
    per_domain: dict[str, dict[str, float]] = Field(default_factory=dict)
    per_size: dict[str, dict[str, float]] = Field(default_factory=dict)
    num_images: int = 0
    num_ground_truth_objects: int = 0
    num_predictions: int = 0
    meta: dict[str, Any] = Field(default_factory=dict)
