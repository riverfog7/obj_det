from __future__ import annotations

from typing import Any

from pydantic import Field

from obj_det.datasets.models import BBox

from .base import ModelSchema


class PredictionObject(ModelSchema):
    bbox: BBox
    label: str
    score: float = Field(ge=0.0, le=1.0)
    label_id: int | str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class PredictionRecord(ModelSchema):
    image_id: str
    dataset: str
    split: str
    model_key: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    predictions: list[PredictionObject] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
