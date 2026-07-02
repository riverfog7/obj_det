from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from .base import ModelSchema


class ModelArtifact(ModelSchema):
    model_key: str
    backend: str
    run_key: str
    classes: list[str]
    label_mode: str
    artifact_path: Path | None = None
    checkpoint_path: Path | None = None
    best_metric_name: str | None = None
    best_metric_value: float | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
