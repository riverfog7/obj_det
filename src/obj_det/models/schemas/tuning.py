from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from .base import ModelSchema


SamplerName = Literal["tpe", "random"]
PrunerName = Literal["none", "median", "asha"]


class TuningConfig(ModelSchema):
    study_name: str
    direction: Literal["maximize", "minimize"] = "maximize"
    n_trials: int = Field(default=10, gt=0)
    timeout_seconds: int | None = Field(default=None, gt=0)
    sampler: SamplerName = "tpe"
    pruner: PrunerName = "median"
    seed: int = 0
    objective_metric: str = "map_50_95"
    storage: str | None = None
    output_dir: Path
    log_to_wandb: bool = False
    meta: dict[str, Any] = Field(default_factory=dict)


class SearchSpace(ModelSchema):
    params: dict[str, dict[str, Any]] = Field(default_factory=dict)


class TrialResult(ModelSchema):
    trial_number: int
    state: str
    hparams: dict[str, Any] = Field(default_factory=dict)
    metric_name: str | None = None
    metric_value: float | None = None
    artifact_path: Path | None = None
    error: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class BestTrial(ModelSchema):
    study_name: str
    trial_number: int
    hparams: dict[str, Any]
    metric_name: str
    metric_value: float
    artifact_path: Path | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class TuningResult(ModelSchema):
    study_name: str
    best_trial: BestTrial | None = None
    trials: list[TrialResult] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
