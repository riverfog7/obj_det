from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from .base import ModelSchema


SamplerName = Literal["tpe", "random"]
PrunerName = Literal["none", "median", "asha"]


class TuningConfig(ModelSchema):
    study_name: str
    direction: Literal["maximize", "minimize"] = "maximize"
    n_trials: int = Field(default=8, gt=0)
    trial_epochs: int = Field(default=10, gt=0)
    timeout_seconds: int | None = Field(default=None, gt=0)
    sampler: SamplerName = "tpe"
    sampler_params: dict[str, Any] = Field(default_factory=lambda: {"n_startup_trials": 3})
    pruner: PrunerName = "none"
    seed: int = 0
    objective_metric: str = "map_50_95"
    save_strategy: Literal["final_only"] = "final_only"
    early_stopping: bool = False
    storage: str | None = None
    output_dir: Path
    catch_trial_errors: bool = False
    detailed_eval: bool = False
    meta: dict[str, Any] = Field(default_factory=dict)


class SearchSpace(ModelSchema):
    params: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_specs(self) -> "SearchSpace":
        for name, spec in self.params.items():
            kind = spec.get("type")
            if kind not in {"float", "int", "categorical"}:
                raise ValueError(f"Invalid search space for {name!r}: type must be float, int, or categorical")

            if kind in {"float", "int"}:
                if "choices" in spec:
                    raise ValueError(f"Invalid search space for {name!r}: choices is only valid for categorical")
                if "low" not in spec or "high" not in spec:
                    raise ValueError(f"Invalid search space for {name!r}: {kind} requires low and high")
                if spec["low"] > spec["high"]:
                    raise ValueError(f"Invalid search space for {name!r}: low must be <= high")
                continue

            if "log" in spec:
                raise ValueError(f"Invalid search space for {name!r}: log is only valid for float or int")
            choices = spec.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError(f"Invalid search space for {name!r}: categorical requires non-empty choices")

        return self


class TrialResult(ModelSchema):
    trial_number: int
    state: str
    hparams: dict[str, Any] = Field(default_factory=dict)
    metric_name: str | None = None
    metric_value: float | None = None
    artifact_path: Path | None = None
    checkpoint_path: Path | None = None
    checkpoint_meta: dict[str, Any] = Field(default_factory=dict)
    resolved_train_config: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class BestTrial(ModelSchema):
    study_name: str
    trial_number: int
    hparams: dict[str, Any]
    metric_name: str
    metric_value: float
    artifact_path: Path | None = None
    checkpoint_path: Path | None = None
    checkpoint_meta: dict[str, Any] = Field(default_factory=dict)
    resolved_train_config: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)


class TuningResult(ModelSchema):
    study_name: str
    best_trial: BestTrial | None = None
    trials: list[TrialResult] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
