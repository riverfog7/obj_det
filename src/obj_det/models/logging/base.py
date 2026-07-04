from __future__ import annotations

from abc import ABC
from typing import Any

from obj_det.models.schemas.result import EvalResult


class BaseExperimentLogger(ABC):
    def start_run(self, name: str, config: dict[str, Any]) -> None:
        pass

    def finish_run(self, state: str = "finished", error: str | None = None) -> None:
        pass

    def start_trial(self, trial_number: int, hparams: dict[str, Any]) -> None:
        pass

    def finish_trial(self, state: str, error: str | None = None) -> None:
        pass

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        pass

    def log_eval_result(self, result: EvalResult, step: int | None = None, prefix: str | None = None) -> None:
        from obj_det.models.logging.metrics import flatten_eval_result

        self.log_metrics(flatten_eval_result(result, prefix=prefix), step=step)

    def log_artifact(self, path, name: str | None = None) -> None:
        pass
