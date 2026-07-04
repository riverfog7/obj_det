from __future__ import annotations

from typing import Any

from obj_det.models.schemas.result import EvalResult

from .base import BaseExperimentLogger


class CompositeLogger(BaseExperimentLogger):
    def __init__(self, loggers: list[BaseExperimentLogger]):
        self.loggers = loggers

    def start_run(self, name: str, config: dict[str, Any]) -> None:
        for logger in self.loggers:
            logger.start_run(name, config)

    def finish_run(self, state: str = "finished", error: str | None = None) -> None:
        for logger in reversed(self.loggers):
            logger.finish_run(state=state, error=error)

    def start_trial(self, trial_number: int, hparams: dict[str, Any]) -> None:
        for logger in self.loggers:
            logger.start_trial(trial_number, hparams)

    def finish_trial(self, state: str, error: str | None = None) -> None:
        for logger in self.loggers:
            logger.finish_trial(state, error=error)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        for logger in self.loggers:
            logger.log_metrics(metrics, step=step)

    def log_eval_result(self, result: EvalResult, step: int | None = None, prefix: str | None = None) -> None:
        for logger in self.loggers:
            logger.log_eval_result(result, step=step, prefix=prefix)

    def log_artifact(self, path, name: str | None = None) -> None:
        for logger in self.loggers:
            logger.log_artifact(path, name=name)
