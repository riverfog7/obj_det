from __future__ import annotations

from typing import Any

from obj_det.models.schemas.result import EvalResult

from .base import BaseExperimentLogger
from .metrics import flatten_eval_result, scalar_or_none


class WandbLogger(BaseExperimentLogger):
    def __init__(
        self,
        *,
        project: str,
        entity: str | None = None,
        group: str | None = None,
        name: str | None = None,
        mode: str = "online",
        tags: list[str] | None = None,
        **init_kwargs: Any,
    ):
        self.project = project
        self.entity = entity
        self.group = group
        self.name = name
        self.mode = mode
        self.tags = tags or []
        self.init_kwargs = init_kwargs
        self.run = None

    def _wandb(self):
        try:
            import wandb
        except ImportError as exc:
            raise ImportError("Install wandb or use NullLogger/LocalJsonLogger.") from exc
        return wandb

    def start_run(self, name: str, config: dict[str, Any]) -> None:
        wandb = self._wandb()
        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            group=self.group,
            name=self.name or name,
            mode=self.mode,
            tags=self.tags,
            config=config,
            **self.init_kwargs,
        )

    def finish_run(self, state: str = "finished", error: str | None = None) -> None:
        if self.run is not None:
            if error:
                self.run.summary["error"] = error
            self.run.finish(exit_code=0 if state in {"finished", "complete"} else 1)
            self.run = None

    def start_trial(self, trial_number: int, hparams: dict[str, Any]) -> None:
        self.log_metrics({"trial/number": trial_number})
        if self.run is not None:
            self.run.config.update(
                {
                    "trial/number": trial_number,
                    **{f"hparams/{key}": value for key, value in hparams.items()},
                },
                allow_val_change=True,
            )

    def finish_trial(self, state: str, error: str | None = None) -> None:
        data: dict[str, Any] = {"trial/state": state}
        if error:
            data["trial/error"] = error
        self._wandb().log(data)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        scalars = {}
        for key, value in metrics.items():
            scalar = scalar_or_none(value)
            if scalar is not None:
                scalars[key] = scalar
        if scalars:
            self._wandb().log(scalars, step=step)

    def log_eval_result(self, result: EvalResult, step: int | None = None, prefix: str | None = None) -> None:
        metrics = flatten_eval_result(result, prefix=prefix)
        if step is None:
            self.log_metrics(metrics)
            return
        self._wandb().log(metrics, step=step, commit=False)

    def log_artifact(self, path, name: str | None = None) -> None:
        if self.run is not None:
            self.run.summary[f"artifact/{name or 'path'}"] = str(path)
