from __future__ import annotations

from typing import Any

from obj_det.models.schemas.result import EvalResult

from .base import BaseExperimentLogger


class WandbLogger(BaseExperimentLogger):
    def __init__(self, *, project: str, entity: str | None = None, **init_kwargs: Any):
        self.project = project
        self.entity = entity
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
            name=name,
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
        self.log_metrics({f"trial/{k}": v for k, v in hparams.items() if isinstance(v, (int, float, bool))})

    def finish_trial(self, state: str, error: str | None = None) -> None:
        data: dict[str, Any] = {"trial/state": state}
        if error:
            data["trial/error"] = error
        self._wandb().log(data)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._wandb().log(metrics, step=step)

    def log_eval_result(self, result: EvalResult, step: int | None = None) -> None:
        self.log_metrics(result.metrics, step=step)

    def log_artifact(self, path, name: str | None = None) -> None:
        wandb = self._wandb()
        artifact = wandb.Artifact(name=name or "artifact", type="model")
        artifact.add_file(str(path))
        wandb.log_artifact(artifact)
