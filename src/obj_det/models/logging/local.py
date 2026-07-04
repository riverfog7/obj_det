from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from obj_det.models.schemas.result import EvalResult

from .base import BaseExperimentLogger
from .metrics import flatten_eval_result


class LocalJsonLogger(BaseExperimentLogger):
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def start_run(self, name: str, config: dict[str, Any]) -> None:
        self._write({"event": "start_run", "name": name, "config": config})

    def finish_run(self, state: str = "finished", error: str | None = None) -> None:
        self._write({"event": "finish_run", "state": state, "error": error})

    def start_trial(self, trial_number: int, hparams: dict[str, Any]) -> None:
        self._write({"event": "start_trial", "trial_number": trial_number, "hparams": hparams})

    def finish_trial(self, state: str, error: str | None = None) -> None:
        self._write({"event": "finish_trial", "state": state, "error": error})

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self._write({"event": "metrics", "step": step, "metrics": metrics})

    def log_eval_result(self, result: EvalResult, step: int | None = None, prefix: str | None = None) -> None:
        self._write({
            "event": "eval_result",
            "step": step,
            "prefix": prefix,
            "metrics": flatten_eval_result(result, prefix=prefix),
            "result": result.model_dump(mode="json"),
        })

    def log_artifact(self, path, name: str | None = None) -> None:
        self._write({"event": "artifact", "path": str(path), "name": name})

    def _write(self, row: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str, sort_keys=True) + "\n")
