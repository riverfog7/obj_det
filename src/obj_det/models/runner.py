from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

from datasets import load_from_disk

from obj_det.models.adapters.factory import model_adapter_from_config
from obj_det.models.logging.factory import child_logger_factory_from_config, logger_from_config
from obj_det.models.schemas.artifact import ModelArtifact
from obj_det.models.schemas.experiment import ExperimentConfig
from obj_det.models.schemas.result import EvalResult
from obj_det.models.schemas.tuning import BestTrial, TuningResult
from obj_det.models.tuning.final import FinalSeedRun, run_final_seeds
from obj_det.models.tuning.runner import TuningRunner


class ExperimentRunner:
    """Python-first experiment orchestration for train/eval/HPO/final workflows."""

    def __init__(self, exp: ExperimentConfig):
        self.exp = exp

    def train(self) -> ModelArtifact:
        dataset = self._dataset()
        adapter = self._adapter()
        logger = self._logger(default_log_path=self.exp.train.output_dir / "logs/events.jsonl", run_name=self.exp.train.run_key)
        state = "finished"
        error = None

        try:
            if logger:
                logger.start_run(self.exp.train.run_key, self.run_config())
                logger.log_metrics({"run/started": 1}, step=0)
            artifact = adapter.train(
                train_ds=self._split(dataset, self.exp.dataset.train_split),
                val_ds=self._split(dataset, self.exp.dataset.val_split),
                train_cfg=self.exp.train,
                epoch_eval_cfg=self.exp.eval if self.exp.train.eval_strategy.enabled else None,
                logger=logger,
                log_prefix="train",
            )
            if logger and artifact.checkpoint_path is not None:
                logger.log_artifact(artifact.checkpoint_path, name="checkpoint")
            return artifact
        except Exception as exc:
            state = "failed"
            error = repr(exc)
            raise
        finally:
            if logger:
                logger.finish_run(state=state, error=error)

    def evaluate(self, artifact: ModelArtifact, *, split: str = "test", out_path: Path | None = None) -> EvalResult:
        dataset = self._dataset()
        adapter = self._adapter()
        log_dir = (out_path.parent if out_path is not None else self.artifact_dir(artifact) / "logs")
        logger = self._logger(
            default_log_path=log_dir / "events.jsonl",
            run_name=f"{artifact.run_key}_eval_{split}",
        )
        state = "finished"
        error = None

        try:
            if logger:
                logger.start_run(f"{artifact.run_key}_eval_{split}", self.run_config())
                logger.log_metrics({"run/started": 1}, step=0)
            result = adapter.evaluate(
                self._split(dataset, split),
                artifact,
                self.exp.eval,
                logger=logger,
                log_prefix=f"eval/{split}",
            )
            if logger and out_path is not None:
                logger.log_artifact(out_path, name=f"eval_{split}")
            return result
        except Exception as exc:
            state = "failed"
            error = repr(exc)
            raise
        finally:
            if logger:
                logger.finish_run(state=state, error=error)

    def optimize(self) -> TuningResult:
        if self.exp.tuning is None:
            raise ValueError("Experiment config is missing tuning")
        if self.exp.search_space is None:
            raise ValueError("Experiment config is missing search_space or search_space_file")

        dataset = self._dataset()
        adapter = self._adapter()
        logger_factory = self._child_logger_factory(wandb_group=self.exp.tuning.study_name)
        return TuningRunner(logger_factory=logger_factory).optimize(
            adapter=adapter,
            train_ds=self._split(dataset, self.exp.dataset.train_split),
            val_ds=self._split(dataset, self.exp.dataset.val_split),
            base_train_cfg=self.exp.train,
            eval_cfg=self.exp.eval,
            search_space=self.exp.search_space,
            tuning_cfg=self.exp.tuning,
            run_config=self.run_config(),
        )

    def final(self, best_trial: BestTrial, *, output_dir: Path | None = None) -> list[FinalSeedRun]:
        dataset = self._dataset()
        adapter = self._adapter()
        final_output_dir = output_dir or self.exp.final.output_dir or self.exp.train.output_dir / "final"
        logger_factory = self._child_logger_factory(wandb_group=f"{self.exp.train.run_key}_final")
        return run_final_seeds(
            adapter=adapter,
            train_ds=self._split(dataset, self.exp.dataset.train_split),
            val_ds=self._split(dataset, self.exp.dataset.val_split),
            test_ds=self._split(dataset, self.exp.dataset.test_split),
            base_train_cfg=self.exp.train,
            eval_cfg=self.exp.eval,
            hparams=best_trial.hparams,
            seeds=self.exp.final.seeds,
            output_dir=final_output_dir,
            evaluate_val=self.exp.final.evaluate_val,
            logger_factory=logger_factory,
            run_config=self.run_config(),
        )

    def artifact_dir(self, artifact: ModelArtifact, artifact_file: Path | None = None) -> Path:
        if artifact.artifact_path is not None:
            return artifact.artifact_path
        if artifact_file is not None:
            return artifact_file.parent
        return self.exp.train.output_dir

    def run_config(self) -> dict:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        data = self.exp.model_dump(mode="json")
        gradient_accumulation_steps = int(self.exp.train.gradient_accumulation_steps)
        data["batch"] = {
            "batch_size": self.exp.train.batch_size,
            "world_size": world_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": self.exp.train.batch_size * world_size * gradient_accumulation_steps,
        }
        return data

    def _dataset(self):
        return load_from_disk(self.exp.dataset.path)

    def _adapter(self):
        if self.exp.model is None:
            raise ValueError("Experiment config has no resolved model")
        return model_adapter_from_config(self.exp.model)

    def _logger(self, *, default_log_path: Path, run_name: str):
        return logger_from_config(self.exp.logging, default_log_path=default_log_path, run_name=run_name)

    def _child_logger_factory(self, *, wandb_group: str):
        return child_logger_factory_from_config(self.exp.logging, wandb_group=wandb_group)

    def _split(self, dataset, split: str):
        if split not in dataset:
            raise KeyError(f"Missing split {split!r}. Available splits: {list(dataset.keys())}")
        return dataset[split]

def write_final_results(runs: list[FinalSeedRun], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = [run for run in runs if run.state == "complete"]
    failed = [run for run in runs if run.state != "complete"]
    data = {
        "aggregate": {
            "successful_seeds": len(completed),
            "failed_seeds": len(failed),
            "val": _aggregate_results(completed, "val_result"),
            "test": _aggregate_results(completed, "test_result"),
        },
        "runs": [
            {
                "seed": run.seed,
                "state": run.state,
                "error": run.error,
                "artifact": run.artifact.model_dump(mode="json") if run.artifact is not None else None,
                "val_result": run.val_result.model_dump(mode="json") if run.val_result is not None else None,
                "test_result": run.test_result.model_dump(mode="json") if run.test_result is not None else None,
            }
            for run in runs
        ]
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def _aggregate_results(runs: list[FinalSeedRun], attribute: str) -> dict:
    results = [getattr(run, attribute) for run in runs]
    results = [result for result in results if result is not None]
    if not results:
        return {}

    shared_metrics = set(results[0].metrics)
    for result in results[1:]:
        shared_metrics.intersection_update(result.metrics)

    aggregate = {}
    for name in sorted(shared_metrics):
        values = [float(result.metrics[name]) for result in results]
        aggregate[name] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    return aggregate
