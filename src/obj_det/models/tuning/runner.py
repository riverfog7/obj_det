from __future__ import annotations

from typing import Any

from datasets import Dataset

from obj_det.models.adapters.base import BaseModelAdapter
from obj_det.models.logging.factory import LoggerFactory
from obj_det.models.schemas.config import EvalConfig, TrainConfig
from obj_det.models.schemas.tuning import BestTrial, SearchSpace, TrialResult, TuningConfig, TuningResult


class TuningRunner:
    def __init__(self, logger_factory: LoggerFactory | None = None):
        self.logger_factory = logger_factory

    def optimize(
        self,
        adapter: BaseModelAdapter,
        train_ds: Dataset,
        val_ds: Dataset,
        base_train_cfg: TrainConfig,
        eval_cfg: EvalConfig,
        search_space: SearchSpace,
        tuning_cfg: TuningConfig,
        run_config: dict[str, Any] | None = None,
    ) -> TuningResult:
        try:
            import optuna
        except ImportError as exc:
            raise ImportError("Install optuna to use TuningRunner.") from exc

        tuning_cfg.output_dir.mkdir(parents=True, exist_ok=True)
        sampler = self._build_sampler(tuning_cfg, optuna)
        pruner = self._build_pruner(tuning_cfg, optuna)
        study = optuna.create_study(
            study_name=tuning_cfg.study_name,
            direction=tuning_cfg.direction,
            sampler=sampler,
            pruner=pruner,
            storage=tuning_cfg.storage,
            load_if_exists=bool(tuning_cfg.storage),
        )
        trial_results: list[TrialResult] = []

        def objective(trial) -> float:
            hparams = self._sample_hparams(trial, search_space)
            trial_output_dir = tuning_cfg.output_dir / f"trial_{trial.number:04d}"
            train_cfg = base_train_cfg.model_copy(
                update={
                    "output_dir": trial_output_dir,
                    "hparams": {**base_train_cfg.hparams, **hparams},
                },
                deep=True,
            )
            run_name = f"{tuning_cfg.study_name}_trial_{trial.number:04d}"
            logger = self._make_logger(run_name, trial_output_dir / "logs/events.jsonl")
            run_payload = {
                **(run_config or tuning_cfg.model_dump(mode="json")),
                "trial": {
                    "number": trial.number,
                    "hparams": hparams,
                },
            }
            state = "finished"
            error = None
            try:
                if logger:
                    logger.start_run(run_name, run_payload)
                    logger.start_trial(trial.number, hparams)
                artifact = adapter.train(
                    train_ds,
                    val_ds,
                    train_cfg,
                    logger=logger,
                    log_prefix="train",
                )
                if logger and artifact.checkpoint_path is not None:
                    logger.log_artifact(artifact.checkpoint_path, name="checkpoint")
                result = adapter.evaluate(
                    val_ds,
                    artifact,
                    eval_cfg,
                    logger=logger,
                    log_prefix="val",
                )
                metric_value = result.metrics.get(tuning_cfg.objective_metric, result.primary_metric_value)
                trial.report(metric_value, step=0)
                if logger:
                    logger.log_metrics({f"objective/{tuning_cfg.objective_metric}": metric_value})
                trial_results.append(
                    TrialResult(
                        trial_number=trial.number,
                        state="complete",
                        hparams=hparams,
                        metric_name=tuning_cfg.objective_metric,
                        metric_value=metric_value,
                        artifact_path=artifact.artifact_path,
                    )
                )
                if logger:
                    logger.finish_trial("complete")
                return metric_value
            except Exception as exc:
                state = "failed"
                error = repr(exc)
                trial_results.append(
                    TrialResult(
                        trial_number=trial.number,
                        state="failed",
                        hparams=hparams,
                        metric_name=tuning_cfg.objective_metric,
                        error=repr(exc),
                    )
                )
                if logger:
                    logger.finish_trial("failed", error=error)
                raise
            finally:
                if logger:
                    logger.finish_run(state=state, error=error)

        study.optimize(
            objective,
            n_trials=tuning_cfg.n_trials,
            timeout=tuning_cfg.timeout_seconds,
            catch=(Exception,),
        )

        try:
            best_trial = study.best_trial
        except ValueError:
            return TuningResult(study_name=tuning_cfg.study_name, trials=trial_results)

        best_artifact = None
        for trial_result in trial_results:
            if trial_result.trial_number == best_trial.number:
                best_artifact = trial_result.artifact_path
                break

        best = BestTrial(
            study_name=tuning_cfg.study_name,
            trial_number=best_trial.number,
            hparams=dict(best_trial.params),
            metric_name=tuning_cfg.objective_metric,
            metric_value=float(study.best_value),
            artifact_path=best_artifact,
        )
        return TuningResult(study_name=tuning_cfg.study_name, best_trial=best, trials=trial_results)

    def _sample_hparams(self, trial, search_space: SearchSpace) -> dict[str, Any]:
        sampled = {}
        for name, spec in search_space.params.items():
            kind = spec["type"]
            if kind == "float":
                sampled[name] = trial.suggest_float(
                    name,
                    spec["low"],
                    spec["high"],
                    log=spec.get("log", False),
                )
            elif kind == "int":
                sampled[name] = trial.suggest_int(
                    name,
                    spec["low"],
                    spec["high"],
                    log=spec.get("log", False),
                )
            elif kind == "categorical":
                sampled[name] = trial.suggest_categorical(name, spec["choices"])
            else:
                raise ValueError(f"Unsupported search-space type: {kind}")
        return sampled

    def _build_sampler(self, tuning_cfg: TuningConfig, optuna):
        if tuning_cfg.sampler == "random":
            return optuna.samplers.RandomSampler(seed=tuning_cfg.seed)
        if tuning_cfg.sampler == "tpe":
            return optuna.samplers.TPESampler(seed=tuning_cfg.seed)
        raise ValueError(f"Unknown sampler: {tuning_cfg.sampler}")

    def _build_pruner(self, tuning_cfg: TuningConfig, optuna):
        if tuning_cfg.pruner == "none":
            return optuna.pruners.NopPruner()
        if tuning_cfg.pruner == "median":
            return optuna.pruners.MedianPruner()
        if tuning_cfg.pruner == "asha":
            return optuna.pruners.SuccessiveHalvingPruner()
        raise ValueError(f"Unknown pruner: {tuning_cfg.pruner}")

    def _make_logger(self, run_name: str, default_log_path):
        if self.logger_factory is None:
            return None
        return self.logger_factory(run_name, default_log_path)
