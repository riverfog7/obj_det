from __future__ import annotations

import logging
import math
from typing import Any

from datasets import Dataset

from obj_det.models.adapters.base import BaseModelAdapter
from obj_det.models.logging.factory import LoggerFactory
from obj_det.models.schemas.config import EvalConfig, TrainConfig
from obj_det.models.schemas.tuning import BestTrial, SearchSpace, TrialResult, TuningConfig, TuningResult
from obj_det.models.training import require_metric, require_single_process


logger = logging.getLogger(__name__)


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

        require_single_process(context="Controlled HPO")
        if tuning_cfg.early_stopping:
            raise ValueError("Controlled HPO requires early_stopping=false")
        names = set(search_space.params)
        if names != {"learning_rate"}:
            raise ValueError(
                "Controlled HPO must sample only canonical 'learning_rate'; "
                f"received parameters: {sorted(names)}"
            )

        tuning_cfg.output_dir.mkdir(parents=True, exist_ok=True)
        sampler = self._build_sampler(tuning_cfg, optuna)
        pruner = self._build_pruner(tuning_cfg, optuna)
        study = optuna.create_study(
            study_name=tuning_cfg.study_name,
            direction=tuning_cfg.direction,
            sampler=sampler,
            pruner=pruner,
        )
        trial_results: list[TrialResult] = []
        trial_hparams: dict[int, dict[str, Any]] = {}
        trial_train_configs: dict[int, dict[str, Any]] = {}

        def objective(trial) -> float:
            sampled_hparams = self._sample_hparams(trial, search_space)
            hparams = {**base_train_cfg.hparams, **sampled_hparams}
            trial_hparams[trial.number] = sampled_hparams
            trial_output_dir = tuning_cfg.output_dir / f"trial_{trial.number:04d}"
            train_cfg = base_train_cfg.model_copy(
                update={
                    "output_dir": trial_output_dir,
                    "max_epochs": tuning_cfg.trial_epochs,
                    "seed": tuning_cfg.seed,
                    "hparams": hparams,
                    "eval_strategy": base_train_cfg.eval_strategy.model_copy(update={"enabled": False}),
                    "early_stopping": base_train_cfg.early_stopping.model_copy(update={"enabled": False}),
                    "checkpoint": base_train_cfg.checkpoint.model_copy(
                        update={
                            "save_every_epochs": tuning_cfg.trial_epochs,
                            "save_best": False,
                            "save_last": True,
                            "keep_all_epoch_checkpoints": False,
                        }
                    ),
                },
                deep=True,
            )
            resolved_train_config = train_cfg.model_dump(mode="json")
            trial_train_configs[trial.number] = resolved_train_config
            run_name = f"{tuning_cfg.study_name}_trial_{trial.number:04d}"
            logger = self._make_logger(run_name, trial_output_dir / "logs/events.jsonl")
            run_payload = {
                **(run_config or tuning_cfg.model_dump(mode="json")),
                "trial": {
                    "number": trial.number,
                    "hparams": sampled_hparams,
                    "resolved_train_config": resolved_train_config,
                },
            }
            state = "finished"
            error = None
            try:
                if logger:
                    logger.start_run(run_name, run_payload)
                    logger.start_trial(trial.number, hparams)
                    logger.log_metrics({"run/started": 1}, step=0)
                artifact = adapter.train(
                    train_ds,
                    val_ds,
                    train_cfg,
                    logger=logger,
                    log_prefix="train",
                )
                artifact.meta.update(
                    {
                        "checkpoint_selection": "trial_final",
                        "trial_number": trial.number,
                        "trial_epochs": tuning_cfg.trial_epochs,
                        "scheduler_total_epochs": train_cfg.scheduler.total_epochs,
                        "resolved_train_config": resolved_train_config,
                    }
                )
                if logger and artifact.checkpoint_path is not None:
                    logger.log_artifact(artifact.checkpoint_path, name="checkpoint")
                result = adapter.evaluate(
                    val_ds,
                    artifact,
                    self._hpo_eval_cfg(eval_cfg, detailed=tuning_cfg.detailed_eval),
                    logger=logger,
                    log_prefix="val",
                )
                metric_value = require_metric(
                    result.metrics,
                    tuning_cfg.objective_metric,
                    context="HPO objective",
                )
                artifact.best_metric_name = tuning_cfg.objective_metric
                artifact.best_metric_value = metric_value
                artifact.meta.update(
                    {
                        "objective_metric": tuning_cfg.objective_metric,
                        "objective_metric_value": metric_value,
                    }
                )
                checkpoint_meta = {
                    "checkpoint_selection": "trial_final",
                    "checkpoint_epoch": tuning_cfg.trial_epochs,
                    "scheduler_total_epochs": train_cfg.scheduler.total_epochs,
                    "optimizer_steps": artifact.meta.get(
                        "optimizer_steps",
                        artifact.meta.get("trainer_global_step"),
                    ),
                }
                trial.report(metric_value, step=tuning_cfg.trial_epochs)
                if logger:
                    logger.log_metrics({f"objective/{tuning_cfg.objective_metric}": metric_value})
                trial_results.append(
                    TrialResult(
                        trial_number=trial.number,
                        state="complete",
                        hparams=sampled_hparams,
                        metric_name=tuning_cfg.objective_metric,
                        metric_value=metric_value,
                        artifact_path=artifact.artifact_path,
                        checkpoint_path=artifact.checkpoint_path,
                        checkpoint_meta=checkpoint_meta,
                        resolved_train_config=resolved_train_config,
                        meta={"model_artifact": artifact.model_dump(mode="json")},
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
                        hparams=sampled_hparams,
                        metric_name=tuning_cfg.objective_metric,
                        resolved_train_config=resolved_train_config,
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
            catch=(Exception,) if tuning_cfg.catch_trial_errors else (),
        )

        try:
            best_trial = study.best_trial
        except ValueError:
            return TuningResult(study_name=tuning_cfg.study_name, trials=trial_results)

        winning_result = next(
            (item for item in trial_results if item.trial_number == best_trial.number),
            None,
        )

        best = BestTrial(
            study_name=tuning_cfg.study_name,
            trial_number=best_trial.number,
            hparams=trial_hparams.get(best_trial.number, dict(best_trial.params)),
            metric_name=tuning_cfg.objective_metric,
            metric_value=float(study.best_value),
            artifact_path=winning_result.artifact_path if winning_result is not None else None,
            checkpoint_path=winning_result.checkpoint_path if winning_result is not None else None,
            checkpoint_meta=(winning_result.checkpoint_meta if winning_result is not None else {}),
            resolved_train_config=trial_train_configs.get(best_trial.number, {}),
        )
        boundary_warning = self._boundary_warning(best.hparams, search_space)
        if boundary_warning is not None:
            logger.warning(boundary_warning)
            best.meta["boundary_warning"] = boundary_warning
        return TuningResult(
            study_name=tuning_cfg.study_name,
            best_trial=best,
            trials=trial_results,
            meta={"boundary_warning": boundary_warning} if boundary_warning is not None else {},
        )

    def _hpo_eval_cfg(self, eval_cfg: EvalConfig, *, detailed: bool) -> EvalConfig:
        if detailed:
            return eval_cfg
        return eval_cfg.model_copy(
            update={
                "compute_per_class": False,
                "compute_per_condition": False,
                "compute_per_domain": False,
                "compute_per_size": False,
            },
            deep=True,
        )

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
            return optuna.samplers.TPESampler(seed=tuning_cfg.seed, **tuning_cfg.sampler_params)
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

    def _boundary_warning(self, hparams: dict[str, Any], search_space: SearchSpace) -> str | None:
        if "learning_rate" not in hparams or "learning_rate" not in search_space.params:
            return None
        spec = search_space.params["learning_rate"]
        if spec.get("type") != "float" or not spec.get("log", False):
            return None
        low = float(spec["low"])
        high = float(spec["high"])
        value = float(hparams["learning_rate"])
        if low <= 0.0 or high <= low or value <= 0.0:
            return None
        position = (math.log(value) - math.log(low)) / (math.log(high) - math.log(low))
        if position <= 0.05:
            return f"Best learning_rate={value:g} lies within 5% of the lower log-space boundary {low:g}"
        if position >= 0.95:
            return f"Best learning_rate={value:g} lies within 5% of the upper log-space boundary {high:g}"
        return None
