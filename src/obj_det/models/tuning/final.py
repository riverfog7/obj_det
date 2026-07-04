from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset

from obj_det.models.adapters.base import BaseModelAdapter
from obj_det.models.logging.factory import LoggerFactory
from obj_det.models.schemas.artifact import ModelArtifact
from obj_det.models.schemas.config import EvalConfig, TrainConfig
from obj_det.models.schemas.result import EvalResult


@dataclass
class FinalSeedRun:
    seed: int
    artifact: ModelArtifact
    val_result: EvalResult | None
    test_result: EvalResult | None


def run_final_seeds(
    *,
    adapter: BaseModelAdapter,
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset | None,
    base_train_cfg: TrainConfig,
    eval_cfg: EvalConfig,
    hparams: dict[str, Any],
    seeds: Iterable[int],
    output_dir: Path | None = None,
    evaluate_val: bool = True,
    logger_factory: LoggerFactory | None = None,
    run_config: dict[str, Any] | None = None,
) -> list[FinalSeedRun]:
    """Train/evaluate final runs for fixed seeds without choosing a best seed."""

    runs: list[FinalSeedRun] = []
    for seed in seeds:
        run_output_dir = (output_dir or base_train_cfg.output_dir) / f"final_seed{seed}"
        train_cfg = base_train_cfg.model_copy(
            update={
                "run_key": f"{base_train_cfg.run_key}_seed{seed}",
                "seed": seed,
                "hparams": hparams,
                "output_dir": run_output_dir,
            },
            deep=True,
        )
        run_name = f"{base_train_cfg.run_key}_final_seed{seed}"
        logger = logger_factory(run_name, run_output_dir / "logs/events.jsonl") if logger_factory else None
        state = "finished"
        error = None
        try:
            if logger is not None:
                logger.start_run(
                    run_name,
                    {
                        **(run_config or {}),
                        "final": {
                            "seed": seed,
                            "hparams": hparams,
                        },
                    },
                )
            artifact = adapter.train(train_ds, val_ds, train_cfg, logger=logger, log_prefix="train")
            val_result = (
                adapter.evaluate(val_ds, artifact, eval_cfg, logger=logger, log_prefix="val")
                if evaluate_val
                else None
            )
            test_result = (
                adapter.evaluate(test_ds, artifact, eval_cfg, logger=logger, log_prefix="test")
                if test_ds is not None
                else None
            )
            if logger is not None and artifact.checkpoint_path is not None:
                logger.log_artifact(artifact.checkpoint_path, name="checkpoint")
        except Exception as exc:
            state = "failed"
            error = repr(exc)
            raise
        finally:
            if logger is not None:
                logger.finish_run(state=state, error=error)
        runs.append(FinalSeedRun(seed=seed, artifact=artifact, val_result=val_result, test_result=test_result))
    return runs
