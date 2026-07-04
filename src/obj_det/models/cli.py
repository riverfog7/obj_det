from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from datasets import load_from_disk

from obj_det.models.experiment import (
    load_best_trial,
    load_experiment_config,
    load_model_artifact,
    save_best_trial,
    save_eval_result,
    save_model_artifact,
    save_tuning_result,
)
from obj_det.models.schemas.artifact import ModelArtifact
from obj_det.models.schemas.experiment import ExperimentConfig
from obj_det.models.logging.wandb import WandbLogger
from obj_det.models.tuning.final import FinalSeedRun, run_final_seeds
from obj_det.models.tuning.runner import TuningRunner


app = typer.Typer(no_args_is_help=True)


@app.command()
def train(
    config: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
) -> None:
    exp = load_experiment_config(config)
    dataset = load_from_disk(exp.dataset.path)
    adapter = _adapter(exp)

    artifact = adapter.train(
        train_ds=_split(dataset, exp.dataset.train_split),
        val_ds=_split(dataset, exp.dataset.val_split),
        train_cfg=exp.train,
    )
    path = save_model_artifact(artifact, exp.train.output_dir / "artifact.json")
    typer.echo(f"Saved artifact metadata to {path}")


@app.command()
def evaluate(
    config: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    artifact: Annotated[
        Path,
        typer.Option("--artifact", exists=True, dir_okay=False, readable=True),
    ],
    split: Annotated[
        str,
        typer.Option("--split", help="Dataset split to evaluate."),
    ] = "test",
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Output JSON path."),
    ] = None,
) -> None:
    exp = load_experiment_config(config)
    artifact_obj = load_model_artifact(artifact)
    dataset = load_from_disk(exp.dataset.path)
    adapter = _adapter(exp)

    result = adapter.evaluate(_split(dataset, split), artifact_obj, exp.eval)
    out_path = out or _artifact_dir(artifact_obj, artifact) / f"eval_{split}.json"
    save_eval_result(result, out_path)
    typer.echo(f"Saved eval result to {out_path}")


@app.command()
def optimize(
    config: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
) -> None:
    exp = load_experiment_config(config)
    if exp.tuning is None:
        raise typer.ClickException("Experiment config is missing tuning")
    if exp.search_space is None:
        raise typer.ClickException("Experiment config is missing search_space or search_space_file")

    dataset = load_from_disk(exp.dataset.path)
    adapter = _adapter(exp)

    result = TuningRunner(logger=_tuning_logger(exp)).optimize(
        adapter=adapter,
        train_ds=_split(dataset, exp.dataset.train_split),
        val_ds=_split(dataset, exp.dataset.val_split),
        base_train_cfg=exp.train,
        eval_cfg=exp.eval,
        search_space=exp.search_space,
        tuning_cfg=exp.tuning,
    )
    result_path = save_tuning_result(result, exp.tuning.output_dir / "tuning_result.json")
    typer.echo(f"Saved tuning result to {result_path}")

    if result.best_trial is not None:
        best_path = save_best_trial(result.best_trial, exp.tuning.output_dir / "best_trial.json")
        typer.echo(f"Saved best trial to {best_path}")
    else:
        typer.echo("No completed trials; best_trial.json was not written")


@app.command("final")
def final_(
    config: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    best_trial: Annotated[
        Path,
        typer.Option("--best-trial", exists=True, dir_okay=False, readable=True),
    ],
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Output JSON path."),
    ] = None,
) -> None:
    exp = load_experiment_config(config)
    best = load_best_trial(best_trial)
    dataset = load_from_disk(exp.dataset.path)
    adapter = _adapter(exp)

    output_dir = exp.final.output_dir or exp.train.output_dir / "final"
    runs = run_final_seeds(
        adapter=adapter,
        train_ds=_split(dataset, exp.dataset.train_split),
        val_ds=_split(dataset, exp.dataset.val_split),
        test_ds=_split(dataset, exp.dataset.test_split),
        base_train_cfg=exp.train,
        eval_cfg=exp.eval,
        hparams=best.hparams,
        seeds=exp.final.seeds,
        output_dir=output_dir,
        evaluate_val=exp.final.evaluate_val,
    )
    out_path = out or output_dir / "final_results.json"
    _write_final_results(runs, out_path)
    typer.echo(f"Saved final results to {out_path}")


def _model(exp: ExperimentConfig):
    if exp.model is None:
        raise ValueError("Experiment config has no resolved model")
    return exp.model


def _adapter(exp: ExperimentConfig):
    from obj_det.models.adapters.factory import model_adapter_from_config

    return model_adapter_from_config(_model(exp))


def _tuning_logger(exp: ExperimentConfig):
    if exp.tuning is None or not exp.tuning.log_to_wandb:
        return None
    return WandbLogger(project=exp.tuning.study_name)


def _split(dataset, split: str):
    if split not in dataset:
        raise KeyError(f"Missing split {split!r}. Available splits: {list(dataset.keys())}")
    return dataset[split]


def _artifact_dir(artifact: ModelArtifact, artifact_file: Path) -> Path:
    if artifact.artifact_path is not None:
        return artifact.artifact_path
    return artifact_file.parent


def _write_final_results(runs: list[FinalSeedRun], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "runs": [
            {
                "seed": run.seed,
                "artifact": run.artifact.model_dump(mode="json"),
                "val_result": run.val_result.model_dump(mode="json") if run.val_result is not None else None,
                "test_result": run.test_result.model_dump(mode="json") if run.test_result is not None else None,
            }
            for run in runs
        ]
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path
