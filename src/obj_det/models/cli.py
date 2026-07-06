from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from obj_det.models.experiment import (
    load_best_trial,
    load_experiment_config,
    load_model_artifact,
    save_best_trial,
    save_eval_result,
    save_model_artifact,
    save_tuning_result,
)
from obj_det.models.runner import ExperimentRunner, write_final_results


app = typer.Typer(no_args_is_help=True)


@app.command()
def train(
    config: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
) -> None:
    exp = load_experiment_config(config)
    artifact = ExperimentRunner(exp).train()
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
    runner = ExperimentRunner(exp)
    artifact_obj = load_model_artifact(artifact)
    out_path = out or runner.artifact_dir(artifact_obj, artifact) / f"eval_{split}.json"
    result = runner.evaluate(artifact_obj, split=split, out_path=out_path)
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
    try:
        result = ExperimentRunner(exp).optimize()
    except ValueError as exc:
        raise typer.ClickException(str(exc)) from exc

    assert exp.tuning is not None
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
    output_dir = exp.final.output_dir or exp.train.output_dir / "final"
    out_path = out or output_dir / "final_results.json"
    runs = ExperimentRunner(exp).final(best, output_dir=output_dir)
    write_final_results(runs, out_path)
    typer.echo(f"Saved final results to {out_path}")
