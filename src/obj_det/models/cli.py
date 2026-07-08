from __future__ import annotations

from pathlib import Path
from typing import Annotated

import click
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
from obj_det.models.plan import expand_experiment_plan, load_experiment_plan, write_resolved_experiments
from obj_det.models.runner import ExperimentRunner, write_final_results


app = typer.Typer(no_args_is_help=True)
plan_app = typer.Typer(no_args_is_help=True)
app.add_typer(plan_app, name="plan")


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
    _optimize_experiment(exp)


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
    _final_experiment(exp, best_trial=best_trial, out=out)


@plan_app.command("list")
def plan_list(
    plan: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
) -> None:
    for exp in _experiments_from_plan(plan, model_keys=None):
        assert exp.model is not None
        typer.echo(f"{exp.model.key}\t{exp.model.backend}")


@plan_app.command("resolve")
def plan_resolve(
    plan: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    models: Annotated[
        list[str] | None,
        typer.Option("--model", help="Model key to include. Repeat for multiple models."),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Resolved config output directory."),
    ] = None,
) -> None:
    plan_cfg = load_experiment_plan(plan)
    out_dir = out or Path("runs") / "resolved_configs" / plan_cfg.key
    paths = write_resolved_experiments(plan_cfg, out_dir, model_keys=_model_keys(models))
    for path in paths:
        typer.echo(path)


@plan_app.command("optimize")
def plan_optimize(
    plan: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    models: Annotated[
        list[str] | None,
        typer.Option("--model", help="Model key to optimize. Repeat for multiple models."),
    ] = None,
) -> None:
    experiments = _experiments_from_plan(plan, model_keys=_model_keys(models))
    typer.echo(f"Optimizing {len(experiments)} experiment(s)")
    for exp in experiments:
        _optimize_experiment(exp)


@plan_app.command("final")
def plan_final(
    plan: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    models: Annotated[
        list[str] | None,
        typer.Option("--model", help="Model key to run final seeds for. Repeat for multiple models."),
    ] = None,
) -> None:
    experiments = _experiments_from_plan(plan, model_keys=_model_keys(models))
    typer.echo(f"Running final stage for {len(experiments)} experiment(s)")
    for exp in experiments:
        _final_experiment(exp, best_trial=_default_best_trial_path(exp), out=None)


@plan_app.command("run")
def plan_run(
    plan: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    models: Annotated[
        list[str] | None,
        typer.Option("--model", help="Model key to run. Repeat for multiple models."),
    ] = None,
    all_models: Annotated[
        bool,
        typer.Option("--all", help="Run every model in the plan."),
    ] = False,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Resolved config output directory."),
    ] = None,
) -> None:
    model_keys = _model_keys(models)
    if not model_keys and not all_models:
        raise click.ClickException("plan run requires at least one --model or explicit --all")

    plan_cfg = load_experiment_plan(plan)
    out_dir = out or Path("runs") / "resolved_configs" / plan_cfg.key
    paths = write_resolved_experiments(plan_cfg, out_dir, model_keys=model_keys)
    typer.echo(f"Resolved {len(paths)} config(s) to {out_dir}")

    experiments = expand_experiment_plan(plan_cfg, model_keys=model_keys)
    typer.echo(f"Running optimize + final for {len(experiments)} experiment(s)")
    for exp in experiments:
        _optimize_experiment(exp)
        _final_experiment(exp, best_trial=_default_best_trial_path(exp), out=None)


def _experiments_from_plan(plan: Path, *, model_keys: list[str] | None):
    return expand_experiment_plan(load_experiment_plan(plan), model_keys=model_keys)


def _model_keys(models: list[str] | None) -> list[str] | None:
    return models or None


def _optimize_experiment(exp) -> None:
    try:
        result = ExperimentRunner(exp).optimize()
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    assert exp.tuning is not None
    result_path = save_tuning_result(result, exp.tuning.output_dir / "tuning_result.json")
    typer.echo(f"Saved tuning result to {result_path}")

    if result.best_trial is not None:
        best_path = save_best_trial(result.best_trial, exp.tuning.output_dir / "best_trial.json")
        typer.echo(f"Saved best trial to {best_path}")
    else:
        typer.echo("No completed trials; best_trial.json was not written")


def _final_experiment(exp, *, best_trial: Path, out: Path | None) -> None:
    if not best_trial.exists():
        raise click.ClickException(f"Missing best trial file: {best_trial}")
    best = load_best_trial(best_trial)
    output_dir = exp.final.output_dir or exp.train.output_dir / "final"
    out_path = out or output_dir / "final_results.json"
    runs = ExperimentRunner(exp).final(best, output_dir=output_dir)
    write_final_results(runs, out_path)
    typer.echo(f"Saved final results to {out_path}")


def _default_best_trial_path(exp) -> Path:
    if exp.tuning is None:
        raise click.ClickException(f"Experiment {exp.train.run_key!r} is missing tuning config")
    return exp.tuning.output_dir / "best_trial.json"
