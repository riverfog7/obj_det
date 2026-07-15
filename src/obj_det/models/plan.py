from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from obj_det.models.experiment import (
    _read_yaml_mapping,
    load_augmentation_config,
    load_model_config,
    load_search_space,
)
from obj_det.models.runner import ExperimentRunner
from obj_det.models.schemas.experiment import ExperimentConfig
from obj_det.models.schemas.plan import (
    ClassSpaceConfig,
    DatasetRefConfig,
    ExperimentPlanConfig,
    ModelGroupConfig,
    RecipeConfig,
)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_experiment_plan(path: Path) -> ExperimentPlanConfig:
    path = Path(path)
    plan = ExperimentPlanConfig.model_validate(_read_yaml_mapping(path))
    plan._base_dir = path.parent
    return plan


def load_and_expand_experiment_plan(
    path: Path,
    *,
    model_keys: list[str] | None = None,
) -> list[ExperimentConfig]:
    return expand_experiment_plan(load_experiment_plan(path), model_keys=model_keys)


def expand_experiment_plan(
    plan: ExperimentPlanConfig,
    *,
    model_keys: list[str] | None = None,
) -> list[ExperimentConfig]:
    base_dir = getattr(plan, "_base_dir", Path("."))
    dataset = _load_dataset_ref(_resolve(base_dir, plan.dataset_file))
    class_space = _load_class_space(_resolve(base_dir, plan.class_space_file))
    recipe_path = _resolve(base_dir, plan.recipe_file)
    recipe = _load_recipe(recipe_path)
    augmentation = None
    if recipe.augmentation_file is not None:
        augmentation = load_augmentation_config(_resolve(recipe_path.parent, recipe.augmentation_file))
    recipe_search_space = None
    if recipe.search_space_file is not None:
        recipe_search_space = load_search_space(_resolve(recipe_path.parent, recipe.search_space_file))

    selected = set(model_keys or [])
    experiments: list[ExperimentConfig] = []

    for model_path in _model_paths(plan, base_dir):
        model = load_model_config(model_path)
        if selected and model.key not in selected:
            continue
        if model.backend not in plan.backend_defaults:
            raise ValueError(
                f"Plan {plan.key!r} has no backend_defaults for backend {model.backend!r} "
                f"used by model {model.key!r}"
            )
        backend_defaults = plan.backend_defaults[model.backend]
        model_override = plan.model_overrides.get(model.key, {})
        if recipe_search_space is not None:
            _reject_recipe_search_space_override(
                recipe_file=recipe_path,
                model_key=model.key,
                backend_defaults=backend_defaults,
                model_override=model_override,
            )

        data = _base_experiment_dict(
            dataset=dataset,
            class_space=class_space,
            recipe=recipe,
            model=model.model_dump(mode="json"),
            augmentation=augmentation.model_dump(mode="json") if augmentation is not None else None,
            search_space=(
                recipe_search_space.model_dump(mode="json") if recipe_search_space is not None else None
            ),
        )
        data = deep_merge(data, backend_defaults)
        data = deep_merge(data, model_override)
        data = _apply_run_template(data, plan=plan, dataset=dataset, model_key=model.key, protocol=recipe.protocol)
        data = _resolve_search_space(data, base_dir)
        experiments.append(ExperimentConfig.model_validate(data))

    if selected:
        found = {exp.model.key for exp in experiments if exp.model is not None}
        missing = selected - found
        if missing:
            raise ValueError(f"Plan {plan.key!r} does not define model keys: {sorted(missing)}")

    return experiments


def write_resolved_experiments(
    plan: ExperimentPlanConfig,
    out_dir: Path,
    *,
    model_keys: list[str] | None = None,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for exp in expand_experiment_plan(plan, model_keys=model_keys):
        if exp.model is None:
            raise ValueError("Resolved experiment is missing model")
        path = out_dir / f"{exp.model.key}.yaml"
        data = exp.model_dump(mode="json", exclude_none=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        paths.append(path)
    return paths


class ExperimentPlanRunner:
    """Thin convenience wrapper around expanded ExperimentConfig objects."""

    def __init__(self, plan: ExperimentPlanConfig):
        self.plan = plan

    def experiments(self, model_keys: list[str] | None = None) -> list[ExperimentConfig]:
        return expand_experiment_plan(self.plan, model_keys=model_keys)

    def optimize_all(self, model_keys: list[str] | None = None):
        return [ExperimentRunner(exp).optimize() for exp in self.experiments(model_keys)]

    def final_all(self, best_trials: dict[str, Any], model_keys: list[str] | None = None):
        runs = []
        for exp in self.experiments(model_keys):
            if exp.model is None:
                raise ValueError("Resolved experiment is missing model")
            if exp.model.key not in best_trials:
                raise KeyError(f"Missing best trial for model {exp.model.key!r}")
            runs.extend(ExperimentRunner(exp).final(best_trials[exp.model.key]))
        return runs


def _load_dataset_ref(path: Path) -> DatasetRefConfig:
    return DatasetRefConfig.model_validate(_read_yaml_mapping(path))


def _load_class_space(path: Path) -> ClassSpaceConfig:
    return ClassSpaceConfig.model_validate(_read_yaml_mapping(path))


def _load_recipe(path: Path) -> RecipeConfig:
    return RecipeConfig.model_validate(_read_yaml_mapping(path))


def _load_model_group(path: Path) -> ModelGroupConfig:
    return ModelGroupConfig.model_validate(_read_yaml_mapping(path))


def _model_paths(plan: ExperimentPlanConfig, base_dir: Path) -> list[Path]:
    paths = [_resolve(base_dir, path) for path in plan.model_files]
    if plan.model_group_file is not None:
        group_path = _resolve(base_dir, plan.model_group_file)
        group = _load_model_group(group_path)
        paths.extend(_resolve(group_path.parent, path) for path in group.models)
    return paths


def _base_experiment_dict(
    *,
    dataset: DatasetRefConfig,
    class_space: ClassSpaceConfig,
    recipe: RecipeConfig,
    model: dict[str, Any],
    augmentation: dict[str, Any] | None,
    search_space: dict[str, Any] | None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "dataset": {
            "path": dataset.path,
            "train_split": dataset.train_split,
            "val_split": dataset.val_split,
            "test_split": dataset.test_split,
        },
        "classes": class_space.classes,
        "model": model,
        "train": deepcopy(recipe.train),
        "eval": deepcopy(recipe.eval),
        "predict": deepcopy(recipe.predict),
        "tuning": deepcopy(recipe.tuning),
        "final": deepcopy(recipe.final),
        "logging": deepcopy(recipe.logging),
    }
    if augmentation is not None:
        data["augmentation"] = augmentation
    if search_space is not None:
        data["search_space"] = search_space

    data["train"].setdefault("protocol", recipe.protocol)
    data["train"].setdefault("label_mode", class_space.label_mode)
    data["eval"].setdefault("label_mode", class_space.label_mode)
    data["predict"].setdefault("label_mode", class_space.label_mode)
    return data


def _apply_run_template(
    data: dict[str, Any],
    *,
    plan: ExperimentPlanConfig,
    dataset: DatasetRefConfig,
    model_key: str,
    protocol: str,
) -> dict[str, Any]:
    data = deepcopy(data)
    values = {
        "model_key": model_key,
        "dataset_key": dataset.key,
        "protocol": protocol,
        "plan_key": plan.key,
    }
    template = plan.run_template

    data.setdefault("train", {})["run_key"] = _render(template.run_key, values)
    data["train"]["output_dir"] = _render(template.output_dir, values)

    data.setdefault("tuning", {})["study_name"] = _render(template.tuning_study_name, values)
    data["tuning"]["output_dir"] = _render(template.tuning_output_dir, values)

    data.setdefault("final", {})["output_dir"] = _render(template.final_output_dir, values)

    logging = data.setdefault("logging", {})
    wandb = logging.setdefault("wandb", {})
    wandb["project"] = _render(template.wandb_project, values)
    if plan.tags:
        existing_tags = wandb.get("tags") or []
        wandb["tags"] = list(dict.fromkeys([*existing_tags, *plan.tags]))

    return data


def _resolve_search_space(data: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    data = deepcopy(data)
    search_space_file = data.pop("search_space_file", None)
    if search_space_file is not None:
        if data.get("search_space") is not None:
            raise ValueError("Use either search_space or search_space_file, not both")
        search_space = load_search_space(_resolve(base_dir, Path(search_space_file)))
        data["search_space"] = search_space.model_dump(mode="json")
    return data


def _reject_recipe_search_space_override(
    *,
    recipe_file: Path,
    model_key: str,
    backend_defaults: dict[str, Any],
    model_override: dict[str, Any],
) -> None:
    for source_name, values in (
        ("backend_defaults", backend_defaults),
        ("model_overrides", model_override),
    ):
        if values.get("search_space") is not None or values.get("search_space_file") is not None:
            raise ValueError(
                f"Recipe {recipe_file!s} defines search_space_file, but {source_name} also defines "
                f"a search space for model {model_key!r}"
            )


def _resolve(base_dir: Path, path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else base_dir / path


def _render(template: str, values: dict[str, str]) -> str:
    try:
        return template.format(**values)
    except KeyError as exc:
        raise ValueError(f"Unknown run template variable {exc.args[0]!r} in {template!r}") from exc
