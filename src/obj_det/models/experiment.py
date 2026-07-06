from __future__ import annotations

from pathlib import Path
import yaml

from obj_det.models.schemas.artifact import ModelArtifact
from obj_det.models.schemas.config import AugmentationConfig, ModelConfig, PreprocessConfig
from obj_det.models.schemas.experiment import ExperimentConfig
from obj_det.models.schemas.result import EvalResult
from obj_det.models.schemas.tuning import BestTrial, SearchSpace, TuningResult


def load_experiment_config(path: Path) -> ExperimentConfig:
    path = Path(path)
    data = _read_yaml_mapping(path)

    if data.get("preprocess_file") is not None:
        if data.get("preprocess") is not None:
            raise ValueError("Use either preprocess or preprocess_file, not both")
        preprocess_path = _resolve(path.parent, Path(data["preprocess_file"]))
        data["preprocess"] = _read_yaml_mapping(preprocess_path)
        data.pop("preprocess_file")

    if data.get("augmentation_file") is not None:
        if data.get("augmentation") is not None:
            raise ValueError("Use either augmentation or augmentation_file, not both")
        augmentation_path = _resolve(path.parent, Path(data["augmentation_file"]))
        data["augmentation"] = _read_yaml_mapping(augmentation_path)
        data.pop("augmentation_file")

    cfg = ExperimentConfig.model_validate(data)

    if cfg.model_file is not None:
        model_path = _resolve(path.parent, cfg.model_file)
        cfg = cfg.model_copy(update={"model": load_model_config(model_path), "model_file": None})

    if cfg.search_space_file is not None:
        search_path = _resolve(path.parent, cfg.search_space_file)
        cfg = cfg.model_copy(update={"search_space": load_search_space(search_path), "search_space_file": None})

    return cfg


def load_model_config(path: Path) -> ModelConfig:
    return ModelConfig.model_validate(_read_yaml_mapping(Path(path)))


def load_preprocess_config(path: Path) -> PreprocessConfig:
    return PreprocessConfig.model_validate(_read_yaml_mapping(Path(path)))


def load_augmentation_config(path: Path) -> AugmentationConfig:
    return AugmentationConfig.model_validate(_read_yaml_mapping(Path(path)))


def load_search_space(path: Path) -> SearchSpace:
    data = _read_yaml_mapping(Path(path))
    if "params" not in data:
        data = {"params": data}
    return SearchSpace.model_validate(data)


def save_model_artifact(artifact: ModelArtifact, path: Path) -> Path:
    return _write_json_model(artifact, Path(path))


def load_model_artifact(path: Path) -> ModelArtifact:
    return ModelArtifact.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_eval_result(result: EvalResult, path: Path) -> Path:
    return _write_json_model(result, Path(path))


def save_tuning_result(result: TuningResult, path: Path) -> Path:
    return _write_json_model(result, Path(path))


def save_best_trial(best_trial: BestTrial, path: Path) -> Path:
    return _write_json_model(best_trial, Path(path))


def load_best_trial(path: Path) -> BestTrial:
    return BestTrial.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _read_yaml_mapping(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping in {path}")
    return data


def _resolve(base_dir: Path, path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else base_dir / path


def _write_json_model(model, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path
