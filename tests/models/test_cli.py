from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from typer.testing import CliRunner

from obj_det.models import ExperimentConfig
from obj_det.models.cli import app
from obj_det.models.logging import CompositeLogger, LocalJsonLogger
from obj_det.models.runner import ExperimentRunner
from obj_det.models.schemas.artifact import ModelArtifact


class DummyAdapter:
    def train(self, train_ds, val_ds, train_cfg, *, epoch_eval_cfg=None, logger=None, log_prefix="train"):
        if logger is not None:
            logger.log_metrics({f"{log_prefix}/loss": 1.25}, step=1)
        return ModelArtifact(
            model_key="m",
            backend="torchvision",
            run_key=train_cfg.run_key,
            classes=train_cfg.classes,
            label_mode=train_cfg.label_mode,
            artifact_path=train_cfg.output_dir,
            checkpoint_path=train_cfg.output_dir / "checkpoint.pt",
        )


class CliTest(unittest.TestCase):
    def test_logger_uses_configured_wandb_and_local_backends(self):
        exp = ExperimentConfig.model_validate(
            {
                "dataset": {"path": "datasets/tiny"},
                "classes": ["car"],
                "preprocess": {"image_size": 32},
                "model": {
                    "key": "m",
                    "backend": "torchvision",
                    "model_name_or_path": "fasterrcnn_resnet50_fpn",
                },
                "train": {"run_key": "r", "output_dir": "runs/r"},
                "eval": {},
                "logging": {
                    "backends": ["local", "wandb"],
                    "wandb": {
                        "project": "study",
                        "entity": "ent",
                        "group": "grp",
                        "name": "run-name",
                        "mode": "offline",
                        "tags": ["tag"],
                    },
                },
            }
        )

        with patch("obj_det.models.logging.factory.WandbLogger") as logger_cls:
            logger = ExperimentRunner(exp)._logger(default_log_path=Path("runs/r/logs/events.jsonl"), run_name="fallback")

        self.assertIsInstance(logger, CompositeLogger)
        self.assertIsInstance(logger.loggers[0], LocalJsonLogger)
        self.assertIs(logger.loggers[1], logger_cls.return_value)
        logger_cls.assert_called_once_with(
            project="study",
            entity="ent",
            group="grp",
            name="run-name",
            mode="offline",
            tags=["tag"],
        )

    def test_logger_is_none_when_disabled(self):
        exp = ExperimentConfig.model_validate(
            {
                "dataset": {"path": "datasets/tiny"},
                "classes": ["car"],
                "preprocess": {"image_size": 32},
                "model": {
                    "key": "m",
                    "backend": "torchvision",
                    "model_name_or_path": "fasterrcnn_resnet50_fpn",
                },
                "train": {"run_key": "r", "output_dir": "runs/r"},
                "eval": {},
            }
        )

        self.assertIsNone(ExperimentRunner(exp)._logger(default_log_path=Path("runs/r/logs/events.jsonl"), run_name="r"))

    def test_runner_run_config_logs_batch_metadata(self):
        exp = ExperimentConfig.model_validate(
            {
                "dataset": {"path": "datasets/tiny"},
                "classes": ["car"],
                "preprocess": {"image_size": 32},
                "model": {
                    "key": "m",
                    "backend": "torchvision",
                    "model_name_or_path": "fasterrcnn_resnet50_fpn",
                },
                "train": {"run_key": "r", "output_dir": "runs/r", "batch_size": 8},
                "eval": {},
            }
        )

        cfg = ExperimentRunner(exp).run_config()

        self.assertEqual(cfg["batch"]["batch_size"], 8)
        self.assertEqual(cfg["batch"]["gradient_accumulation_steps"], 1)
        self.assertEqual(
            cfg["batch"]["effective_batch_size"],
            8 * cfg["batch"]["world_size"] * cfg["batch"]["gradient_accumulation_steps"],
        )

    def test_child_logger_factory_uses_child_name_group_and_path(self):
        exp = ExperimentConfig.model_validate(
            {
                "dataset": {"path": "datasets/tiny"},
                "classes": ["car"],
                "preprocess": {"image_size": 32},
                "model": {
                    "key": "m",
                    "backend": "torchvision",
                    "model_name_or_path": "fasterrcnn_resnet50_fpn",
                },
                "train": {"run_key": "r", "output_dir": "runs/r"},
                "eval": {},
                "logging": {
                    "backends": ["local", "wandb"],
                    "local": {"path": "runs/shared/events.jsonl"},
                    "wandb": {
                        "project": "study",
                        "group": "configured-group",
                        "name": "configured-name",
                        "mode": "offline",
                    },
                },
            }
        )

        with patch("obj_det.models.logging.factory.WandbLogger") as logger_cls:
            factory = ExperimentRunner(exp)._child_logger_factory(wandb_group="child-group")
            logger = factory("child-run", Path("runs/child/logs/events.jsonl"))

        self.assertIsInstance(logger, CompositeLogger)
        self.assertIsInstance(logger.loggers[0], LocalJsonLogger)
        self.assertEqual(logger.loggers[0].path, Path("runs/child/logs/events.jsonl"))
        logger_cls.assert_called_once_with(
            project="study",
            entity=None,
            group="child-group",
            name="child-run",
            mode="offline",
            tags=[],
        )

    def test_train_command_writes_local_log_events(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_path = root / "events.jsonl"
            config = root / "exp.yaml"
            config.write_text(
                "\n".join(
                    [
                        "dataset:",
                        "  path: datasets/tiny",
                        "classes: [car]",
                        "preprocess:",
                        "  image_size: 32",
                        "model:",
                        "  key: m",
                        "  backend: torchvision",
                        "  model_name_or_path: fasterrcnn_resnet50_fpn",
                        "train:",
                        "  run_key: r",
                        f"  output_dir: {root / 'run'}",
                        "eval: {}",
                        "logging:",
                        "  backends: [local]",
                        "  local:",
                        f"    path: {log_path}",
                    ]
                ),
                encoding="utf-8",
            )
            dataset = {"train": [object()], "validation": [object()]}

            with (
                patch("obj_det.models.runner.load_from_disk", return_value=dataset),
                patch("obj_det.models.runner.model_adapter_from_config", return_value=DummyAdapter()),
            ):
                result = CliRunner().invoke(app, ["train", str(config)])

            rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(rows[0]["event"], "start_run")
        self.assertTrue(any(row.get("metrics") == {"train/loss": 1.25} for row in rows))
        self.assertEqual(rows[-1]["event"], "finish_run")

class PlanCliTest(unittest.TestCase):
    def test_models_cli_exposes_plan_commands(self):
        result = CliRunner().invoke(app, ["plan", "--help"])

        self.assertEqual(result.exit_code, 0, result.output)
        for command in ["list", "resolve", "optimize", "final", "run"]:
            self.assertIn(command, result.output)

    def test_plan_list_prints_model_keys_and_backends(self):
        result = CliRunner().invoke(app, ["plan", "list", "configs/plans/hazydet_controlled.yaml"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("yolo26m\tultralytics", result.output)
        self.assertIn("rtdetr_r50vd\thf_trainer", result.output)

    def test_plan_resolve_writes_selected_config(self):
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "resolved"
            result = CliRunner().invoke(
                app,
                [
                    "plan",
                    "resolve",
                    "configs/plans/hazydet_controlled.yaml",
                    "--model",
                    "yolo26m",
                    "--out",
                    str(out_dir),
                ],
            )

            path = out_dir / "yolo26m.yaml"
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("yolo26m.yaml", result.output)
            self.assertTrue(path.exists())

    def test_plan_run_requires_model_or_all(self):
        result = CliRunner().invoke(app, ["plan", "run", "configs/plans/hazydet_controlled.yaml"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("requires at least one --model or explicit --all", result.output)

    def test_plan_optimize_selected_model_saves_results(self):
        from obj_det.models.schemas.tuning import BestTrial, TuningResult

        class FakeRunner:
            def __init__(self, exp):
                self.exp = exp

            def optimize(self):
                return TuningResult(
                    study_name=self.exp.tuning.study_name,
                    best_trial=BestTrial(
                        study_name=self.exp.tuning.study_name,
                        trial_number=0,
                        hparams={"learning_rate": 0.003},
                        metric_name="map_50_95",
                        metric_value=0.1,
                    ),
                )

        saved_paths = []

        def fake_save(model, path):
            saved_paths.append(Path(path))
            return Path(path)

        with (
            patch("obj_det.models.cli.ExperimentRunner", FakeRunner),
            patch("obj_det.models.cli.save_tuning_result", side_effect=fake_save),
            patch("obj_det.models.cli.save_best_trial", side_effect=fake_save),
        ):
            result = CliRunner().invoke(
                app,
                [
                    "plan",
                    "optimize",
                    "configs/plans/hazydet_controlled.yaml",
                    "--model",
                    "yolo26m",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            saved_paths,
            [
                Path("runs/hpo/yolo26m_hazydet_controlled/tuning_result.json"),
                Path("runs/hpo/yolo26m_hazydet_controlled/best_trial.json"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
