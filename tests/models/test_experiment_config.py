from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import ValidationError

from obj_det.models.experiment import load_experiment_config
from obj_det.models.schemas.experiment import ExperimentConfig


class ExperimentConfigTest(unittest.TestCase):
    def test_classes_are_injected_into_train_eval_predict(self):
        cfg = ExperimentConfig.model_validate(
            {
                "dataset": {"path": "datasets/tiny"},
                "classes": ["car", "person"],
                "model": {
                    "key": "m",
                    "backend": "torchvision",
                    "model_name_or_path": "fasterrcnn_resnet50_fpn",
                },
                "transform": {
                    "image_size": 320,
                    "horizontal_flip_p": 0.5,
                    "color_jitter_strength": 0.1,
                },
                "train": {
                    "run_key": "r",
                    "label_mode": "meta",
                    "output_dir": "runs/r",
                    "eval_strategy": {
                        "enabled": True,
                        "every_epochs": 1,
                    },
                    "loader": {
                        "num_workers": 2,
                        "pin_memory": True,
                        "persistent_workers": True,
                        "prefetch_factor": 2,
                        "predecode_images": True,
                    },
                },
                "eval": {},
                "predict": {},
            }
        )

        self.assertEqual(cfg.train.classes, ["car", "person"])
        self.assertEqual(cfg.eval.classes, ["car", "person"])
        self.assertEqual(cfg.predict.classes, ["car", "person"])
        self.assertEqual(cfg.eval.label_mode, "meta")
        self.assertEqual(cfg.predict.transform.image_size, 320)
        self.assertEqual(cfg.train.transform.horizontal_flip_p, 0.5)
        self.assertTrue(cfg.train.eval_strategy.enabled)
        self.assertEqual(cfg.train.eval_strategy.every_epochs, 1)
        self.assertEqual(cfg.train.loader.num_workers, 2)
        self.assertTrue(cfg.train.loader.predecode_images)

    def test_relative_model_transform_and_search_space_files_load(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "models").mkdir()
            (root / "transforms").mkdir()
            (root / "spaces").mkdir()
            (root / "experiments").mkdir()
            (root / "models" / "m.yaml").write_text(
                "\n".join(
                    [
                        "key: m",
                        "backend: torchvision",
                        "model_name_or_path: fasterrcnn_resnet50_fpn",
                    ]
                ),
                encoding="utf-8",
            )
            (root / "transforms" / "t.yaml").write_text(
                "\n".join(
                    [
                        "image_size: 64",
                        "horizontal_flip_p: 0.0",
                        "color_jitter_strength: 0.0",
                    ]
                ),
                encoding="utf-8",
            )
            (root / "spaces" / "s.yaml").write_text(
                "\n".join(
                    [
                        "params:",
                        "  learning_rate:",
                        "    type: float",
                        "    low: 0.0001",
                        "    high: 0.001",
                    ]
                ),
                encoding="utf-8",
            )
            exp_path = root / "experiments" / "exp.yaml"
            exp_path.write_text(
                "\n".join(
                    [
                        "dataset:",
                        "  path: datasets/tiny",
                        "classes: [car]",
                        "model_file: ../models/m.yaml",
                        "transform_file: ../transforms/t.yaml",
                        "train:",
                        "  run_key: r",
                        "  output_dir: runs/r",
                        "  loader:",
                        "    num_workers: 3",
                        "    predecode_images: true",
                        "eval: {}",
                        "search_space_file: ../spaces/s.yaml",
                    ]
                ),
                encoding="utf-8",
            )

            cfg = load_experiment_config(exp_path)

        self.assertEqual(cfg.model.key, "m")
        self.assertEqual(cfg.train.transform.image_size, 64)
        self.assertEqual(cfg.train.loader.num_workers, 3)
        self.assertTrue(cfg.train.loader.predecode_images)
        self.assertIn("learning_rate", cfg.search_space.params)

    def test_rejects_invalid_loader_config(self):
        with self.assertRaises(ValidationError):
            ExperimentConfig.model_validate(
                {
                    "dataset": {"path": "datasets/tiny"},
                    "classes": ["car"],
                    "transform": {"image_size": 32},
                    "model": {
                        "key": "m",
                        "backend": "torchvision",
                        "model_name_or_path": "fasterrcnn_resnet50_fpn",
                    },
                    "train": {
                        "run_key": "r",
                        "output_dir": "runs/r",
                        "loader": {"num_workers": -1},
                    },
                    "eval": {},
                }
            )

        with self.assertRaises(ValidationError):
            ExperimentConfig.model_validate(
                {
                    "dataset": {"path": "datasets/tiny"},
                    "classes": ["car"],
                    "transform": {"image_size": 32},
                    "model": {
                        "key": "m",
                        "backend": "torchvision",
                        "model_name_or_path": "fasterrcnn_resnet50_fpn",
                    },
                    "train": {
                        "run_key": "r",
                        "output_dir": "runs/r",
                        "loader": {"prefetch_factor": 0},
                    },
                    "eval": {},
                }
            )

    def test_rejects_removed_train_fields(self):
        removed_fields = {
            "gradient_accumulation_steps": 2,
            "effective_batch_size": 16,
            "per_device_batch_size": 8,
            "eval_metric": "map_50_95",
            "eval_every_epochs": 1,
        }

        for field, value in removed_fields.items():
            with self.subTest(field=field), self.assertRaises(ValidationError):
                ExperimentConfig.model_validate(
                    {
                        "dataset": {"path": "datasets/tiny"},
                        "classes": ["car"],
                        "transform": {"image_size": 32},
                        "model": {
                            "key": "m",
                            "backend": "torchvision",
                            "model_name_or_path": "fasterrcnn_resnet50_fpn",
                        },
                        "train": {
                            "run_key": "r",
                            "output_dir": "runs/r",
                            field: value,
                        },
                        "eval": {},
                    }
                )

    def test_tuning_log_to_wandb_loads(self):
        cfg = ExperimentConfig.model_validate(
            {
                "dataset": {"path": "datasets/tiny"},
                "classes": ["car"],
                "transform": {"image_size": 32},
                "model": {
                    "key": "m",
                    "backend": "torchvision",
                    "model_name_or_path": "fasterrcnn_resnet50_fpn",
                },
                "train": {"run_key": "r", "output_dir": "runs/r"},
                "eval": {},
                "tuning": {
                    "study_name": "s",
                    "output_dir": "runs/hpo",
                    "log_to_wandb": True,
                },
            }
        )

        self.assertTrue(cfg.tuning.log_to_wandb)

    def test_rejects_duplicate_model_transform_or_search_space_source(self):
        base = {
            "dataset": {"path": "datasets/tiny"},
            "classes": ["car"],
            "transform": {"image_size": 32},
            "train": {"run_key": "r", "output_dir": "runs/r"},
            "eval": {},
        }
        with self.assertRaises(ValidationError):
            ExperimentConfig.model_validate(
                {
                    **base,
                    "model": {
                        "key": "m",
                        "backend": "torchvision",
                        "model_name_or_path": "fasterrcnn_resnet50_fpn",
                    },
                    "model_file": "m.yaml",
                }
            )

        with self.assertRaises(ValidationError):
            ExperimentConfig.model_validate(
                {
                    **base,
                    "model": {
                        "key": "m",
                        "backend": "torchvision",
                        "model_name_or_path": "fasterrcnn_resnet50_fpn",
                    },
                    "transform_file": "t.yaml",
                }
            )

        with self.assertRaises(ValidationError):
            ExperimentConfig.model_validate(
                {
                    **base,
                    "model": {
                        "key": "m",
                        "backend": "torchvision",
                        "model_name_or_path": "fasterrcnn_resnet50_fpn",
                    },
                    "search_space": {"params": {}},
                    "search_space_file": "s.yaml",
                }
            )


if __name__ == "__main__":
    unittest.main()
