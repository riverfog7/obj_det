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
                "preprocess": {"image_size": 320},
                "augmentation": {
                    "policy": "basic",
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
        self.assertEqual(cfg.predict.preprocess.image_size, 320)
        self.assertEqual(cfg.train.augmentation.horizontal_flip_p, 0.5)
        self.assertTrue(cfg.train.eval_strategy.enabled)
        self.assertEqual(cfg.train.eval_strategy.every_epochs, 1)
        self.assertEqual(cfg.train.loader.num_workers, 2)
        self.assertTrue(cfg.train.loader.predecode_images)
        self.assertEqual(cfg.train.logging_steps, 10)

    def test_relative_model_preprocess_augmentation_and_search_space_files_load(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "models").mkdir()
            (root / "preprocess").mkdir()
            (root / "augmentations").mkdir()
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
            (root / "preprocess" / "p.yaml").write_text("image_size: 64\n", encoding="utf-8")
            (root / "augmentations" / "a.yaml").write_text(
                "\n".join(
                    [
                        "policy: basic",
                        "horizontal_flip_p: 0.5",
                        "color_jitter_strength: 0.1",
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
                        "preprocess_file: ../preprocess/p.yaml",
                        "augmentation_file: ../augmentations/a.yaml",
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
        self.assertEqual(cfg.train.preprocess.image_size, 64)
        self.assertEqual(cfg.train.loader.num_workers, 3)
        self.assertTrue(cfg.train.loader.predecode_images)
        self.assertIn("learning_rate", cfg.search_space.params)

    def test_rejects_invalid_loader_config(self):
        with self.assertRaises(ValidationError):
            ExperimentConfig.model_validate(
                {
                    "dataset": {"path": "datasets/tiny"},
                    "classes": ["car"],
                    "preprocess": {"image_size": 32},
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
                    "preprocess": {"image_size": 32},
                    "model": {
                        "key": "m",
                        "backend": "torchvision",
                        "model_name_or_path": "fasterrcnn_resnet50_fpn",
                    },
                    "train": {
                        "run_key": "r",
                        "output_dir": "runs/r",
                        "logging_steps": 0,
                    },
                    "eval": {},
                }
            )

        with self.assertRaises(ValidationError):
            ExperimentConfig.model_validate(
                {
                    "dataset": {"path": "datasets/tiny"},
                    "classes": ["car"],
                    "preprocess": {"image_size": 32},
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
                        "preprocess": {"image_size": 32},
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

    def test_top_level_logging_config_loads(self):
        cfg = ExperimentConfig.model_validate(
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
                "tuning": {
                    "study_name": "s",
                    "output_dir": "runs/hpo",
                },
                "logging": {
                    "backends": ["local", "wandb"],
                    "wandb": {
                        "project": "obj-det-tests",
                        "mode": "offline",
                        "tags": ["unit"],
                    },
                },
            }
        )

        self.assertEqual(cfg.logging.backends, ["local", "wandb"])
        self.assertEqual(cfg.logging.wandb.project, "obj-det-tests")
        self.assertEqual(cfg.logging.wandb.mode, "offline")
        self.assertEqual(cfg.logging.wandb.tags, ["unit"])

    def test_rejects_duplicate_model_preprocess_augmentation_or_search_space_source(self):
        base = {
            "dataset": {"path": "datasets/tiny"},
            "classes": ["car"],
            "preprocess": {"image_size": 32},
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
                    "preprocess_file": "p.yaml",
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
