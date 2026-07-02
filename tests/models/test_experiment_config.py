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
                "train": {
                    "run_key": "r",
                    "label_mode": "meta",
                    "image_size": 320,
                    "output_dir": "runs/r",
                },
                "eval": {},
                "predict": {},
            }
        )

        self.assertEqual(cfg.train.classes, ["car", "person"])
        self.assertEqual(cfg.eval.classes, ["car", "person"])
        self.assertEqual(cfg.predict.classes, ["car", "person"])
        self.assertEqual(cfg.eval.label_mode, "meta")
        self.assertEqual(cfg.predict.image_size, 320)

    def test_relative_model_and_search_space_files_load(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "models").mkdir()
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
                        "train:",
                        "  run_key: r",
                        "  output_dir: runs/r",
                        "eval: {}",
                        "search_space_file: ../spaces/s.yaml",
                    ]
                ),
                encoding="utf-8",
            )

            cfg = load_experiment_config(exp_path)

        self.assertEqual(cfg.model.key, "m")
        self.assertIn("learning_rate", cfg.search_space.params)

    def test_rejects_duplicate_model_or_search_space_source(self):
        base = {
            "dataset": {"path": "datasets/tiny"},
            "classes": ["car"],
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
                    "search_space": {"params": {}},
                    "search_space_file": "s.yaml",
                }
            )


if __name__ == "__main__":
    unittest.main()
