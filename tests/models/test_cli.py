from __future__ import annotations

import unittest
from unittest.mock import patch

from obj_det.models import ExperimentConfig
from obj_det.models.cli import _tuning_logger


class CliTest(unittest.TestCase):
    def test_tuning_logger_uses_wandb_when_enabled(self):
        exp = ExperimentConfig.model_validate(
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
                    "study_name": "study",
                    "output_dir": "runs/hpo",
                    "log_to_wandb": True,
                },
            }
        )

        with patch("obj_det.models.cli.WandbLogger") as logger_cls:
            logger = _tuning_logger(exp)

        self.assertIs(logger, logger_cls.return_value)
        logger_cls.assert_called_once_with(project="study")

    def test_tuning_logger_is_none_when_disabled(self):
        exp = ExperimentConfig.model_validate(
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
                    "study_name": "study",
                    "output_dir": "runs/hpo",
                    "log_to_wandb": False,
                },
            }
        )

        self.assertIsNone(_tuning_logger(exp))


if __name__ == "__main__":
    unittest.main()
