from __future__ import annotations

import unittest

from obj_det.models import (
    AugmentationConfig,
    CheckpointConfig,
    DataLoaderConfig,
    EarlyStoppingConfig,
    EvalConfig,
    EvalStrategyConfig,
    ExperimentConfig,
    ExperimentRunner,
    ModelConfig,
    OptimizerConfig,
    PreprocessConfig,
    SearchSpace,
    SchedulerConfig,
    TrainConfig,
)
from obj_det.models.adapters.factory import model_adapter_from_config


class ImportTest(unittest.TestCase):
    def test_schemas_and_factory_import(self):
        cfg = ModelConfig(
            key="fasterrcnn",
            backend="torchvision",
            model_name_or_path="fasterrcnn_resnet50_fpn",
            preprocess=PreprocessConfig(
                resize_mode="shortest_edge", shortest_edge=800, longest_edge=1333
            ),
        )
        exp = ExperimentConfig.model_validate({
            "dataset": {"path": "/tmp/ds"},
            "classes": ["car"],
            "model": {
                "key": "m",
                "backend": "torchvision",
                "model_name_or_path": "fasterrcnn_resnet50_fpn",
                "preprocess": {"resize_mode": "letterbox", "height": 32, "width": 32},
            },
            "train": {"run_key": "r", "output_dir": "/tmp/r"},
            "eval": {},
        })
        self.assertIs(ExperimentRunner(exp).exp, exp)
        adapter = model_adapter_from_config(cfg)
        preprocess = PreprocessConfig(resize_mode="letterbox", height=32, width=32)
        self.assertEqual(adapter.key, "fasterrcnn")
        eval_cfg = EvalConfig(classes=["car"], preprocess=preprocess)
        self.assertEqual(eval_cfg.classes, ["car"])
        self.assertFalse(eval_cfg.compute_per_class)
        self.assertFalse(EvalStrategyConfig().enabled)
        self.assertEqual(DataLoaderConfig(num_workers=2).num_workers, 2)
        self.assertEqual(DataLoaderConfig(decode_backend="opencv", profile_every_n=10).decode_backend, "opencv")
        self.assertEqual(AugmentationConfig(policy="basic").policy, "basic")
        self.assertEqual(OptimizerConfig().name, "adamw")
        self.assertEqual(SchedulerConfig().total_epochs, 50)
        self.assertTrue(CheckpointConfig().keep_all_epoch_checkpoints)
        self.assertEqual(EarlyStoppingConfig().patience, 8)
        self.assertEqual(SearchSpace().params, {})
        self.assertEqual(TrainConfig(run_key="r", classes=["car"], output_dir="/tmp/x", preprocess=preprocess).label_mode, "meta")
        self.assertEqual(
            ExperimentConfig.model_validate({
                "dataset": {"path": "/tmp/ds"},
                "classes": ["car"],
                "model": {
                    "key": "m",
                    "backend": "torchvision",
                    "model_name_or_path": "fasterrcnn_resnet50_fpn",
                    "preprocess": {"resize_mode": "letterbox", "height": 32, "width": 32},
                },
                "train": {"run_key": "r", "output_dir": "/tmp/r"},
                "eval": {},
            }).train.classes,
            ["car"],
        )


if __name__ == "__main__":
    unittest.main()
