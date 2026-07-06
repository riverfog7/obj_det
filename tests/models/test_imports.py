from __future__ import annotations

import unittest

from obj_det.models import (
    AugmentationConfig,
    DataLoaderConfig,
    EvalConfig,
    EvalStrategyConfig,
    ExperimentConfig,
    ModelConfig,
    PreprocessConfig,
    SearchSpace,
    TrainConfig,
)
from obj_det.models.adapters.factory import model_adapter_from_config


class ImportTest(unittest.TestCase):
    def test_schemas_and_factory_import(self):
        cfg = ModelConfig(
            key="fasterrcnn",
            backend="torchvision",
            model_name_or_path="fasterrcnn_resnet50_fpn",
        )
        adapter = model_adapter_from_config(cfg)
        preprocess = PreprocessConfig(image_size=32)
        self.assertEqual(adapter.key, "fasterrcnn")
        eval_cfg = EvalConfig(classes=["car"], preprocess=preprocess)
        self.assertEqual(eval_cfg.classes, ["car"])
        self.assertFalse(eval_cfg.compute_per_class)
        self.assertFalse(EvalStrategyConfig().enabled)
        self.assertEqual(DataLoaderConfig(num_workers=2).num_workers, 2)
        self.assertEqual(AugmentationConfig(policy="basic").policy, "basic")
        self.assertEqual(SearchSpace().params, {})
        self.assertEqual(TrainConfig(run_key="r", classes=["car"], output_dir="/tmp/x", preprocess=preprocess).label_mode, "meta")
        self.assertEqual(
            ExperimentConfig.model_validate({
                "dataset": {"path": "/tmp/ds"},
                "classes": ["car"],
                "preprocess": {"image_size": 32},
                "model": {
                    "key": "m",
                    "backend": "torchvision",
                    "model_name_or_path": "fasterrcnn_resnet50_fpn",
                },
                "train": {"run_key": "r", "output_dir": "/tmp/r"},
                "eval": {},
            }).train.classes,
            ["car"],
        )


if __name__ == "__main__":
    unittest.main()
