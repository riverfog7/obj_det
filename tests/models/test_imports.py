from __future__ import annotations

import unittest

from obj_det.models import EvalConfig, ExperimentConfig, ModelConfig, SearchSpace, TrainConfig
from obj_det.models.adapters.factory import model_adapter_from_config


class ImportTest(unittest.TestCase):
    def test_schemas_and_factory_import(self):
        cfg = ModelConfig(
            key="fasterrcnn",
            backend="torchvision",
            model_name_or_path="fasterrcnn_resnet50_fpn",
        )
        adapter = model_adapter_from_config(cfg)
        self.assertEqual(adapter.key, "fasterrcnn")
        self.assertEqual(EvalConfig(classes=["car"]).classes, ["car"])
        self.assertEqual(SearchSpace().params, {})
        self.assertEqual(TrainConfig(run_key="r", classes=["car"], output_dir="/tmp/x").label_mode, "meta")
        self.assertEqual(
            ExperimentConfig.model_validate({
                "dataset": {"path": "/tmp/ds"},
                "classes": ["car"],
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
