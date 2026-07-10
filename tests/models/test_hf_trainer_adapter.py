from __future__ import annotations

import unittest
from pathlib import Path

from obj_det.models.adapters.hf_trainer import HFTrainerDetectionAdapter
from obj_det.models.schemas import ModelConfig, PreprocessConfig, TrainConfig


class HFTrainerAdapterTest(unittest.TestCase):
    def setUp(self):
        self.adapter = HFTrainerDetectionAdapter(
            ModelConfig(
                key="hf",
                backend="hf_trainer",
                model_name_or_path="dummy/model",
            )
        )

    def test_controlled_training_args_disable_native_schedule_and_metric_selection(self):
        cfg = TrainConfig(
            run_key="r",
            classes=["car"],
            output_dir=Path("runs/test"),
            preprocess=PreprocessConfig(image_size=32),
            hparams={"learning_rate": 3.0e-4},
        )

        args = self.adapter._training_args(cfg, epoch_eval_enabled=True)

        self.assertEqual(args.learning_rate, 3.0e-4)
        self.assertEqual(str(args.lr_scheduler_type), "SchedulerType.CONSTANT")
        self.assertEqual(str(args.eval_strategy), "IntervalStrategy.NO")
        self.assertEqual(str(args.save_strategy), "SaveStrategy.EPOCH")
        self.assertFalse(args.load_best_model_at_end)

    def test_ecosystem_training_args_preserve_legacy_scheduler_hparams(self):
        cfg = TrainConfig(
            run_key="r",
            classes=["car"],
            output_dir=Path("runs/test"),
            preprocess=PreprocessConfig(image_size=32),
            protocol="ecosystem",
            eval_strategy={"enabled": True},
            hparams={
                "learning_rate": 2.0e-5,
                "weight_decay": 0.02,
                "warmup_ratio": 0.1,
                "lr_scheduler_type": "cosine",
            },
        )

        args = self.adapter._training_args(cfg)

        self.assertEqual(args.learning_rate, 2.0e-5)
        self.assertEqual(args.weight_decay, 0.02)
        self.assertEqual(args.warmup_ratio, 0.1)
        self.assertEqual(str(args.lr_scheduler_type), "SchedulerType.COSINE")
        self.assertEqual(str(args.eval_strategy), "IntervalStrategy.EPOCH")
        self.assertEqual(str(args.save_strategy), "SaveStrategy.EPOCH")


if __name__ == "__main__":
    unittest.main()
