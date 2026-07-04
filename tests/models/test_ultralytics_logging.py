from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from obj_det.models.adapters.ultralytics import UltralyticsDetectionAdapter
from obj_det.models.schemas import ModelConfig


class RecordingLogger:
    def __init__(self):
        self.events = []

    def log_metrics(self, metrics, step=None):
        self.events.append((metrics, step))


class FakeDetectionTrainer:
    loss_names = ("box_loss", "cls_loss")

    def __init__(self, *args, **kwargs):
        self.callbacks = {}
        self.stop = False
        self.epoch = 0
        self.tloss = torch.tensor([1.0, 2.0])
        self.optimizer = SimpleNamespace(param_groups=[{"lr": 0.01}])

    def run_callbacks(self, event: str):
        pass

    def label_loss_items(self, loss_items, prefix="train"):
        return {f"{prefix}/{name}": float(value) for name, value in zip(self.loss_names, loss_items)}


class UltralyticsLoggingTest(unittest.TestCase):
    def test_custom_trainer_logs_every_configured_step_and_final_short_step(self):
        adapter = UltralyticsDetectionAdapter(
            ModelConfig(key="yolo", backend="ultralytics", model_name_or_path="yolo11n.pt")
        )
        logger = RecordingLogger()
        trainer_cls = adapter._trainer_class(FakeDetectionTrainer)
        trainer = trainer_cls(
            train_source=None,
            val_source=None,
            transform=None,
            loader_cfg=None,
            classes=["car"],
            max_steps=None,
            logger=logger,
            log_prefix="train",
            logging_steps=2,
        )

        trainer.run_callbacks("on_train_batch_end")
        self.assertEqual(logger.events, [])

        trainer.run_callbacks("on_train_batch_end")
        self.assertEqual(len(logger.events), 1)
        self.assertEqual(logger.events[0][1], 2)
        self.assertEqual(logger.events[0][0]["train/box_loss"], 1.0)
        self.assertEqual(logger.events[0][0]["train/lr/pg0"], 0.01)

        trainer.tloss = torch.tensor([0.5, 1.5])
        trainer.run_callbacks("on_train_batch_end")
        self.assertEqual(len(logger.events), 1)

        trainer.run_callbacks("on_train_end")
        self.assertEqual(len(logger.events), 2)
        self.assertEqual(logger.events[1][1], 3)
        self.assertEqual(logger.events[1][0]["train/box_loss"], 0.5)


if __name__ == "__main__":
    unittest.main()
