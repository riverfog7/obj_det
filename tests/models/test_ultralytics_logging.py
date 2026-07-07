from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from obj_det.models.adapters.ultralytics import UltralyticsDetectionAdapter
from obj_det.models.schemas import DataLoaderConfig, PreprocessConfig, ModelConfig, TrainConfig


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
            train_transform=None,
            eval_transform=None,
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

    def test_custom_trainer_uses_separate_train_and_eval_transforms(self):
        adapter = UltralyticsDetectionAdapter(
            ModelConfig(key="yolo", backend="ultralytics", model_name_or_path="yolo11n.pt")
        )
        trainer_cls = adapter._trainer_class(FakeDetectionTrainer)
        train_transform = object()
        eval_transform = object()
        trainer = trainer_cls(
            train_source=[object()],
            val_source=[object()],
            train_transform=train_transform,
            eval_transform=eval_transform,
            loader_cfg=DataLoaderConfig(num_workers=0, pin_memory=False),
            classes=["car"],
            max_steps=None,
            logger=None,
            log_prefix="train",
            logging_steps=100,
        )

        train_loader = trainer.get_dataloader("unused", batch_size=1, mode="train")
        val_loader = trainer.get_dataloader("unused", batch_size=1, mode="val")

        self.assertIs(train_loader.dataset.transform, train_transform)
        self.assertIs(val_loader.dataset.transform, eval_transform)

    def test_custom_trainer_refuses_validation_without_val_source(self):
        adapter = UltralyticsDetectionAdapter(
            ModelConfig(key="yolo", backend="ultralytics", model_name_or_path="yolo11n.pt")
        )
        trainer_cls = adapter._trainer_class(FakeDetectionTrainer)
        trainer = trainer_cls(
            train_source=[object()],
            val_source=None,
            train_transform=object(),
            eval_transform=object(),
            loader_cfg=DataLoaderConfig(num_workers=0, pin_memory=False),
            classes=["car"],
            max_steps=None,
            logger=None,
            log_prefix="train",
            logging_steps=100,
        )

        with self.assertRaisesRegex(RuntimeError, "validation source is disabled"):
            trainer.get_dataloader("unused", batch_size=1, mode="val")

    def test_artifact_uses_last_checkpoint_for_external_eval(self):
        adapter = UltralyticsDetectionAdapter(
            ModelConfig(key="yolo", backend="ultralytics", model_name_or_path="yolo11n.pt")
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            last = root / "last.pt"
            best = root / "best.pt"
            last.write_text("last", encoding="utf-8")
            best.write_text("best", encoding="utf-8")
            trainer = SimpleNamespace(
                save_dir=root,
                last=last,
                best=best,
                args=SimpleNamespace(batch=8),
            )
            artifact = adapter._artifact_from_trainer(
                trainer,
                TrainConfig(
                    run_key="r",
                    classes=["car"],
                    output_dir=root,
                    preprocess=PreprocessConfig(image_size=64),
                ),
            )

        self.assertEqual(artifact.checkpoint_path, last)
        self.assertEqual(artifact.meta["ultralytics_best"], str(best))
        self.assertEqual(artifact.meta["checkpoint_selection"], "last_external_eval")


if __name__ == "__main__":
    unittest.main()
