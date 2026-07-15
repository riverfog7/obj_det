from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from datasets import Dataset

from obj_det.models.adapters.ultralytics import _NoOpEpochScheduler, UltralyticsDetectionAdapter
from obj_det.models.schemas import DataLoaderConfig, EvalConfig, ModelConfig, PredictConfig, PreprocessConfig, TrainConfig
from obj_det.models.schemas.artifact import ModelArtifact
from obj_det.models.training import CheckpointState, MAX_GRAD_NORM

from .helpers import image_bytes, row


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

    def optimizer_step(self):
        return None

class UltralyticsLoggingTest(unittest.TestCase):
    def test_epoch_scheduler_reset_does_not_rewind_optimizer_step_schedule(self):
        optimizer_step_scheduler = SimpleNamespace(last_epoch=0)
        wrapper = _NoOpEpochScheduler(optimizer_step_scheduler)

        wrapper.last_epoch = -1

        self.assertEqual(optimizer_step_scheduler.last_epoch, 0)

    def test_custom_trainer_logs_every_configured_step_and_final_short_step(self):
        adapter = UltralyticsDetectionAdapter(
            ModelConfig(key="yolo", backend="ultralytics", detector_pretraining_dataset="coco", model_name_or_path="yolo11n.pt", preprocess=PreprocessConfig(resize_mode="letterbox", height=64, width=64))
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
            ModelConfig(key="yolo", backend="ultralytics", detector_pretraining_dataset="coco", model_name_or_path="yolo11n.pt", preprocess=PreprocessConfig(resize_mode="letterbox", height=64, width=64))
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

    def test_custom_trainer_returns_empty_validation_loader_without_val_source(self):
        adapter = UltralyticsDetectionAdapter(
            ModelConfig(key="yolo", backend="ultralytics", detector_pretraining_dataset="coco", model_name_or_path="yolo11n.pt", preprocess=PreprocessConfig(resize_mode="letterbox", height=64, width=64))
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

        val_loader = trainer.get_dataloader("unused", batch_size=1, mode="val")

        self.assertEqual(len(val_loader.dataset), 0)
        self.assertEqual(len(val_loader), 0)

    def test_evaluate_drops_reversed_prediction_box(self):
        class FakeYOLO:
            def __init__(self, checkpoint):
                pass

            def predict(self, **kwargs):
                boxes = SimpleNamespace(
                    xyxy=torch.tensor(
                        [
                            [10.0, 2.0, 9.9281005859375, 10.0],
                            [-1.0, 4.0, 12.0, 16.0],
                        ]
                    ),
                    conf=torch.tensor([0.9, 0.8]),
                    cls=torch.tensor([0.0, 0.0]),
                )
                return [SimpleNamespace(boxes=boxes)]

        fake_ultralytics = ModuleType("ultralytics")
        fake_ultralytics.YOLO = FakeYOLO
        adapter = UltralyticsDetectionAdapter(
            ModelConfig(key="yolo", backend="ultralytics", detector_pretraining_dataset="coco", model_name_or_path="unused.pt", preprocess=PreprocessConfig(resize_mode="letterbox", height=32, width=32))
        )
        artifact = ModelArtifact(
            model_key=adapter.key,
            backend=adapter.backend,
            run_key="r",
            classes=["car"],
            label_mode="meta",
        )

        with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
            result = adapter.evaluate(
                Dataset.from_list([row()]),
                artifact,
                EvalConfig(classes=["car"], preprocess=PreprocessConfig(resize_mode="letterbox", height=32, width=32)),
            )

        self.assertEqual(result.num_predictions, 1)
        self.assertEqual(result.meta["invalid_prediction_boxes_dropped"], 1)

    def test_predict_converts_canonical_rgb_to_provider_bgr(self):
        captured_sources = []

        class FakeYOLO:
            def __init__(self, checkpoint):
                pass

            def predict(self, **kwargs):
                captured_sources.extend(kwargs["source"])
                boxes = SimpleNamespace(
                    xyxy=torch.empty((0, 4)),
                    conf=torch.empty(0),
                    cls=torch.empty(0),
                )
                return [SimpleNamespace(boxes=boxes)]

        fake_ultralytics = ModuleType("ultralytics")
        fake_ultralytics.YOLO = FakeYOLO
        adapter = UltralyticsDetectionAdapter(
            ModelConfig(key="yolo", backend="ultralytics", detector_pretraining_dataset="coco", model_name_or_path="unused.pt", preprocess=PreprocessConfig(resize_mode="letterbox", height=32, width=32))
        )
        artifact = ModelArtifact(
            model_key=adapter.key,
            backend=adapter.backend,
            run_key="r",
            classes=["car"],
            label_mode="meta",
        )
        dataset_row = row()
        dataset_row["image"]["bytes"] = image_bytes(color=(10, 20, 30))

        with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
            list(
                adapter.predict(
                    Dataset.from_list([dataset_row]),
                    artifact,
                    PredictConfig(classes=["car"], preprocess=PreprocessConfig(resize_mode="letterbox", height=32, width=32)),
                )
            )

        self.assertEqual(captured_sources[0][4, 0].tolist(), [30, 20, 10])

    def test_controlled_protocol_disables_provider_augmentation_overrides(self):
        adapter = UltralyticsDetectionAdapter(
            ModelConfig(key="yolo", backend="ultralytics", detector_pretraining_dataset="coco", model_name_or_path="yolo11n.pt", preprocess=PreprocessConfig(resize_mode="letterbox", height=64, width=64))
        )
        cfg = TrainConfig(
            run_key="r",
            classes=["car"],
            output_dir=Path("runs/test"),
            preprocess=PreprocessConfig(resize_mode="letterbox", height=64, width=64),
            protocol="controlled",
            backend_params={"overrides": {"mosaic": 1.0, "hsv_h": 0.5}},
        )

        overrides = adapter._train_overrides(cfg)

        self.assertEqual(overrides["mosaic"], 0.0)
        self.assertEqual(overrides["hsv_h"], 0.0)
        self.assertEqual(overrides["mixup"], 0.0)

    def test_controlled_protocol_rejects_provider_optimizer_scheduler_overrides(self):
        adapter = UltralyticsDetectionAdapter(
            ModelConfig(key="yolo", backend="ultralytics", detector_pretraining_dataset="coco", model_name_or_path="yolo11n.pt", preprocess=PreprocessConfig(resize_mode="letterbox", height=64, width=64))
        )
        cfg = TrainConfig(
            run_key="r",
            classes=["car"],
            output_dir=Path("runs/test"),
            preprocess=PreprocessConfig(resize_mode="letterbox", height=64, width=64),
            protocol="controlled",
            hparams={"learning_rate": 3.0e-4},
            backend_params={
                "overrides": {
                    "epochs": 7,
                    "optimizer": "SGD",
                    "lr0": 0.5,
                    "lrf": 0.5,
                    "weight_decay": 0.2,
                    "momentum": 0.5,
                    "warmup_epochs": 4.0,
                    "cos_lr": True,
                    "patience": 99,
                    "nbs": 64,
                    "freeze": 10,
                }
            },
        )

        overrides = adapter._train_overrides(cfg)

        self.assertEqual(overrides["epochs"], 50)
        self.assertEqual(overrides["optimizer"], "AdamW")
        self.assertEqual(overrides["lr0"], 3.0e-4)
        self.assertEqual(overrides["lrf"], 0.01)
        self.assertEqual(overrides["weight_decay"], 1.0e-4)
        self.assertEqual(overrides["momentum"], 0.9)
        self.assertEqual(overrides["warmup_epochs"], 0.0)
        self.assertFalse(overrides["cos_lr"])
        self.assertEqual(overrides["patience"], 0)
        self.assertEqual(overrides["nbs"], cfg.batch_size)
        self.assertEqual(overrides["freeze"], 0)

    def test_optimizer_rejects_a_frozen_backbone(self):
        adapter = UltralyticsDetectionAdapter(
            ModelConfig(key="yolo", backend="ultralytics", detector_pretraining_dataset="coco", model_name_or_path="yolo11n.pt", preprocess=PreprocessConfig(resize_mode="letterbox", height=64, width=64))
        )
        trainer = adapter._trainer_class(FakeDetectionTrainer)(
            train_source=None,
            val_source=None,
            train_transform=None,
            eval_transform=None,
            loader_cfg=None,
            classes=["car"],
            max_steps=None,
            logger=None,
            log_prefix="train",
            logging_steps=100,
            optimizer_cfg=TrainConfig(
                run_key="r",
                classes=["car"],
                output_dir=Path("runs/test"),
                preprocess=PreprocessConfig(resize_mode="letterbox", height=64, width=64),
            ).optimizer,
        )
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
        model[0].weight.requires_grad = False

        with self.assertRaisesRegex(ValueError, "fully trainable backbone"):
            trainer.build_optimizer(model)

        model[0].weight.requires_grad = True
        trainer.build_optimizer(model)
        self.assertEqual(trainer._protocol_backbone_meta["backbone_parameters_frozen"], 0)
        self.assertGreater(trainer._protocol_backbone_meta["backbone_parameters_total"], 0)

    def test_configured_pretrained_weights_are_used_as_training_source(self):
        adapter = UltralyticsDetectionAdapter(
            ModelConfig(
                key="yolo",
                backend="ultralytics",
                detector_pretraining_dataset="coco",
                model_name_or_path="architecture.yaml",
                preprocess=PreprocessConfig(resize_mode="letterbox", height=64, width=64),
                weights="pretrained.pt",
            )
        )
        cfg = TrainConfig(
            run_key="r",
            classes=["car"],
            output_dir=Path("runs/test"),
            preprocess=PreprocessConfig(resize_mode="letterbox", height=64, width=64),
            hparams={"learning_rate": 3.0e-4},
        )

        overrides = adapter._train_overrides(cfg)

        self.assertEqual(overrides["model"], "pretrained.pt")

    def test_multi_device_training_fails_explicitly(self):
        adapter = UltralyticsDetectionAdapter(
            ModelConfig(key="yolo", backend="ultralytics", detector_pretraining_dataset="coco", model_name_or_path="yolo11n.pt", preprocess=PreprocessConfig(resize_mode="letterbox", height=64, width=64))
        )
        cfg = TrainConfig(
            run_key="r",
            classes=["car"],
            output_dir=Path("runs/test"),
            preprocess=PreprocessConfig(resize_mode="letterbox", height=64, width=64),
            backend_params={"device": "0,1"},
        )

        with self.assertRaisesRegex(NotImplementedError, "single training process"):
            adapter._validate_single_process_device(cfg)

    def test_shared_scheduler_forces_one_update_per_batch(self):
        adapter = UltralyticsDetectionAdapter(
            ModelConfig(key="yolo", backend="ultralytics", detector_pretraining_dataset="coco", model_name_or_path="yolo11n.pt", preprocess=PreprocessConfig(resize_mode="letterbox", height=64, width=64))
        )
        trainer_cls = adapter._trainer_class(FakeDetectionTrainer)
        trainer = trainer_cls(
            train_source=None,
            val_source=None,
            train_transform=None,
            eval_transform=None,
            loader_cfg=None,
            classes=["car"],
            max_steps=None,
            logger=None,
            log_prefix="train",
            logging_steps=100,
            scheduler_cfg=SimpleNamespace(
                warmup_epochs=1,
                total_epochs=50,
                min_lr_ratio=0.01,
            ),
        )
        trainer.train_loader = [object()] * 10
        scheduler = SimpleNamespace(last_epoch=0)
        with patch(
            "obj_det.models.adapters.ultralytics.build_warmup_cosine_scheduler",
            return_value=scheduler,
        ) as factory:
            trainer._setup_scheduler()

        self.assertEqual(trainer.accumulate, 1)
        factory.assert_called_once_with(
            trainer.optimizer,
            warmup_steps=10,
            total_steps=500,
            min_lr_ratio=0.01,
        )

    def test_amp_overflow_does_not_advance_optimizer_step_scheduler(self):
        class ScaleTracker:
            def __init__(self):
                self.value = 8.0
                self.unscaled = False

            def get_scale(self):
                return self.value

            def unscale_(self, optimizer):
                self.unscaled = True

            def step(self, optimizer):
                return None

            def update(self):
                self.value = 4.0

        adapter = UltralyticsDetectionAdapter(
            ModelConfig(key="yolo", backend="ultralytics", detector_pretraining_dataset="coco", model_name_or_path="yolo11n.pt", preprocess=PreprocessConfig(resize_mode="letterbox", height=64, width=64))
        )
        trainer_cls = adapter._trainer_class(FakeDetectionTrainer)
        trainer = trainer_cls(
            train_source=None,
            val_source=None,
            train_transform=None,
            eval_transform=None,
            loader_cfg=None,
            classes=["car"],
            max_steps=None,
            logger=None,
            log_prefix="train",
            logging_steps=100,
        )
        trainer.scaler = ScaleTracker()
        trainer.model = torch.nn.Linear(2, 1)
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
        trainer._protocol_scheduler = SimpleNamespace(step=lambda: self.fail("scheduler advanced"))

        trainer.optimizer_step()

        self.assertEqual(trainer._protocol_optimizer_steps, 0)
        self.assertTrue(trainer.scaler.unscaled)

    def test_optimizer_step_uses_shared_clipping_without_ema_update(self):
        class ScaleTracker:
            def get_scale(self):
                return 1.0

            def unscale_(self, optimizer):
                pass

            def step(self, optimizer):
                optimizer.step()

            def update(self):
                pass

        adapter = UltralyticsDetectionAdapter(
            ModelConfig(key="yolo", backend="ultralytics", detector_pretraining_dataset="coco", model_name_or_path="yolo11n.pt", preprocess=PreprocessConfig(resize_mode="letterbox", height=64, width=64))
        )
        trainer = adapter._trainer_class(FakeDetectionTrainer)(
            train_source=None,
            val_source=None,
            train_transform=None,
            eval_transform=None,
            loader_cfg=None,
            classes=["car"],
            max_steps=None,
            logger=None,
            log_prefix="train",
            logging_steps=100,
        )
        trainer.model = torch.nn.Linear(2, 1)
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
        trainer.scaler = ScaleTracker()
        trainer.ema = SimpleNamespace(update=lambda model: self.fail("EMA updated"))
        trainer._protocol_scheduler = SimpleNamespace(step=lambda: None)

        with patch("torch.nn.utils.clip_grad_norm_") as clip:
            trainer.optimizer_step()

        clip.assert_called_once()
        clipped_parameters = list(clip.call_args.args[0])
        self.assertEqual(clipped_parameters, list(trainer.model.parameters()))
        self.assertEqual(clip.call_args.kwargs["max_norm"], MAX_GRAD_NORM)
        self.assertEqual(trainer._protocol_optimizer_steps, 1)

    def test_save_model_copies_raw_weights_into_provider_container(self):
        class SavingFakeDetectionTrainer(FakeDetectionTrainer):
            def save_model(self):
                self.saved_weight = self.ema.ema.weight.detach().clone()
                return True

        adapter = UltralyticsDetectionAdapter(
            ModelConfig(key="yolo", backend="ultralytics", detector_pretraining_dataset="coco", model_name_or_path="yolo11n.pt", preprocess=PreprocessConfig(resize_mode="letterbox", height=64, width=64))
        )
        trainer = adapter._trainer_class(SavingFakeDetectionTrainer)(
            train_source=None,
            val_source=None,
            train_transform=None,
            eval_transform=None,
            loader_cfg=None,
            classes=["car"],
            max_steps=None,
            logger=None,
            log_prefix="train",
            logging_steps=100,
        )
        trainer.model = torch.nn.Linear(2, 1, bias=False)
        trainer.ema = SimpleNamespace(ema=torch.nn.Linear(2, 1, bias=False))
        with torch.no_grad():
            trainer.model.weight.fill_(3.0)
            trainer.ema.ema.weight.zero_()

        trainer.save_model()

        self.assertTrue(torch.equal(trainer.saved_weight, trainer.model.weight))

    def test_trial_final_save_redirects_provider_best_to_single_checkpoint(self):
        class SavingFakeDetectionTrainer(FakeDetectionTrainer):
            def save_model(self):
                self.last.write_text("last", encoding="utf-8")
                if self.best_fitness == self.fitness:
                    self.best.write_text("best", encoding="utf-8")
                return True

        adapter = UltralyticsDetectionAdapter(
            ModelConfig(key="yolo", backend="ultralytics", detector_pretraining_dataset="coco", model_name_or_path="yolo11n.pt", preprocess=PreprocessConfig(resize_mode="letterbox", height=64, width=64))
        )
        trainer_cls = adapter._trainer_class(SavingFakeDetectionTrainer)
        trainer = trainer_cls(
            train_source=None,
            val_source=None,
            train_transform=None,
            eval_transform=None,
            loader_cfg=None,
            classes=["car"],
            max_steps=None,
            logger=None,
            log_prefix="train",
            logging_steps=100,
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            trainer.last = root / "last.pt"
            trainer.best = root / "best.pt"
            trainer.best_fitness = None
            trainer.fitness = None

            trainer._save_trial_final_checkpoint()

            self.assertTrue(trainer.last.exists())
            self.assertFalse((root / "best.pt").exists())
            self.assertEqual(trainer.best, root / "best.pt")

    def test_hpo_epoch_end_does_not_record_nonexistent_intermediate_checkpoints(self):
        adapter = UltralyticsDetectionAdapter(
            ModelConfig(key="yolo", backend="ultralytics", detector_pretraining_dataset="coco", model_name_or_path="yolo11n.pt", preprocess=PreprocessConfig(resize_mode="letterbox", height=64, width=64))
        )
        trainer_cls = adapter._trainer_class(FakeDetectionTrainer)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_state = CheckpointState(root)
            trainer = trainer_cls(
                train_source=None,
                val_source=None,
                train_transform=None,
                eval_transform=None,
                loader_cfg=None,
                classes=["car"],
                max_steps=None,
                logger=None,
                log_prefix="train",
                logging_steps=100,
                checkpoint_state=checkpoint_state,
            )
            trainer.epoch = 0
            trainer.wdir = root / "weights"
            trainer.last = trainer.wdir / "last.pt"
            trainer.args = SimpleNamespace(save=False)
            trainer._protocol_stop_at_epoch_end = False

            trainer._protocol_epoch_end()

            self.assertIsNone(checkpoint_state.last_checkpoint)
            self.assertFalse(checkpoint_state.manifest_path.exists())

    def test_controlled_train_overrides_use_fixed_final_lr_ratio(self):
        adapter = UltralyticsDetectionAdapter(
            ModelConfig(key="yolo", backend="ultralytics", detector_pretraining_dataset="coco", model_name_or_path="yolo11n.pt", preprocess=PreprocessConfig(resize_mode="letterbox", height=64, width=64))
        )
        cfg = TrainConfig(
            run_key="r",
            classes=["car"],
            output_dir=Path("runs/test"),
            preprocess=PreprocessConfig(resize_mode="letterbox", height=64, width=64),
            hparams={"lrf": 0.05},
        )

        overrides = adapter._train_overrides(cfg)

        self.assertEqual(overrides["lrf"], 0.01)

    def test_artifact_uses_last_checkpoint_for_external_eval(self):
        adapter = UltralyticsDetectionAdapter(
            ModelConfig(key="yolo", backend="ultralytics", detector_pretraining_dataset="coco", model_name_or_path="yolo11n.pt", preprocess=PreprocessConfig(resize_mode="letterbox", height=64, width=64))
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
                    preprocess=PreprocessConfig(resize_mode="letterbox", height=64, width=64),
                ),
            )

        self.assertEqual(artifact.checkpoint_path, last)
        self.assertEqual(artifact.meta["protocol"], "controlled")
        self.assertEqual(artifact.meta["ultralytics_best"], str(best))
        self.assertEqual(artifact.meta["checkpoint_selection"], "last")
        self.assertEqual(artifact.meta["weight_source"], "raw")
        self.assertFalse(artifact.meta["ema_enabled"])
        self.assertEqual(artifact.meta["max_grad_norm"], MAX_GRAD_NORM)
        self.assertEqual(artifact.meta["detector_pretraining_dataset"], "coco")
        self.assertTrue(artifact.meta["backbone_pretraining_allowed"])
        self.assertTrue(artifact.meta["class_head_reinitialized"])


if __name__ == "__main__":
    unittest.main()
