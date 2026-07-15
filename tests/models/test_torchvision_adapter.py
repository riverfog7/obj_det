from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from obj_det.models.adapters.torchvision import (
    _TORCHVISION_MODEL_SPECS,
    _TorchvisionTrainer,
    TorchvisionDetectionAdapter,
    _target_dict,
)
from obj_det.models.data.sample import DetectionSample, DetectionTarget
from obj_det.models.schemas import ModelConfig, PredictConfig, PreprocessConfig, TrainConfig
from obj_det.models.training import MAX_GRAD_NORM


class TorchvisionAdapterTest(unittest.TestCase):
    def _adapter(self, model_name: str, *, weights=None, params=None) -> TorchvisionDetectionAdapter:
        return TorchvisionDetectionAdapter(
            ModelConfig(
                key="tv",
                backend="torchvision",
                detector_pretraining_dataset="coco",
                model_name_or_path=model_name,
                preprocess=PreprocessConfig(
                    resize_mode="shortest_edge", shortest_edge=800, longest_edge=1333
                ),
                weights=weights,
                params=params or {},
            )
        )

    def _sample(self) -> DetectionSample:
        return DetectionSample(
            image=np.zeros((16, 16, 3), dtype=np.uint8),
            image_id="image-1",
            dataset="tiny",
            split="train",
            width=16,
            height=16,
            targets=[
                DetectionTarget(
                    bbox_xywh=(1.0, 2.0, 3.0, 4.0),
                    label="car",
                    label_id=0,
                )
            ],
        )

    def test_model_specs_centralize_background_label_semantics(self):
        expected = {
            "fasterrcnn_resnet50_fpn": (3, 1, 0),
            "maskrcnn_resnet50_fpn": (3, 1, 0),
            "retinanet_resnet50_fpn": (2, 0, 0),
            "fcos_resnet50_fpn": (2, 0, 0),
        }

        self.assertEqual(set(_TORCHVISION_MODEL_SPECS), set(expected))
        for model_name, (num_classes, training_label, canonical_label) in expected.items():
            with self.subTest(model=model_name):
                spec = _TORCHVISION_MODEL_SPECS[model_name]
                self.assertEqual(spec.model_num_classes(2), num_classes)
                self.assertEqual(spec.training_label(0), training_label)
                self.assertEqual(spec.canonical_label(training_label), canonical_label)

    def test_target_labels_offset_only_two_stage_models(self):
        sample = self._sample()

        faster_labels = _target_dict(sample, _TORCHVISION_MODEL_SPECS["fasterrcnn_resnet50_fpn"])["labels"]
        retina_labels = _target_dict(sample, _TORCHVISION_MODEL_SPECS["retinanet_resnet50_fpn"])["labels"]

        self.assertEqual(faster_labels.tolist(), [1])
        self.assertEqual(retina_labels.tolist(), [0])

    def test_predict_config_updates_internal_thresholds_for_all_models(self):
        cfg = PredictConfig(
            classes=["car"],
            preprocess=PreprocessConfig(resize_mode="letterbox", height=32, width=32),
            conf_threshold=0.25,
            iou_threshold=0.45,
            backend_params={"max_detections_per_image": 300},
        )

        for model_name, spec in _TORCHVISION_MODEL_SPECS.items():
            with self.subTest(model=model_name):
                threshold_owner = SimpleNamespace(
                    score_thresh=0.0,
                    nms_thresh=0.0,
                    detections_per_img=100,
                )
                model = (
                    SimpleNamespace(roi_heads=threshold_owner)
                    if spec.threshold_parent
                    else threshold_owner
                )
                adapter = self._adapter(model_name)

                adapter._apply_predict_config(model, spec, cfg)

                self.assertEqual(threshold_owner.score_thresh, 0.25)
                self.assertEqual(threshold_owner.nms_thresh, 0.45)
                self.assertEqual(threshold_owner.detections_per_img, 300)

    def test_max_detections_must_be_positive(self):
        model_name = "fasterrcnn_resnet50_fpn"
        spec = _TORCHVISION_MODEL_SPECS[model_name]
        model = SimpleNamespace(roi_heads=SimpleNamespace())
        cfg = SimpleNamespace(
            conf_threshold=0.001,
            iou_threshold=0.7,
            max_detections_per_image=0,
            backend_params={},
        )

        with self.assertRaisesRegex(ValueError, "max_detections_per_image"):
            self._adapter(model_name)._apply_predict_config(model, spec, cfg)

    def test_explicit_default_weights_resolve_and_are_recorded(self):
        model_name = "fasterrcnn_resnet50_fpn"
        adapter = self._adapter(model_name, weights="default")
        spec = _TORCHVISION_MODEL_SPECS[model_name]

        weights = adapter._resolve_weights(spec)
        metadata = adapter._pretraining_metadata(spec, weights)

        self.assertIs(weights, spec.weights_cls.DEFAULT)
        self.assertTrue(metadata["pretrained"])
        self.assertEqual(metadata["detector_pretraining_dataset"], "coco")
        self.assertTrue(metadata["backbone_pretraining_allowed"])
        self.assertTrue(metadata["class_head_reinitialized"])
        self.assertEqual(metadata["pretrained_config"], "default")
        self.assertEqual(metadata["pretrained_weights"], weights.name)
        self.assertTrue(metadata["pretrained_source"])

    def test_none_weights_are_explicitly_recorded_as_untrained(self):
        model_name = "retinanet_resnet50_fpn"
        adapter = self._adapter(model_name, params={"weights": "none"})
        spec = _TORCHVISION_MODEL_SPECS[model_name]

        weights = adapter._resolve_weights(spec)
        metadata = adapter._pretraining_metadata(spec, weights)

        self.assertIsNone(weights)
        self.assertFalse(metadata["pretrained"])
        self.assertEqual(metadata["pretrained_config"], "none")
        self.assertIsNone(metadata["pretrained_source"])

    def test_trainer_uses_canonical_adamw_settings(self):
        trainer = object.__new__(_TorchvisionTrainer)
        trainer.model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.LayerNorm(2))
        trainer.train_cfg = TrainConfig(
            run_key="r",
            classes=["car"],
            output_dir=Path("runs/test"),
            preprocess=PreprocessConfig(resize_mode="letterbox", height=32, width=32),
            hparams={"learning_rate": 3.0e-4},
            optimizer={
                "weight_decay": 2.0e-4,
                "beta1": 0.8,
                "beta2": 0.98,
                "epsilon": 1.0e-7,
            },
        )
        trainer.optimizer = None

        optimizer = trainer.create_optimizer()

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.defaults["lr"], 3.0e-4)
        self.assertEqual(optimizer.defaults["betas"], (0.8, 0.98))
        self.assertEqual(optimizer.defaults["eps"], 1.0e-7)
        self.assertEqual({group["weight_decay"] for group in optimizer.param_groups}, {0.0, 2.0e-4})

    def test_trainer_scheduler_uses_configured_fifty_epoch_horizon(self):
        trainer = object.__new__(_TorchvisionTrainer)
        trainer.model = torch.nn.Linear(2, 2)
        trainer.train_cfg = TrainConfig(
            run_key="r",
            classes=["car"],
            output_dir=Path("runs/test"),
            preprocess=PreprocessConfig(resize_mode="letterbox", height=32, width=32),
            hparams={"learning_rate": 1.0e-3},
            scheduler={
                "warmup_epochs": 1,
                "total_epochs": 50,
                "min_lr_ratio": 0.01,
            },
        )
        trainer.optimizer = None
        trainer.lr_scheduler = None
        trainer.get_train_dataloader = lambda: [None] * 4
        optimizer = trainer.create_optimizer()

        scheduler = trainer.create_scheduler(num_training_steps=40, optimizer=optimizer)
        factor = scheduler.lr_lambdas[0]

        self.assertAlmostEqual(factor(4), 1.0)
        self.assertAlmostEqual(factor(200), 0.01)

    def test_training_args_use_shared_gradient_clipping(self):
        cfg = TrainConfig(
            run_key="r",
            classes=["car"],
            output_dir=Path("runs/test"),
            preprocess=PreprocessConfig(resize_mode="letterbox", height=32, width=32),
            hparams={"max_grad_norm": 99.0},
        )

        args = self._adapter("fasterrcnn_resnet50_fpn")._training_args(cfg)

        self.assertEqual(args.max_grad_norm, MAX_GRAD_NORM)
        self.assertEqual(args.data_seed, cfg.seed)


if __name__ == "__main__":
    unittest.main()
