from __future__ import annotations

import unittest
from unittest.mock import Mock

from datasets import Dataset

from obj_det.datasets.models import BBox
from obj_det.models.adapters.base import BaseModelAdapter
from obj_det.models.evaluation import DetectionEvaluator
from obj_det.models.data.row_parser import HFDetectionRowParser
from obj_det.models.schemas.artifact import ModelArtifact
from obj_det.models.schemas.config import EvalConfig, ModelConfig, PreprocessConfig
from obj_det.models.schemas.prediction import PredictionObject, PredictionRecord

from .helpers import row


class _PredictionConfigRecordingAdapter(BaseModelAdapter):
    def __init__(self):
        super().__init__(
            ModelConfig(
                key="recording",
                backend="torchvision",
                model_name_or_path="unused",
                preprocess=PreprocessConfig(resize_mode="letterbox", height=32, width=32),
            )
        )
        self.predict_cfg = None

    def train(self, train_ds, val_ds, train_cfg, *, logger=None, log_prefix="train"):
        raise NotImplementedError

    def predict(self, ds, artifact, predict_cfg):
        self.predict_cfg = predict_cfg
        for item in ds:
            yield PredictionRecord(
                image_id=str(item["image_id"]),
                dataset=str(item["dataset"]),
                split=str(item["split"]),
                model_key=self.key,
                width=int(item["width"]),
                height=int(item["height"]),
                predictions=[],
            )


class EvaluatorTest(unittest.TestCase):
    def test_perfect_prediction_scores_high(self):
        ds = Dataset.from_list([row()])
        predictions = [
            PredictionRecord(
                image_id="img1",
                dataset="tiny",
                split="val",
                model_key="dummy",
                width=32,
                height=24,
                predictions=[PredictionObject(bbox=BBox.from_xywh([4, 5, 8, 10]), label="car", score=0.9)],
            )
        ]
        result = DetectionEvaluator().evaluate(
            ds,
            predictions,
            EvalConfig(
                classes=["car"],
                label_mode="meta",
                preprocess=PreprocessConfig(resize_mode="letterbox", height=32, width=32),
                compute_per_class=True,
                compute_per_condition=True,
                compute_per_domain=True,
                compute_per_size=True,
            ),
            model_key="dummy",
        )

        self.assertGreater(result.metrics["map_50_95"], 0.9)
        self.assertIn("car", result.per_class)
        self.assertIn("clear", result.per_condition)
        self.assertIn("general", result.per_domain)
        self.assertIn("ar_300", result.metrics)
        self.assertIn("ar_300", result.per_size["small"])
        self.assertEqual(result.meta["evaluation_protocol"], "unified_harmonized")
        self.assertEqual(result.meta["max_detections_per_image"], 300)

    def test_no_predictions_scores_zero(self):
        ds = Dataset.from_list([row()])
        predictions = [
            PredictionRecord(
                image_id="img1",
                dataset="tiny",
                split="val",
                model_key="dummy",
                width=32,
                height=24,
                predictions=[],
            )
        ]
        result = DetectionEvaluator().evaluate(
            ds,
            predictions,
            EvalConfig(classes=["car"], label_mode="meta", preprocess=PreprocessConfig(resize_mode="letterbox", height=32, width=32)),
            model_key="dummy",
        )

        self.assertEqual(result.primary_metric_value, 0.0)

    def test_evaluation_aggregates_dropped_prediction_boxes(self):
        ds = Dataset.from_list([row(image_id="img1"), row(image_id="img2")])
        predictions = [
            PredictionRecord(
                image_id="img1",
                dataset="tiny",
                split="val",
                model_key="dummy",
                width=32,
                height=24,
                meta={"invalid_prediction_boxes_dropped": 1},
            ),
            PredictionRecord(
                image_id="img2",
                dataset="tiny",
                split="val",
                model_key="dummy",
                width=32,
                height=24,
                meta={"invalid_prediction_boxes_dropped": 2},
            ),
        ]

        result = DetectionEvaluator().evaluate(
            ds,
            predictions,
            EvalConfig(classes=["car"], preprocess=PreprocessConfig(resize_mode="letterbox", height=32, width=32)),
            model_key="dummy",
        )

        self.assertEqual(result.meta["invalid_prediction_boxes_dropped"], 3)

    def test_evaluation_keeps_empty_images(self):
        ds = Dataset.from_list([row(objects=[])])
        result = DetectionEvaluator().evaluate(
            ds,
            [],
            EvalConfig(classes=["car"], label_mode="meta", preprocess=PreprocessConfig(resize_mode="letterbox", height=32, width=32)),
            model_key="dummy",
        )

        self.assertEqual(result.num_images, 1)
        self.assertEqual(result.num_ground_truth_objects, 0)

    def test_dense_scene_metrics_use_configured_max_detections(self):
        objects = []
        predicted_objects = []
        for index in range(120):
            x = float((index % 12) * 10)
            y = float((index // 12) * 10)
            bbox = [x, y, 5.0, 5.0]
            objects.append(
                {
                    "bbox": bbox,
                    "native_label": "car",
                    "native_label_id": "1",
                    "meta_label": "car",
                    "ignore": False,
                    "iscrowd": False,
                    "meta_json": "{}",
                }
            )
            predicted_objects.append(
                PredictionObject(
                    bbox=BBox.from_xywh(bbox),
                    label="car",
                    score=1.0 - index * 0.001,
                )
            )

        ds = Dataset.from_list([row(size=(128, 108), objects=objects)])
        predictions = [
            PredictionRecord(
                image_id="img1",
                dataset="tiny",
                split="val",
                model_key="dummy",
                width=128,
                height=108,
                predictions=predicted_objects,
            )
        ]
        evaluator = DetectionEvaluator()
        preprocess = PreprocessConfig(resize_mode="letterbox", height=128, width=128)

        capped_100 = evaluator.evaluate(
            ds,
            predictions,
            EvalConfig(
                classes=["car"],
                preprocess=preprocess,
                max_detections_per_image=100,
            ),
            model_key="dummy",
        )
        capped_300 = evaluator.evaluate(
            ds,
            predictions,
            EvalConfig(
                classes=["car"],
                preprocess=preprocess,
                max_detections_per_image=300,
            ),
            model_key="dummy",
        )

        self.assertIn("ar_100", capped_100.metrics)
        self.assertIn("ar_300", capped_300.metrics)
        self.assertGreater(capped_300.metrics["map_50_95"], capped_100.metrics["map_50_95"])
        self.assertAlmostEqual(capped_300.metrics["map_50_95"], 1.0)

    def test_missing_primary_metric_raises(self):
        ds = Dataset.from_list([row(objects=[])])

        with self.assertRaisesRegex(ValueError, "Primary evaluation metric 'missing_metric' is missing"):
            DetectionEvaluator().evaluate(
                ds,
                [],
                EvalConfig(
                    classes=["car"],
                    preprocess=PreprocessConfig(resize_mode="letterbox", height=32, width=32),
                    primary_metric="missing_metric",
                ),
                model_key="dummy",
            )

    def test_base_adapter_propagates_max_detections_to_prediction(self):
        ds = Dataset.from_list([row(objects=[])])
        adapter = _PredictionConfigRecordingAdapter()
        artifact = ModelArtifact(
            model_key=adapter.key,
            backend=adapter.backend,
            run_key="recording",
            classes=["car"],
            label_mode="meta",
        )

        adapter.evaluate(
            ds,
            artifact,
            EvalConfig(
                classes=["car"],
                preprocess=PreprocessConfig(resize_mode="letterbox", height=32, width=32),
                max_detections_per_image=217,
            ),
        )

        self.assertIsNotNone(adapter.predict_cfg)
        self.assertEqual(adapter.predict_cfg.max_detections_per_image, 217)

    def test_base_adapter_logs_evaluation_at_explicit_step(self):
        ds = Dataset.from_list([row(objects=[])])
        adapter = _PredictionConfigRecordingAdapter()
        artifact = ModelArtifact(
            model_key=adapter.key,
            backend=adapter.backend,
            run_key="recording",
            classes=["car"],
            label_mode="meta",
        )
        logger = Mock()

        result = adapter.evaluate(
            ds,
            artifact,
            EvalConfig(classes=["car"], preprocess=PreprocessConfig(resize_mode="letterbox", height=32, width=32)),
            logger=logger,
            log_prefix="val/epoch",
            log_step=17,
        )

        logger.log_eval_result.assert_called_once_with(
            result,
            step=17,
            prefix="val/epoch",
        )

    def test_evaluator_does_not_decode_images(self):
        ds = Dataset.from_list([row()])
        predictions = [
            PredictionRecord(
                image_id="img1",
                dataset="tiny",
                split="val",
                model_key="dummy",
                width=32,
                height=24,
                predictions=[PredictionObject(bbox=BBox.from_xywh([4, 5, 8, 10]), label="car", score=0.9)],
            )
        ]
        original_decode = HFDetectionRowParser.decode_image

        def fail_decode(self, image_field):
            raise AssertionError("decode_image should not be called")

        try:
            HFDetectionRowParser.decode_image = fail_decode
            result = DetectionEvaluator().evaluate(
                ds,
                predictions,
                EvalConfig(classes=["car"], label_mode="meta", preprocess=PreprocessConfig(resize_mode="letterbox", height=32, width=32)),
                model_key="dummy",
            )
        finally:
            HFDetectionRowParser.decode_image = original_decode

        self.assertEqual(result.num_images, 1)


if __name__ == "__main__":
    unittest.main()
