from __future__ import annotations

import unittest

from datasets import Dataset

from obj_det.datasets.models import BBox
from obj_det.models.evaluation import DetectionEvaluator
from obj_det.models.data.row_parser import HFDetectionRowParser
from obj_det.models.schemas.config import EvalConfig, PreprocessConfig
from obj_det.models.schemas.prediction import PredictionObject, PredictionRecord

from .helpers import row


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
                preprocess=PreprocessConfig(image_size=32),
                compute_per_class=True,
                compute_per_condition=True,
                compute_per_domain=True,
            ),
            model_key="dummy",
        )

        self.assertGreater(result.metrics["map_50_95"], 0.9)
        self.assertIn("car", result.per_class)
        self.assertIn("clear", result.per_condition)
        self.assertIn("general", result.per_domain)

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
            EvalConfig(classes=["car"], label_mode="meta", preprocess=PreprocessConfig(image_size=32)),
            model_key="dummy",
        )

        self.assertEqual(result.primary_metric_value, 0.0)

    def test_evaluation_keeps_empty_images(self):
        ds = Dataset.from_list([row(objects=[])])
        result = DetectionEvaluator().evaluate(
            ds,
            [],
            EvalConfig(classes=["car"], label_mode="meta", preprocess=PreprocessConfig(image_size=32)),
            model_key="dummy",
        )

        self.assertEqual(result.num_images, 1)
        self.assertEqual(result.num_ground_truth_objects, 0)

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
                EvalConfig(classes=["car"], label_mode="meta", preprocess=PreprocessConfig(image_size=32)),
                model_key="dummy",
            )
        finally:
            HFDetectionRowParser.decode_image = original_decode

        self.assertEqual(result.num_images, 1)


if __name__ == "__main__":
    unittest.main()
