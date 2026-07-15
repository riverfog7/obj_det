from __future__ import annotations

import unittest

import numpy as np

from obj_det.models.data.row_parser import HFDetectionRowParser
from obj_det.models.data.transforms import (
    DetectionTransform,
    bbox_to_original,
    build_detection_transform,
    canonicalize_prediction_bbox,
)
from obj_det.datasets.models import BBox
from obj_det.models.schemas import AugmentationConfig, PreprocessConfig

from .helpers import row


class TransformTest(unittest.TestCase):
    def test_basic_transform_uses_albumentations_for_horizontal_flip(self):
        sample = HFDetectionRowParser(["car"], "meta").parse(row(size=(32, 24)))
        transformed = DetectionTransform(
            PreprocessConfig(resize_mode="letterbox", height=32, width=32),
            AugmentationConfig(
                policy="basic",
                horizontal_flip_p=1.0,
                color_jitter_strength=0.0,
            ),
            seed=0,
        )(sample)

        self.assertEqual(transformed.image.shape, (32, 32, 3))
        self.assertEqual(transformed.targets[0].bbox_xywh[0], 20.0)
        self.assertAlmostEqual(transformed.targets[0].bbox_xywh[1], 9.0, places=6)
        self.assertEqual(transformed.targets[0].bbox_xywh[2], 8.0)
        self.assertAlmostEqual(transformed.targets[0].bbox_xywh[3], 10.0, places=6)

    def test_basic_transform_resize_pad_and_inverse(self):
        sample = HFDetectionRowParser(["car"], "meta").parse(row(size=(32, 24)))
        transformed = DetectionTransform(
            PreprocessConfig(resize_mode="letterbox", height=64, width=64),
            AugmentationConfig(
                policy="none",
                horizontal_flip_p=0.0,
                color_jitter_strength=0.0,
            ),
        )(sample)
        bbox = transformed.targets[0].bbox_xywh
        restored = bbox_to_original(BBox.from_xywh(bbox), transformed.meta["preprocess"])

        self.assertEqual(transformed.image.shape, (64, 64, 3))
        self.assertEqual(bbox[0], 8.0)
        self.assertAlmostEqual(bbox[1], 18.0, places=6)
        self.assertEqual(bbox[2], 16.0)
        self.assertAlmostEqual(bbox[3], 20.0, places=6)
        self.assertIsNotNone(restored)
        self.assertAlmostEqual(restored.x, 4.0, places=6)
        self.assertAlmostEqual(restored.y, 5.0, places=6)
        self.assertAlmostEqual(restored.w, 8.0, places=6)
        self.assertAlmostEqual(restored.h, 10.0, places=6)

    def test_exact_resize_and_inverse_mapping(self):
        sample = HFDetectionRowParser(["car"], "meta").parse(row(size=(32, 24)))
        transformed = build_detection_transform(
            PreprocessConfig(resize_mode="exact", height=60, width=80)
        )(sample)
        restored = bbox_to_original(
            BBox.from_xywh(transformed.targets[0].bbox_xywh),
            transformed.meta["preprocess"],
        )

        self.assertEqual(transformed.image.shape, (60, 80, 3))
        for expected, actual in zip(
            (10.0, 12.5, 20.0, 25.0), transformed.targets[0].bbox_xywh
        ):
            self.assertAlmostEqual(actual, expected, places=5)
        self.assertIsNotNone(restored)
        for expected, actual in zip((4.0, 5.0, 8.0, 10.0), restored.xywh()):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_shortest_edge_resize_observes_longest_edge_cap(self):
        preprocess = PreprocessConfig(
            resize_mode="shortest_edge",
            shortest_edge=800,
            longest_edge=1333,
        )
        landscape = build_detection_transform(preprocess)(
            HFDetectionRowParser(["car"], "meta").parse(row(size=(2000, 1000)))
        )
        portrait = build_detection_transform(preprocess)(
            HFDetectionRowParser(["car"], "meta").parse(row(size=(1000, 2000)))
        )

        self.assertEqual(landscape.image.shape[:2], (666, 1333))
        self.assertEqual(portrait.image.shape[:2], (1333, 666))
        self.assertEqual(landscape.meta["preprocess"]["pad_left"], 0)
        self.assertEqual(landscape.meta["preprocess"]["pad_top"], 0)

    def test_resize_pad_inverse_mapping_across_aspect_ratios(self):
        cases = [
            (480, 960),
            (1920, 1080),
            (640, 640),
            (17, 31),
            (31, 17),
            (853, 481),
        ]

        for width, height in cases:
            with self.subTest(size=(width, height)):
                bbox = [
                    width * 0.2,
                    height * 0.25,
                    width * 0.3,
                    height * 0.2,
                ]
                sample = HFDetectionRowParser(["car"], "meta").parse(
                    row(
                        size=(width, height),
                        objects=[
                            {
                                "bbox": bbox,
                                "native_label": "car",
                                "native_label_id": "1",
                                "meta_label": "car",
                                "ignore": False,
                                "iscrowd": False,
                                "meta_json": "{}",
                            }
                        ],
                    )
                )
                transformed = build_detection_transform(PreprocessConfig(resize_mode="letterbox", height=640, width=640))(sample)
                restored = bbox_to_original(
                    BBox.from_xywh(transformed.targets[0].bbox_xywh),
                    transformed.meta["preprocess"],
                )

                self.assertIsNotNone(restored)
                restored_values = restored.xywh()
                for expected, actual in zip(bbox, restored_values):
                    self.assertAlmostEqual(actual, expected, delta=1e-2)

    def test_prediction_bbox_is_clipped_and_mapped_to_original_image(self):
        sample = HFDetectionRowParser(["car"], "meta").parse(row(size=(32, 24)))
        transformed = build_detection_transform(PreprocessConfig(resize_mode="letterbox", height=64, width=64))(sample)

        bbox = canonicalize_prediction_bbox(
            [-1.0, -1.0, 65.0, 65.0],
            image_width=64,
            image_height=64,
            preprocess=transformed.meta["preprocess"],
        )

        self.assertIsNotNone(bbox)
        self.assertEqual(bbox.xywh(), (0.0, 0.0, 32.0, 24.0))

    def test_prediction_bbox_drops_invalid_coordinates(self):
        invalid_boxes = [
            [1.0, 2.0, 3.0],
            [10.0, 2.0, 9.9281005859375, 10.0],
            [2.0, 10.0, 10.0, 9.0],
            [2.0, 2.0, 2.0, 10.0],
            [40.0, 2.0, 50.0, 10.0],
            [float("nan"), 2.0, 10.0, 10.0],
            [2.0, 2.0, float("inf"), 10.0],
        ]

        for box in invalid_boxes:
            with self.subTest(box=box):
                self.assertIsNone(
                    canonicalize_prediction_bbox(
                        box,
                        image_width=32,
                        image_height=24,
                    )
                )

    def test_basic_transform_allows_empty_targets(self):
        sample = HFDetectionRowParser(["car"], "meta").parse(row(size=(32, 24), objects=[]))
        transformed = DetectionTransform(
            PreprocessConfig(resize_mode="letterbox", height=32, width=32),
            AugmentationConfig(
                policy="basic",
                horizontal_flip_p=1.0,
                color_jitter_strength=0.0,
            ),
            seed=0,
        )(sample)

        self.assertEqual(transformed.image.shape, (32, 32, 3))
        self.assertEqual(transformed.targets, [])

    def test_none_policy_resizes_without_random_augmentation(self):
        sample = HFDetectionRowParser(["car"], "meta").parse(row(size=(32, 24)))
        transformed = build_detection_transform(PreprocessConfig(resize_mode="letterbox", height=64, width=64))(sample)

        self.assertEqual(transformed.image.shape, (64, 64, 3))
        self.assertIn("preprocess", transformed.meta)

    def test_preprocess_only_transform_is_deterministic(self):
        sample = HFDetectionRowParser(["car"], "meta").parse(row(size=(32, 24)))
        transform = build_detection_transform(PreprocessConfig(resize_mode="letterbox", height=64, width=64))

        first = transform(sample)
        second = transform(sample)

        self.assertTrue(np.array_equal(first.image, second.image))
        self.assertEqual(first.targets[0].bbox_xywh, second.targets[0].bbox_xywh)

if __name__ == "__main__":
    unittest.main()
