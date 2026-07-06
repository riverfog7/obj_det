from __future__ import annotations

import unittest

import numpy as np

from obj_det.models.data.row_parser import HFDetectionRowParser
from obj_det.models.data.transforms import DetectionTransform, bbox_to_original, build_detection_transform
from obj_det.datasets.models import BBox
from obj_det.models.schemas import AugmentationConfig, PreprocessConfig

from .helpers import row


class TransformTest(unittest.TestCase):
    def test_basic_transform_uses_albumentations_for_horizontal_flip(self):
        sample = HFDetectionRowParser(["car"], "meta").parse(row(size=(32, 24)))
        transformed = DetectionTransform(
            PreprocessConfig(image_size=32),
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
            PreprocessConfig(image_size=64),
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

    def test_basic_transform_allows_empty_targets(self):
        sample = HFDetectionRowParser(["car"], "meta").parse(row(size=(32, 24), objects=[]))
        transformed = DetectionTransform(
            PreprocessConfig(image_size=32),
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
        transformed = build_detection_transform(PreprocessConfig(image_size=64))(sample)

        self.assertEqual(transformed.image.shape, (64, 64, 3))
        self.assertIn("preprocess", transformed.meta)

    def test_preprocess_only_transform_is_deterministic(self):
        sample = HFDetectionRowParser(["car"], "meta").parse(row(size=(32, 24)))
        transform = build_detection_transform(PreprocessConfig(image_size=64))

        first = transform(sample)
        second = transform(sample)

        self.assertTrue(np.array_equal(first.image, second.image))
        self.assertEqual(first.targets[0].bbox_xywh, second.targets[0].bbox_xywh)

if __name__ == "__main__":
    unittest.main()
