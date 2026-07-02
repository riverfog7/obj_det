from __future__ import annotations

import unittest

from obj_det.models.data.row_parser import HFDetectionRowParser
from obj_det.models.data.transforms import BasicDetectionTransform, bbox_to_original, resize_pad_sample

from .helpers import row


class TransformTest(unittest.TestCase):
    def test_basic_transform_uses_albumentations_for_horizontal_flip(self):
        sample = HFDetectionRowParser(["car"], "meta").parse(row(size=(32, 24)))
        transformed = BasicDetectionTransform(
            32,
            horizontal_flip_p=1.0,
            color_jitter_strength=0.0,
            seed=0,
        )(sample)

        self.assertEqual(transformed.image.shape, (32, 32, 3))
        self.assertAlmostEqual(transformed.targets[0].bbox.x, 20.0, places=6)
        self.assertAlmostEqual(transformed.targets[0].bbox.y, 9.0, places=6)
        self.assertAlmostEqual(transformed.targets[0].bbox.w, 8.0, places=6)
        self.assertAlmostEqual(transformed.targets[0].bbox.h, 10.0, places=6)

    def test_resize_pad_and_inverse(self):
        sample = HFDetectionRowParser(["car"], "meta").parse(row(size=(32, 24)))
        transformed = resize_pad_sample(sample, 64)
        bbox = transformed.targets[0].bbox
        restored = bbox_to_original(bbox, transformed.meta["preprocess"])

        self.assertEqual(transformed.image.shape, (64, 64, 3))
        self.assertAlmostEqual(bbox.x, 8.0)
        self.assertAlmostEqual(bbox.y, 18.0)
        self.assertAlmostEqual(bbox.w, 16.0)
        self.assertAlmostEqual(bbox.h, 20.0)
        self.assertIsNotNone(restored)
        self.assertAlmostEqual(restored.x, 4.0)
        self.assertAlmostEqual(restored.y, 5.0)
        self.assertAlmostEqual(restored.w, 8.0)
        self.assertAlmostEqual(restored.h, 10.0)

    def test_basic_transform_allows_empty_targets(self):
        sample = HFDetectionRowParser(["car"], "meta").parse(row(size=(32, 24), objects=[]))
        transformed = BasicDetectionTransform(
            32,
            horizontal_flip_p=1.0,
            color_jitter_strength=0.0,
            seed=0,
        )(sample)

        self.assertEqual(transformed.image.shape, (32, 32, 3))
        self.assertEqual(transformed.targets, [])


if __name__ == "__main__":
    unittest.main()

class WeatherTransformTest(unittest.TestCase):
    def test_weather_transform_preserves_geometry(self):
        from obj_det.models.data.transforms import WeatherDetectionTransform

        sample = HFDetectionRowParser(["car"], "meta").parse(row(size=(32, 24), condition="haze"))
        transformed = WeatherDetectionTransform(
            64,
            {
                "effect": "low_light",
                "horizontal_flip_p": 0.0,
                "color_jitter_strength": 0.0,
                "seed": 0,
            },
        )(sample)

        self.assertEqual(transformed.image.shape, (64, 64, 3))
        self.assertEqual(len(transformed.targets), 1)
        self.assertIn("preprocess", transformed.meta)
