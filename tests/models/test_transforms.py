from __future__ import annotations

import unittest

from obj_det.models.data.row_parser import HFDetectionRowParser
from obj_det.models.data.transforms import bbox_to_original, horizontal_flip_sample, resize_pad_sample

from .helpers import row


class TransformTest(unittest.TestCase):
    def test_horizontal_flip_updates_box(self):
        sample = HFDetectionRowParser(["car"], "meta").parse(row(size=(32, 24)))
        flipped = horizontal_flip_sample(sample)

        self.assertEqual(flipped.targets[0].bbox.xywh(), (20.0, 5.0, 8.0, 10.0))

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


if __name__ == "__main__":
    unittest.main()

class WeatherTransformTest(unittest.TestCase):
    def test_weather_transform_preserves_geometry(self):
        from obj_det.models.data.transforms import WeatherDetectionTransform

        sample = HFDetectionRowParser(["car"], "meta").parse(row(size=(32, 24), condition="haze"))
        transformed = WeatherDetectionTransform(64, {"effect": "low_light", "horizontal_flip_p": 0.0})(sample)

        self.assertEqual(transformed.image.shape, (64, 64, 3))
        self.assertEqual(len(transformed.targets), 1)
        self.assertIn("preprocess", transformed.meta)
