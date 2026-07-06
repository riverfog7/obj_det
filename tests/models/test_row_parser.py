from __future__ import annotations

import tempfile
import unittest

import numpy as np
from PIL import Image

from obj_det.models.data.row_parser import HFDetectionRowParser

from .helpers import image_bytes, row


class RowParserTest(unittest.TestCase):
    def test_meta_mode_decodes_bytes_and_maps_labels(self):
        parser = HFDetectionRowParser(classes=["car", "person"], label_mode="meta")
        sample = parser.parse(row())

        self.assertEqual(sample.image.shape, (24, 32, 3))
        self.assertEqual(sample.image.dtype, np.uint8)
        self.assertEqual(sample.targets[0].label, "car")
        self.assertEqual(sample.targets[0].label_id, 0)
        self.assertEqual(sample.targets[0].bbox_xywh, (4.0, 5.0, 8.0, 10.0))
        self.assertEqual(sample.meta["row"], "img1")

    def test_native_mode_keeps_native_label(self):
        parser = HFDetectionRowParser(classes=["automobile"], label_mode="native")
        sample = parser.parse(row(objects=[{
            "bbox": [1, 2, 3, 4],
            "native_label": "automobile",
            "native_label_id": "7",
            "meta_label": "car",
            "ignore": False,
            "iscrowd": False,
            "meta_json": "{}",
        }]))

        self.assertEqual(sample.targets[0].label, "automobile")
        self.assertEqual(sample.targets[0].label_id, 0)

    def test_parse_targets_only_does_not_decode_image(self):
        class NoDecodeParser(HFDetectionRowParser):
            def decode_image(self, image_field):
                raise AssertionError("decode_image should not be called")

        sample = NoDecodeParser(classes=["car"], label_mode="meta").parse_targets_only(row())

        self.assertIsNone(sample.image)
        self.assertEqual(sample.targets[0].bbox_xywh, (4.0, 5.0, 8.0, 10.0))

    def test_drops_ignore_missing_meta_and_unknown_label(self):
        parser = HFDetectionRowParser(classes=["car"], label_mode="meta")
        sample = parser.parse(row(objects=[
            {"bbox": [1, 1, 2, 2], "native_label": "car", "meta_label": "car", "ignore": True, "iscrowd": False, "meta_json": "{}"},
            {"bbox": [1, 1, 2, 2], "native_label": "dog", "meta_label": None, "ignore": False, "iscrowd": False, "meta_json": "{}"},
            {"bbox": [1, 1, 2, 2], "native_label": "dog", "meta_label": "dog", "ignore": False, "iscrowd": False, "meta_json": "{}"},
        ]))

        self.assertEqual(sample.targets, [])

    def test_decode_path_pil_and_numpy(self):
        parser = HFDetectionRowParser(classes=["car"], label_mode="meta")
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            f.write(image_bytes())
            f.flush()
            path_image = parser.decode_image({"path": f.name})
        pil_image = parser.decode_image(Image.new("L", (3, 2), color=1))
        np_image = parser.decode_image(np.zeros((2, 3), dtype=np.uint8))

        self.assertEqual(path_image.shape, (24, 32, 3))
        self.assertEqual(pil_image.shape, (2, 3, 3))
        self.assertEqual(np_image.shape, (2, 3, 3))

    def test_decode_backend_opencv_decodes_bytes(self):
        parser = HFDetectionRowParser(classes=["car"], label_mode="meta", decode_backend="opencv")
        image = parser.decode_image({"bytes": image_bytes()})

        self.assertEqual(image.shape, (24, 32, 3))
        self.assertEqual(image.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
