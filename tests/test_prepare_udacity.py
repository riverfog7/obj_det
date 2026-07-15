from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.prepare_udacity import CATEGORIES, prepare_udacity


class PrepareUdacityTest(unittest.TestCase):
    def test_selects_v3_export_and_creates_temporal_coco_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export_dir = root / "data" / "export"
            export_dir.mkdir(parents=True)
            names = [
                "6000000000000_jpg.rf.ValidA.jpg",
                "6060000000000_jpg.rf.ValidB.jpg",
                "6120000000000_jpg.rf.ValidC.jpg",
                "6180000000000_jpg.rf.ValidD.jpg",
            ]
            for name in names:
                Image.new("RGB", (4, 4)).save(export_dir / name)

            old_name = "6000000000000_jpg.rf.0123456789abcdef0123456789abcdef.jpg"
            Image.new("RGB", (4, 4)).save(export_dir / old_name)
            with (export_dir / "_annotations.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "filename",
                        "width",
                        "height",
                        "class",
                        "xmin",
                        "ymin",
                        "xmax",
                        "ymax",
                    ]
                )
                writer.writerow([names[0], 4, 4, "car", 0, 0, 3, 3])
                writer.writerow([names[1], 4, 4, "pedestrian", 1, 1, 4, 4])
                writer.writerow([names[2], 4, 4, "truck", 0, 1, 2, 4])
                writer.writerow([old_name, 4, 4, "car", 0, 0, 3, 3])

            counts = prepare_udacity(root, expected_images=4, expected_annotations=3)

            self.assertEqual(
                counts, {"train": 2, "val": 1, "test": 1, "annotations": 3}
            )
            for directory in ("train", "valid", "test"):
                self.assertFalse((root / directory / old_name).exists())
            self.assertEqual(len(list((root / "train").glob("*.jpg"))), 2)
            self.assertEqual(len(list((root / "valid").glob("*.jpg"))), 1)
            self.assertEqual(len(list((root / "test").glob("*.jpg"))), 1)

            payloads = {
                split: json.loads(
                    (root / directory / "_annotations.coco.json").read_text(
                        encoding="utf-8"
                    )
                )
                for split, directory in {
                    "train": "train",
                    "val": "valid",
                    "test": "test",
                }.items()
            }

        self.assertEqual(
            {category["name"] for category in payloads["train"]["categories"]},
            set(CATEGORIES),
        )
        self.assertEqual(sum(len(data["images"]) for data in payloads.values()), 4)
        self.assertEqual(sum(len(data["annotations"]) for data in payloads.values()), 3)


if __name__ == "__main__":
    unittest.main()
