from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

from obj_det.datasets.models.source_config import SourceDatasetConfig
from obj_det.datasets.sources import Bdd100kSourceDataset, source_from_config
from obj_det.models.schemas import DatasetRefConfig


class Bdd100kSourceTest(unittest.TestCase):
    def test_reads_scalabel_boxes_and_weather_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            Image.new("RGB", (20, 10)).save(images / "sample.jpg")
            annotations = root / "labels.json"
            annotations.write_text(
                json.dumps(
                    [
                        {
                            "name": "sample.jpg",
                            "timestamp": 1000,
                            "attributes": {
                                "weather": "rainy",
                                "scene": "city street",
                                "timeofday": "daytime",
                            },
                            "labels": [
                                {
                                    "id": "1",
                                    "category": "car",
                                    "attributes": {"occluded": False},
                                    "box2d": {"x1": -1, "y1": 1, "x2": 8, "y2": 9},
                                },
                                {"id": "2", "category": "traffic sign"},
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            cfg = SourceDatasetConfig.model_validate(
                {
                    "key": "bdd100k",
                    "root": root,
                    "source_format": "bdd100k",
                    "splits": {
                        "train": {
                            "paths": {"images": "images", "annotations": "labels.json"}
                        }
                    },
                    "class_map": {"car": "car"},
                    "bbox_policy": "clip",
                }
            )

            source = source_from_config(cfg)
            records = list(source.iter_records("train"))

        self.assertIsInstance(source, Bdd100kSourceDataset)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].objects[0].bbox.xywh(), (0.0, 1.0, 8.0, 8.0))
        self.assertEqual(records[0].condition, "rainy")
        self.assertEqual(records[0].meta["scene"], "city street")
        self.assertEqual(records[0].objects[0].native_label_id, "1")

    def test_repo_ref_is_fail_closed_without_a_controlled_plan(self):
        cfg = SourceDatasetConfig.model_validate(
            yaml.safe_load(Path("configs/datasets/bdd100k.yaml").read_text(encoding="utf-8"))
        )
        ref = DatasetRefConfig.model_validate(
            yaml.safe_load(Path("configs/dataset_refs/bdd100k.yaml").read_text(encoding="utf-8"))
        )

        self.assertEqual(cfg.splits.keys(), {"train", "val"})
        self.assertFalse(ref.meta["controlled_plan_enabled"])
        self.assertFalse(Path("configs/plans/bdd100k_controlled.yaml").exists())


if __name__ == "__main__":
    unittest.main()
