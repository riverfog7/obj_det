from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

from obj_det.datasets.models.source_config import SourceDatasetConfig
from obj_det.datasets.sources import CocoSourceDataset, source_from_config
from obj_det.models.schemas import DatasetRefConfig


class AcdcSourceTest(unittest.TestCase):
    def test_reuses_coco_loader_and_derives_weather_condition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "rgb_anon" / "fog" / "train"
            images.mkdir(parents=True)
            Image.new("RGB", (20, 10)).save(images / "sample.png")
            annotations = root / "train.json"
            annotations.write_text(
                json.dumps(
                    {
                        "images": [
                            {
                                "id": 1,
                                "file_name": "fog/train/sample.png",
                                "width": 20,
                                "height": 10,
                            }
                        ],
                        "categories": [{"id": 1, "name": "car"}],
                        "annotations": [
                            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 3, 4]}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            cfg = SourceDatasetConfig.model_validate(
                {
                    "key": "acdc",
                    "root": root,
                    "source_format": "coco",
                    "splits": {
                        "train": {
                            "paths": {"images": "rgb_anon", "annotations": "train.json"},
                            "meta": {"condition_from_file_name_part": 0},
                        }
                    },
                    "class_map": {"car": "car"},
                    "bbox_policy": "clip",
                }
            )

            source = source_from_config(cfg)
            records = list(source.iter_records("train"))

        self.assertIsInstance(source, CocoSourceDataset)
        self.assertEqual(records[0].condition, "fog")
        self.assertEqual(records[0].objects[0].meta_label, "car")

    def test_repo_ref_is_fail_closed_without_a_controlled_plan(self):
        cfg = SourceDatasetConfig.model_validate(
            yaml.safe_load(Path("configs/datasets/acdc.yaml").read_text(encoding="utf-8"))
        )
        ref = DatasetRefConfig.model_validate(
            yaml.safe_load(Path("configs/dataset_refs/acdc.yaml").read_text(encoding="utf-8"))
        )

        self.assertEqual(cfg.source_format, "coco")
        self.assertEqual(cfg.splits.keys(), {"train", "val"})
        self.assertFalse(ref.meta["controlled_plan_enabled"])
        self.assertFalse(Path("configs/plans/acdc_controlled.yaml").exists())


if __name__ == "__main__":
    unittest.main()
