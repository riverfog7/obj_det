from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

from obj_det.datasets.models.source_config import SourceDatasetConfig
from obj_det.datasets.sources import CityscapesSourceDataset, source_from_config
from obj_det.models.schemas import DatasetRefConfig


class CityscapesSourceTest(unittest.TestCase):
    def test_converts_instance_polygons_and_group_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "leftImg8bit" / "train" / "aachen"
            annotations = root / "gtFine" / "train" / "aachen"
            images.mkdir(parents=True)
            annotations.mkdir(parents=True)
            stem = "aachen_000000_000001"
            Image.new("RGB", (20, 10)).save(images / f"{stem}_leftImg8bit.png")
            (annotations / f"{stem}_gtFine_polygons.json").write_text(
                json.dumps(
                    {
                        "imgWidth": 20,
                        "imgHeight": 10,
                        "objects": [
                            {"label": "car", "polygon": [[0, 0], [20, 0], [20, 10]]},
                            {"label": "persongroup", "polygon": [[2, 2], [4, 2], [4, 6]]},
                            {"label": "road", "polygon": [[0, 8], [20, 8], [20, 10]]},
                            {
                                "label": "truck",
                                "polygon": [[1, 1], [3, 1], [3, 3]],
                                "deleted": True,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            cfg = SourceDatasetConfig.model_validate(
                {
                    "key": "cityscapes",
                    "root": root,
                    "source_format": "cityscapes",
                    "splits": {
                        "train": {
                            "paths": {
                                "images": "leftImg8bit/train",
                                "annotations": "gtFine/train",
                            }
                        }
                    },
                    "class_map": {"car": "car", "person": "person"},
                    "ignore_labels": ["road"],
                    "bbox_policy": "clip",
                }
            )

            source = source_from_config(cfg)
            records = list(source.iter_records("train"))

        self.assertIsInstance(source, CityscapesSourceDataset)
        self.assertEqual(len(records), 1)
        self.assertEqual([obj.native_label for obj in records[0].objects], ["car", "person"])
        self.assertEqual(records[0].objects[0].bbox.xywh(), (0.0, 0.0, 20.0, 10.0))
        self.assertTrue(records[0].objects[1].iscrowd)
        self.assertEqual(records[0].meta["city"], "aachen")

    def test_repo_ref_is_fail_closed_without_a_controlled_plan(self):
        cfg = SourceDatasetConfig.model_validate(
            yaml.safe_load(Path("configs/datasets/cityscapes.yaml").read_text(encoding="utf-8"))
        )
        ref = DatasetRefConfig.model_validate(
            yaml.safe_load(
                Path("configs/dataset_refs/cityscapes.yaml").read_text(encoding="utf-8")
            )
        )

        self.assertEqual(cfg.splits.keys(), {"train", "val"})
        self.assertEqual(ref.meta["controlled_plan_enabled"], False)
        self.assertFalse(Path("configs/plans/cityscapes_controlled.yaml").exists())


if __name__ == "__main__":
    unittest.main()
