from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

from obj_det.datasets.models.source_config import SourceDatasetConfig
from obj_det.datasets.sources import source_from_config
from obj_det.models.schemas import DatasetRefConfig
from scripts.prepare_nuscenes import CAMERA_CHANNELS, infer_condition


REPO_ROOT = Path(__file__).resolve().parents[1]


class NuscenesConfigTest(unittest.TestCase):
    def test_generated_coco_preserves_projection_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (20, 10)).save(root / "sample.jpg")
            annotations = root / "train.json"
            annotations.write_text(
                json.dumps(
                    {
                        "images": [
                            {
                                "id": 1,
                                "file_name": "sample.jpg",
                                "width": 20,
                                "height": 10,
                                "condition": "night",
                                "domain": "road",
                                "is_synthetic": False,
                                "meta": {"channel": "CAM_FRONT", "sample_token": "sample"},
                            }
                        ],
                        "categories": [{"id": 1, "name": "vehicle.car"}],
                        "annotations": [
                            {
                                "id": 1,
                                "image_id": 1,
                                "category_id": 1,
                                "bbox": [1, 2, 3, 4],
                                "meta": {"visibility_token": "4"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            cfg = SourceDatasetConfig.model_validate(
                {
                    "key": "nuscenes",
                    "root": root,
                    "source_format": "coco",
                    "splits": {
                        "train": {"paths": {"images": ".", "annotations": "train.json"}}
                    },
                    "class_map": {"vehicle.car": "car"},
                }
            )

            records = list(source_from_config(cfg).iter_records("train"))

        self.assertEqual(records[0].condition, "night")
        self.assertEqual(records[0].domain, "road")
        self.assertEqual(records[0].meta["channel"], "CAM_FRONT")
        self.assertEqual(records[0].objects[0].meta["visibility_token"], "4")

    def test_preparation_helpers_do_not_require_devkit_for_help(self):
        result = subprocess.run(
            [".venv/bin/python", "scripts/prepare_nuscenes.py", "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("COCO camera boxes", result.stdout)
        self.assertEqual(len(CAMERA_CHANNELS), 6)
        self.assertEqual(infer_condition("Night scene after rain"), "rain")
        self.assertEqual(infer_condition("Night scene"), "night")
        self.assertEqual(infer_condition("Singapore daytime"), "clear")

    def test_repo_ref_is_fail_closed_without_a_controlled_plan(self):
        cfg = SourceDatasetConfig.model_validate(
            yaml.safe_load(Path("configs/datasets/nuscenes.yaml").read_text(encoding="utf-8"))
        )
        ref = DatasetRefConfig.model_validate(
            yaml.safe_load(Path("configs/dataset_refs/nuscenes.yaml").read_text(encoding="utf-8"))
        )

        self.assertEqual(cfg.source_format, "coco")
        self.assertEqual(cfg.splits.keys(), {"train", "val"})
        self.assertFalse(ref.meta["controlled_plan_enabled"])
        self.assertFalse(Path("configs/plans/nuscenes_controlled.yaml").exists())


if __name__ == "__main__":
    unittest.main()
