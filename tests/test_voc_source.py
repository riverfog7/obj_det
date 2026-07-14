from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

from obj_det.datasets.models.source_config import SourceDatasetConfig
from obj_det.datasets.sources import PascalVocSourceDataset, source_from_config
from obj_det.models.plan import load_and_expand_experiment_plan
from obj_det.models.schemas import DatasetRefConfig


class PascalVocSourceTest(unittest.TestCase):
    def test_reads_inclusive_boxes_and_ignores_difficult_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "JPEGImages"
            annotations = root / "Annotations"
            image_sets = root / "ImageSets" / "Main"
            images.mkdir()
            annotations.mkdir()
            image_sets.mkdir(parents=True)
            Image.new("RGB", (20, 10)).save(images / "000001.jpg")
            (image_sets / "train.txt").write_text("000001\n", encoding="utf-8")
            (annotations / "000001.xml").write_text(
                """<annotation>
  <filename>000001.jpg</filename>
  <object><name>car</name><difficult>0</difficult><truncated>1</truncated>
    <bndbox><xmin>1</xmin><ymin>2</ymin><xmax>20</xmax><ymax>10</ymax></bndbox>
  </object>
  <object><name>person</name><difficult>1</difficult>
    <bndbox><xmin>2</xmin><ymin>2</ymin><xmax>4</xmax><ymax>5</ymax></bndbox>
  </object>
</annotation>""",
                encoding="utf-8",
            )
            cfg = SourceDatasetConfig.model_validate(
                {
                    "key": "voc",
                    "root": root,
                    "source_format": "pascal_voc",
                    "splits": {
                        "train": {
                            "paths": {
                                "images": "JPEGImages",
                                "annotations": "Annotations",
                                "image_set": "ImageSets/Main/train.txt",
                            }
                        }
                    },
                    "class_map": {"car": "car", "person": "person"},
                    "bbox_policy": "clip",
                }
            )

            source = source_from_config(cfg)
            records = list(source.iter_records("train"))

        self.assertIsInstance(source, PascalVocSourceDataset)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].objects[0].bbox.xywh(), (0.0, 1.0, 20.0, 9.0))
        self.assertTrue(records[0].objects[0].meta["truncated"])
        self.assertTrue(records[0].objects[1].ignore)
        self.assertEqual([obj.native_label for obj in records[0].valid_objects()], ["car"])

    def test_repo_config_ref_and_controlled_plan_load(self):
        cfg = SourceDatasetConfig.model_validate(
            yaml.safe_load(Path("configs/datasets/voc2007.yaml").read_text(encoding="utf-8"))
        )
        ref = DatasetRefConfig.model_validate(
            yaml.safe_load(Path("configs/dataset_refs/voc2007.yaml").read_text(encoding="utf-8"))
        )
        experiments = load_and_expand_experiment_plan(
            Path("configs/plans/voc2007_controlled.yaml")
        )

        self.assertEqual(cfg.source_format, "pascal_voc")
        self.assertEqual(ref.path, Path("datasets/voc2007"))
        self.assertEqual(len(experiments), 25)
        self.assertTrue(all(exp.dataset.path == ref.path for exp in experiments))
        self.assertTrue(all(set(exp.search_space.params) == {"learning_rate"} for exp in experiments))


if __name__ == "__main__":
    unittest.main()
