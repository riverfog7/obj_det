from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from obj_det.datasets.models.source_config import SourceDatasetConfig
from obj_det.models.plan import load_and_expand_experiment_plan
from obj_det.models.schemas import DatasetRefConfig


class CarpkConfigTest(unittest.TestCase):
    def test_repo_config_ref_and_controlled_plan_load(self):
        cfg = SourceDatasetConfig.model_validate(
            yaml.safe_load(Path("configs/datasets/carpk.yaml").read_text(encoding="utf-8"))
        )
        ref = DatasetRefConfig.model_validate(
            yaml.safe_load(Path("configs/dataset_refs/carpk.yaml").read_text(encoding="utf-8"))
        )
        experiments = load_and_expand_experiment_plan(
            Path("configs/plans/carpk_controlled.yaml")
        )

        self.assertEqual(cfg.source_format, "coco")
        self.assertEqual(cfg.class_map["0"], "car")
        self.assertEqual(cfg.splits.keys(), {"train", "val", "test"})
        self.assertEqual(ref.path, Path("datasets/carpk"))
        self.assertEqual(len(experiments), 25)
        self.assertTrue(all(exp.dataset.path == ref.path for exp in experiments))
        self.assertTrue(all(set(exp.search_space.params) == {"learning_rate"} for exp in experiments))
        self.assertEqual(
            {exp.model.key: exp for exp in experiments}["yolo26m"].train.batch_size,
            16,
        )


if __name__ == "__main__":
    unittest.main()
