from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from obj_det.datasets.models.source_config import SourceDatasetConfig
from obj_det.datasets.sources import SOURCE_FORMATS
from obj_det.models.plan import load_and_expand_experiment_plan
from obj_det.models.schemas import DatasetRefConfig


RUNNABLE_DATASETS = {
    "carpk",
    "dawn",
    "exdark",
    "hazydet",
    "hazydet_clear",
    "udacity",
    "visdrone",
    "voc2007",
    "xwod",
}
MERGED_DATASETS = {"merged_traffic6"}
RUNNABLE_PLAN_DATASETS = RUNNABLE_DATASETS | MERGED_DATASETS
FAIL_CLOSED_DATASETS = {
    "acdc",
    "bdd100k",
    "hazydet_real",
}
ALL_DATASETS = RUNNABLE_DATASETS | FAIL_CLOSED_DATASETS


class DatasetConfigMatrixTest(unittest.TestCase):
    def test_all_dataset_configs_and_refs_validate(self):
        dataset_paths = sorted(Path("configs/datasets").glob("*.yaml"))
        ref_paths = sorted(Path("configs/dataset_refs").glob("*.yaml"))

        self.assertEqual({path.stem for path in dataset_paths}, ALL_DATASETS)
        self.assertEqual(
            {path.stem for path in ref_paths}, ALL_DATASETS | MERGED_DATASETS
        )

        for dataset_path in dataset_paths:
            with self.subTest(dataset=dataset_path.stem):
                cfg = SourceDatasetConfig.model_validate(
                    yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
                )
                ref_path = Path("configs/dataset_refs") / dataset_path.name
                ref = DatasetRefConfig.model_validate(
                    yaml.safe_load(ref_path.read_text(encoding="utf-8"))
                )

                self.assertEqual(cfg.key, dataset_path.stem)
                self.assertEqual(ref.key, dataset_path.stem)
                self.assertIn(cfg.source_format, SOURCE_FORMATS)
                self.assertEqual(ref.default_class_space, "traffic6")

                plan_path = (
                    Path("configs/plans") / f"{dataset_path.stem}_controlled.yaml"
                )
                if dataset_path.stem in RUNNABLE_DATASETS:
                    self.assertTrue(plan_path.exists())
                    self.assertTrue(
                        {ref.train_split, ref.val_split, ref.test_split}
                        <= set(cfg.splits)
                    )
                else:
                    self.assertFalse(plan_path.exists())
                    self.assertFalse(ref.meta["controlled_plan_enabled"])
                    self.assertTrue(ref.meta["controlled_plan_blocker"])

        merged_ref = DatasetRefConfig.model_validate(
            yaml.safe_load(
                Path("configs/dataset_refs/merged_traffic6.yaml").read_text(
                    encoding="utf-8"
                )
            )
        )
        self.assertEqual(merged_ref.path, Path("datasets/merged_traffic6"))
        self.assertEqual(
            (merged_ref.train_split, merged_ref.val_split, merged_ref.test_split),
            ("train", "val", "test"),
        )

    def test_all_controlled_plans_expand_with_identical_protocols(self):
        plan_paths = sorted(Path("configs/plans").glob("*_controlled.yaml"))
        self.assertEqual(
            {path.stem.removesuffix("_controlled") for path in plan_paths},
            RUNNABLE_PLAN_DATASETS,
        )

        baseline = None
        total_experiments = 0
        for plan_path in plan_paths:
            experiments = load_and_expand_experiment_plan(plan_path)
            total_experiments += len(experiments)
            self.assertEqual(len(experiments), 25)
            self.assertTrue(
                all(
                    exp.dataset.path
                    == Path("datasets") / plan_path.stem.removesuffix("_controlled")
                    for exp in experiments
                )
            )
            self.assertTrue(
                all(
                    set(exp.search_space.params) == {"learning_rate"}
                    for exp in experiments
                )
            )

            current = {
                exp.model.key: {
                    "train": exp.train.model_dump(exclude={"run_key", "output_dir"}),
                    "eval": exp.eval.model_dump(),
                    "predict": exp.predict.model_dump(),
                    "tuning": exp.tuning.model_dump(
                        exclude={"study_name", "output_dir"}
                    ),
                    "final": exp.final.model_dump(exclude={"output_dir"}),
                    "search_space": exp.search_space.model_dump(),
                }
                for exp in experiments
            }
            if baseline is None:
                baseline = current
            else:
                self.assertEqual(current, baseline)

        self.assertEqual(total_experiments, 250)


if __name__ == "__main__":
    unittest.main()
