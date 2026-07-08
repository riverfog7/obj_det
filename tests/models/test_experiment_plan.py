from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from obj_det.models.experiment import load_experiment_config, load_search_space
from obj_det.models.plan import (
    deep_merge,
    expand_experiment_plan,
    load_and_expand_experiment_plan,
    load_experiment_plan,
    write_resolved_experiments,
)
from obj_det.models.schemas import ExperimentConfig
from obj_det.models.schemas.tuning import SearchSpace


class ExperimentPlanTest(unittest.TestCase):
    def test_deep_merge_merges_dicts_and_replaces_lists(self):
        merged = deep_merge(
            {
                "train": {"loader": {"num_workers": 2, "pin_memory": False}, "hparams": {"lr": 1}},
                "classes": ["car", "person"],
            },
            {
                "train": {"loader": {"num_workers": 8}, "hparams": {"momentum": 0.9}},
                "classes": ["truck"],
            },
        )

        self.assertEqual(merged["train"]["loader"], {"num_workers": 8, "pin_memory": False})
        self.assertEqual(merged["train"]["hparams"], {"lr": 1, "momentum": 0.9})
        self.assertEqual(merged["classes"], ["truck"])

    def test_plan_expands_backend_defaults_model_overrides_and_templates(self):
        with TemporaryDirectory() as tmp:
            root = self._write_plan_tree(Path(tmp))
            exps = load_and_expand_experiment_plan(root / "plans" / "p.yaml")

        self.assertEqual([exp.model.key for exp in exps if exp.model is not None], ["yolo", "tv"])
        yolo, tv = exps
        self.assertIsInstance(yolo, ExperimentConfig)
        self.assertEqual(yolo.dataset.path, Path("datasets/tiny"))
        self.assertEqual(yolo.train.run_key, "yolo_tiny_controlled_p")
        self.assertEqual(yolo.train.output_dir, Path("runs/yolo/tiny/controlled"))
        self.assertEqual(yolo.tuning.study_name, "yolo_tiny_controlled")
        self.assertEqual(yolo.tuning.output_dir, Path("runs/hpo/yolo_tiny_controlled"))
        self.assertEqual(yolo.final.output_dir, Path("runs/yolo/tiny/controlled/final"))
        self.assertEqual(yolo.logging.wandb.project, "tiny_controlled")
        self.assertEqual(yolo.logging.wandb.tags, ["unit"])
        self.assertEqual(yolo.train.batch_size, 16)
        self.assertEqual(yolo.eval.batch_size, 4)
        self.assertEqual(yolo.predict.batch_size, 4)
        self.assertEqual(yolo.train.hparams["optimizer"], "SGD")
        self.assertIn("lr0", yolo.search_space.params)

        self.assertEqual(tv.model.backend, "torchvision")
        self.assertEqual(tv.train.batch_size, 2)
        self.assertEqual(tv.train.hparams["optimizer"], "sgd")
        self.assertEqual(tv.train.classes, ["car", "person"])
        self.assertEqual(tv.train.label_mode, "meta")
        self.assertEqual(tv.train.preprocess.image_size, 32)
        self.assertEqual(tv.train.augmentation.color_jitter_p, 0.5)
        self.assertIn("learning_rate", tv.search_space.params)

    def test_plan_can_filter_model_keys(self):
        with TemporaryDirectory() as tmp:
            root = self._write_plan_tree(Path(tmp))
            plan = load_experiment_plan(root / "plans" / "p.yaml")
            exps = expand_experiment_plan(plan, model_keys=["tv"])

        self.assertEqual(len(exps), 1)
        self.assertEqual(exps[0].model.key, "tv")

    def test_missing_backend_default_fails_before_training(self):
        with TemporaryDirectory() as tmp:
            root = self._write_plan_tree(Path(tmp), include_torchvision_default=False)
            plan = load_experiment_plan(root / "plans" / "p.yaml")
            with self.assertRaisesRegex(ValueError, "backend_defaults"):
                expand_experiment_plan(plan)

    def test_resolved_config_writer_outputs_reloadable_experiments(self):
        with TemporaryDirectory() as tmp:
            root = self._write_plan_tree(Path(tmp))
            plan = load_experiment_plan(root / "plans" / "p.yaml")
            paths = write_resolved_experiments(plan, root / "resolved", model_keys=["yolo"])
            cfg = load_experiment_config(paths[0])

        self.assertEqual(len(paths), 1)
        self.assertEqual(cfg.model.key, "yolo")
        self.assertIsNotNone(cfg.search_space)
        self.assertEqual(cfg.train.hparams["lrf"], 0.01)

    def test_repo_model_groups_are_dataset_agnostic_and_loadable(self):
        import yaml

        from obj_det.models.experiment import load_model_config
        from obj_det.models.schemas import ModelGroupConfig

        group_dir = Path("configs/model_groups")
        self.assertFalse((group_dir / "hazydet_main.yaml").exists())

        expected_groups = {
            "detection_main.yaml",
            "yolo_family.yaml",
            "hf_transformers.yaml",
            "torchvision.yaml",
        }
        self.assertTrue(expected_groups.issubset({path.name for path in group_dir.glob("*.yaml")}))

        for group_path in group_dir.glob("*.yaml"):
            with self.subTest(group=group_path.name):
                group = ModelGroupConfig.model_validate(yaml.safe_load(group_path.read_text(encoding="utf-8")))
                for model_path in group.models:
                    cfg = load_model_config(group_path.parent / model_path)
                    self.assertTrue(cfg.key)

    def test_repo_search_spaces_validate(self):
        for path in [
            Path("configs/search_spaces/yolo_controlled_sgd.yaml"),
            Path("configs/search_spaces/hf_transformer_controlled.yaml"),
            Path("configs/search_spaces/torchvision_sgd_controlled.yaml"),
        ]:
            with self.subTest(path=path):
                self.assertTrue(load_search_space(path).params)

    def test_invalid_search_space_fails_early(self):
        with self.assertRaisesRegex(ValueError, "bad"):
            SearchSpace(params={"bad": {"type": "categorical", "choices": []}})

    def test_existing_direct_experiment_configs_still_load(self):
        cfg = load_experiment_config(Path("configs/experiments/yolo26s_hazydet_controlled.yaml"))

        self.assertEqual(cfg.model.key, "yolo26s")
        self.assertIsNotNone(cfg.search_space)

    def _write_plan_tree(self, root: Path, *, include_torchvision_default: bool = True) -> Path:
        for name in [
            "dataset_refs",
            "class_spaces",
            "preprocess",
            "augmentations",
            "recipes",
            "models",
            "model_groups",
            "search_spaces",
            "plans",
        ]:
            (root / name).mkdir(parents=True, exist_ok=True)

        (root / "dataset_refs" / "tiny.yaml").write_text(
            "key: tiny\npath: datasets/tiny\ntrain_split: train\nval_split: val\ntest_split: test\n",
            encoding="utf-8",
        )
        (root / "class_spaces" / "traffic.yaml").write_text(
            "label_mode: meta\nclasses: [car, person]\n",
            encoding="utf-8",
        )
        (root / "preprocess" / "32.yaml").write_text("image_size: 32\n", encoding="utf-8")
        (root / "augmentations" / "basic.yaml").write_text(
            "policy: basic\nhorizontal_flip_p: 0.5\ncolor_jitter_strength: 0.05\ncolor_jitter_p: 0.5\n",
            encoding="utf-8",
        )
        (root / "recipes" / "controlled.yaml").write_text(
            "\n".join(
                [
                    "protocol: controlled",
                    "preprocess_file: ../preprocess/32.yaml",
                    "augmentation_file: ../augmentations/basic.yaml",
                    "train:",
                    "  max_epochs: 1",
                    "eval:",
                    "  conf_threshold: 0.001",
                    "predict:",
                    "  conf_threshold: 0.001",
                    "tuning:",
                    "  n_trials: 1",
                    "  pruner: none",
                    "  objective_metric: map_50_95",
                    "final:",
                    "  seeds: [0]",
                    "logging:",
                    "  backends: [local, wandb]",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (root / "models" / "yolo.yaml").write_text(
            "key: yolo\nbackend: ultralytics\nmodel_name_or_path: yolo.pt\n",
            encoding="utf-8",
        )
        (root / "models" / "tv.yaml").write_text(
            "key: tv\nbackend: torchvision\nmodel_name_or_path: fasterrcnn_resnet50_fpn\n",
            encoding="utf-8",
        )
        (root / "model_groups" / "g.yaml").write_text(
            "models:\n  - ../models/yolo.yaml\n  - ../models/tv.yaml\n",
            encoding="utf-8",
        )
        (root / "search_spaces" / "yolo.yaml").write_text(
            "params:\n  lr0:\n    type: float\n    low: 0.001\n    high: 0.01\n    log: true\n",
            encoding="utf-8",
        )
        (root / "search_spaces" / "tv.yaml").write_text(
            "params:\n  learning_rate:\n    type: float\n    low: 0.001\n    high: 0.01\n    log: true\n",
            encoding="utf-8",
        )

        torchvision_default = ""
        if include_torchvision_default:
            torchvision_default = "\n".join(
                [
                    "  torchvision:",
                    "    train:",
                    "      batch_size: 2",
                    "      hparams:",
                    "        optimizer: sgd",
                    "        learning_rate: 0.005",
                    "    eval:",
                    "      batch_size: 2",
                    "    predict:",
                    "      batch_size: 2",
                    "    search_space_file: ../search_spaces/tv.yaml",
                ]
            )

        (root / "plans" / "p.yaml").write_text(
            "\n".join(
                [
                    "key: p",
                    "dataset_file: ../dataset_refs/tiny.yaml",
                    "class_space_file: ../class_spaces/traffic.yaml",
                    "recipe_file: ../recipes/controlled.yaml",
                    "model_group_file: ../model_groups/g.yaml",
                    "run_template:",
                    "  run_key: '{model_key}_{dataset_key}_{protocol}_{plan_key}'",
                    "  output_dir: 'runs/{model_key}/{dataset_key}/{protocol}'",
                    "  tuning_study_name: '{model_key}_{dataset_key}_{protocol}'",
                    "  tuning_output_dir: 'runs/hpo/{model_key}_{dataset_key}_{protocol}'",
                    "  final_output_dir: 'runs/{model_key}/{dataset_key}/{protocol}/final'",
                    "  wandb_project: '{dataset_key}_{protocol}'",
                    "backend_defaults:",
                    "  ultralytics:",
                    "    train:",
                    "      batch_size: 8",
                    "      hparams:",
                    "        optimizer: SGD",
                    "        lr0: 0.003",
                    "        lrf: 0.01",
                    "    eval:",
                    "      batch_size: 4",
                    "    predict:",
                    "      batch_size: 4",
                    "    search_space_file: ../search_spaces/yolo.yaml",
                    torchvision_default,
                    "model_overrides:",
                    "  yolo:",
                    "    train:",
                    "      batch_size: 16",
                    "tags: [unit]",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return root


if __name__ == "__main__":
    unittest.main()
