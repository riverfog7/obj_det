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
        self.assertEqual(yolo.train.optimizer.name, "adamw")
        self.assertEqual(yolo.train.scheduler.total_epochs, 50)
        self.assertEqual(set(yolo.search_space.params), {"learning_rate"})

        self.assertEqual(tv.model.backend, "torchvision")
        self.assertEqual(tv.train.batch_size, 2)
        self.assertEqual(tv.train.optimizer, yolo.train.optimizer)
        self.assertEqual(tv.train.scheduler, yolo.train.scheduler)
        self.assertEqual(tv.train.classes, ["car", "person"])
        self.assertEqual(tv.train.label_mode, "meta")
        self.assertEqual(tv.train.preprocess.height, 32)
        self.assertEqual(tv.train.augmentation.color_jitter_p, 0.5)
        self.assertEqual(set(tv.search_space.params), {"learning_rate"})

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
        self.assertEqual(cfg.train.optimizer.name, "adamw")
        self.assertEqual(cfg.train.scheduler.total_epochs, 50)
        self.assertEqual(set(cfg.search_space.params), {"learning_rate"})

    def test_recipe_search_space_rejects_backend_or_model_override(self):
        with TemporaryDirectory() as tmp:
            root = self._write_plan_tree(Path(tmp))
            plan = load_experiment_plan(root / "plans" / "p.yaml")
            plan.backend_defaults["ultralytics"]["search_space_file"] = "../search_spaces/global.yaml"

            with self.assertRaisesRegex(ValueError, "Recipe .* defines search_space_file"):
                expand_experiment_plan(plan)

            del plan.backend_defaults["ultralytics"]["search_space_file"]
            plan.model_overrides["yolo"]["search_space"] = {
                "params": {"learning_rate": {"type": "float", "low": 1.0e-5, "high": 1.0e-4}}
            }
            with self.assertRaisesRegex(ValueError, "Recipe .* defines search_space_file"):
                expand_experiment_plan(plan)

    def test_recipe_search_space_path_is_relative_to_recipe(self):
        with TemporaryDirectory() as tmp:
            root = self._write_plan_tree(Path(tmp))
            recipe_path = root / "recipes" / "controlled.yaml"
            recipe_space_dir = root / "recipes" / "spaces"
            recipe_space_dir.mkdir()
            (recipe_space_dir / "recipe_global.yaml").write_text(
                "params:\n  learning_rate:\n    type: float\n    low: 0.000002\n    high: 0.002\n    log: true\n",
                encoding="utf-8",
            )
            recipe_path.write_text(
                recipe_path.read_text(encoding="utf-8").replace(
                    "search_space_file: ../search_spaces/global.yaml",
                    "search_space_file: spaces/recipe_global.yaml",
                ),
                encoding="utf-8",
            )

            exps = load_and_expand_experiment_plan(root / "plans" / "p.yaml")

        self.assertEqual(exps[0].search_space.params["learning_rate"]["low"], 2.0e-6)

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
        paths = sorted(Path("configs/search_spaces").glob("*.yaml"))

        self.assertEqual(paths, [Path("configs/search_spaces/global_learning_rate.yaml")])
        search_space = load_search_space(paths[0])
        self.assertEqual(
            search_space.params,
            {
                "learning_rate": {
                    "type": "float",
                    "low": 3.0e-6,
                    "high": 3.0e-3,
                    "log": True,
                }
            },
        )

    def test_invalid_search_space_fails_early(self):
        with self.assertRaisesRegex(ValueError, "bad"):
            SearchSpace(params={"bad": {"type": "categorical", "choices": []}})

    def test_existing_direct_experiment_configs_still_load(self):
        cfg = load_experiment_config(Path("configs/experiments/yolo26m_hazydet_controlled.yaml"))

        self.assertEqual(cfg.model.key, "yolo26m")
        self.assertIsNotNone(cfg.search_space)
        self.assertEqual(set(cfg.search_space.params), {"learning_rate"})
        self.assertEqual(cfg.train.optimizer.name, "adamw")
        self.assertEqual(cfg.train.scheduler.total_epochs, 50)
        self.assertTrue(cfg.tuning.catch_trial_errors)

    def test_repo_controlled_plan_applies_one_protocol_to_every_backend(self):
        exps = load_and_expand_experiment_plan(Path("configs/plans/hazydet_controlled.yaml"))

        self.assertEqual(len(exps), 25)
        by_key = {exp.model.key: exp for exp in exps if exp.model is not None}
        for exp in exps:
            with self.subTest(model=exp.model.key):
                self.assertEqual(exp.train.optimizer.name, "adamw")
                self.assertEqual(exp.train.optimizer.weight_decay, 1.0e-4)
                self.assertEqual(exp.train.optimizer.beta1, 0.9)
                self.assertEqual(exp.train.optimizer.beta2, 0.999)
                self.assertEqual(exp.train.optimizer.epsilon, 1.0e-8)
                self.assertEqual(exp.train.scheduler.name, "warmup_cosine")
                self.assertEqual(exp.train.scheduler.warmup_epochs, 1.0)
                self.assertEqual(exp.train.scheduler.total_epochs, 50)
                self.assertEqual(exp.train.scheduler.min_lr_ratio, 0.01)
                self.assertEqual(exp.tuning.n_trials, 8)
                self.assertEqual(exp.tuning.trial_epochs, 10)
                self.assertEqual(exp.tuning.sampler_params, {"n_startup_trials": 3})
                self.assertEqual(exp.tuning.pruner, "none")
                self.assertEqual(exp.tuning.save_strategy, "final_only")
                self.assertTrue(exp.tuning.catch_trial_errors)
                self.assertEqual(set(exp.search_space.params), {"learning_rate"})
                self.assertTrue(exp.train.eval_strategy.enabled)
                self.assertTrue(exp.eval.compute_per_class)
                self.assertTrue(exp.eval.compute_per_condition)
                self.assertTrue(exp.eval.compute_per_domain)
                self.assertTrue(exp.eval.compute_per_size)
                self.assertEqual(exp.eval.max_detections_per_image, 300)
                self.assertEqual(exp.predict.max_detections_per_image, 300)

        for exp in exps:
            self.assertEqual(exp.train.batch_size, 16)
            self.assertEqual(exp.eval.batch_size, 16)
            self.assertEqual(exp.predict.batch_size, 16)

    def test_repo_controlled_matrix_expands_all_ten_plans(self):
        expected_dataset_keys = {
            "carpk",
            "dawn",
            "exdark",
            "hazydet",
            "hazydet_clear",
            "merged_traffic6",
            "udacity",
            "visdrone",
            "voc2007",
            "xwod",
        }
        plan_paths = sorted(Path("configs/plans").glob("*_controlled.yaml"))

        self.assertEqual(
            {path.stem.removesuffix("_controlled") for path in plan_paths},
            expected_dataset_keys,
        )

        total_experiments = 0
        for path in plan_paths:
            dataset_key = path.stem.removesuffix("_controlled")
            plan = load_experiment_plan(path)
            exps = expand_experiment_plan(plan)
            with self.subTest(plan=path.name):
                self.assertEqual(plan.recipe_file, Path("../recipes/controlled_native.yaml"))
                self.assertEqual(plan.class_space_file, Path("../class_spaces/traffic6.yaml"))
                self.assertEqual(plan.model_group_file, Path("../model_groups/detection_main.yaml"))
                self.assertEqual(len(exps), 25)
                self.assertTrue(
                    all(exp.dataset.path == Path(f"datasets/{dataset_key}") for exp in exps)
                )
                self.assertTrue(all(exp.train.batch_size == 16 for exp in exps))
                self.assertTrue(all(exp.eval.batch_size == 16 for exp in exps))
                self.assertTrue(all(exp.predict.batch_size == 16 for exp in exps))
                self.assertTrue(
                    all(set(exp.search_space.params) == {"learning_rate"} for exp in exps)
                )
            total_experiments += len(exps)

        self.assertEqual(total_experiments, 250)

    def _write_plan_tree(self, root: Path, *, include_torchvision_default: bool = True) -> Path:
        for name in [
            "dataset_refs",
            "class_spaces",
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
        (root / "augmentations" / "basic.yaml").write_text(
            "policy: basic\nhorizontal_flip_p: 0.5\ncolor_jitter_strength: 0.05\ncolor_jitter_p: 0.5\n",
            encoding="utf-8",
        )
        (root / "recipes" / "controlled.yaml").write_text(
            "\n".join(
                [
                    "protocol: controlled",
                    "augmentation_file: ../augmentations/basic.yaml",
                    "search_space_file: ../search_spaces/global.yaml",
                    "train:",
                    "  max_epochs: 50",
                    "  optimizer:",
                    "    name: adamw",
                    "  scheduler:",
                    "    name: warmup_cosine",
                    "    warmup_epochs: 1",
                    "    total_epochs: 50",
                    "    min_lr_ratio: 0.01",
                    "eval:",
                    "  conf_threshold: 0.001",
                    "predict:",
                    "  conf_threshold: 0.001",
                    "tuning:",
                    "  n_trials: 8",
                    "  trial_epochs: 10",
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
            "key: yolo\nbackend: ultralytics\ndetector_pretraining_dataset: coco\nmodel_name_or_path: yolo.pt\n"
            "preprocess:\n  resize_mode: letterbox\n  height: 32\n  width: 32\n",
            encoding="utf-8",
        )
        (root / "models" / "tv.yaml").write_text(
            "key: tv\nbackend: torchvision\ndetector_pretraining_dataset: coco\nmodel_name_or_path: fasterrcnn_resnet50_fpn\n"
            "preprocess:\n  resize_mode: letterbox\n  height: 32\n  width: 32\n",
            encoding="utf-8",
        )
        (root / "model_groups" / "g.yaml").write_text(
            "models:\n  - ../models/yolo.yaml\n  - ../models/tv.yaml\n",
            encoding="utf-8",
        )
        (root / "search_spaces" / "global.yaml").write_text(
            "params:\n  learning_rate:\n    type: float\n    low: 0.000001\n    high: 0.001\n    log: true\n",
            encoding="utf-8",
        )

        torchvision_default = ""
        if include_torchvision_default:
            torchvision_default = "\n".join(
                [
                    "  torchvision:",
                    "    train:",
                    "      batch_size: 2",
                    "    eval:",
                    "      batch_size: 2",
                    "    predict:",
                    "      batch_size: 2",
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
                    "    eval:",
                    "      batch_size: 4",
                    "    predict:",
                    "      batch_size: 4",
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
