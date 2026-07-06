from __future__ import annotations

import unittest
from pathlib import Path

from obj_det.models.adapters.factory import MODEL_BACKENDS, model_adapter_from_config
from obj_det.models.adapters.torchvision import TorchvisionDetectionAdapter
from obj_det.models.experiment import load_experiment_config, load_model_config, load_search_space


EXPECTED_MODEL_KEYS = {
    "yolo26n",
    "yolo26s",
    "yolo26m",
    "yolo11n",
    "yolo11s",
    "yolo11m",
    "yolov8n",
    "yolov8s",
    "yolov8m",
    "rtdetr_r50vd",
    "dfine_nano",
    "dfine_small",
    "rfdetr_nano",
    "rfdetr_small",
    "rfdetr_medium",
    "rfdetr_base",
    "detr_r50",
    "conditional_detr_r50",
    "deformable_detr",
    "yolos_tiny",
    "yolos_small",
    "fasterrcnn_r50",
    "retinanet_r50",
    "fcos_r50",
    "maskrcnn_r50_boxonly",
}


class ModelConfigMatrixTest(unittest.TestCase):
    def test_expected_model_configs_exist_and_load(self):
        model_paths = sorted(Path("configs/models").glob("*.yaml"))
        loaded = {load_model_config(path).key: path for path in model_paths}

        self.assertEqual(set(loaded), EXPECTED_MODEL_KEYS)
        for key, path in loaded.items():
            with self.subTest(key=key):
                cfg = load_model_config(path)
                self.assertEqual(path.stem, key)
                self.assertIn(cfg.backend, MODEL_BACKENDS)
                self.assertEqual(model_adapter_from_config(cfg).key, key)

    def test_hazydet_controlled_experiments_match_model_matrix(self):
        experiment_paths = sorted(Path("configs/experiments").glob("*_hazydet_controlled.yaml"))
        experiment_keys = {path.name.removesuffix("_hazydet_controlled.yaml") for path in experiment_paths}

        self.assertEqual(experiment_keys, EXPECTED_MODEL_KEYS)
        for path in experiment_paths:
            with self.subTest(path=str(path)):
                key = path.name.removesuffix("_hazydet_controlled.yaml")
                cfg = load_experiment_config(path)

                self.assertIsNotNone(cfg.model)
                self.assertIsNotNone(cfg.search_space)
                self.assertEqual(cfg.model.key, key)
                self.assertEqual(cfg.train.run_key, f"{key}_hazydet_controlled")
                self.assertEqual(cfg.tuning.study_name, f"{key}_hazydet_controlled")
                self.assertEqual(str(cfg.train.output_dir), f"runs/{key}/hazydet/controlled")
                self.assertEqual(cfg.train.logging_steps, 10)
                self.assertEqual(str(cfg.tuning.output_dir), f"runs/hpo/{key}_hazydet_controlled")
                self.assertEqual(cfg.logging.wandb.project, f"{key}_hazydet_controlled")
                self.assertEqual(cfg.classes, ["person", "bicycle", "motorcycle", "car", "bus", "truck"])

    def test_search_spaces_load(self):
        for path in sorted(Path("configs/search_spaces").glob("*.yaml")):
            with self.subTest(path=str(path)):
                cfg = load_search_space(path)
                self.assertTrue(cfg.params)

    def test_torchvision_model_builder_supports_configured_models(self):
        for path in sorted(Path("configs/models").glob("*.yaml")):
            cfg = load_model_config(path)
            if cfg.backend != "torchvision":
                continue
            with self.subTest(key=cfg.key):
                cfg = cfg.model_copy(update={"params": {**cfg.params, "min_size": 64, "max_size": 64}}, deep=True)
                adapter = TorchvisionDetectionAdapter(cfg)
                model = adapter._build_model(num_classes=7, image_size=32)
                self.assertIsNotNone(model)

    def test_torchvision_model_builder_defaults_to_preprocess_image_size(self):
        cfg = load_model_config(Path("configs/models/fasterrcnn_r50.yaml"))
        adapter = TorchvisionDetectionAdapter(cfg)
        model = adapter._build_model(num_classes=7, image_size=640)

        self.assertEqual(model.transform.min_size, (640,))
        self.assertEqual(model.transform.max_size, 640)


if __name__ == "__main__":
    unittest.main()
