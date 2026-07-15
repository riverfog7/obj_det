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

EXPECTED_PREPROCESS = {
    **{
        key: ("letterbox", 640, 640, None, None)
        for key in {
            "yolo26n", "yolo26s", "yolo26m",
            "yolo11n", "yolo11s", "yolo11m",
            "yolov8n", "yolov8s", "yolov8m",
        }
    },
    **{
        key: ("exact", 640, 640, None, None)
        for key in {"rtdetr_r50vd", "dfine_nano", "dfine_small"}
    },
    "rfdetr_nano": ("exact", 384, 384, None, None),
    "rfdetr_small": ("exact", 512, 512, None, None),
    "rfdetr_base": ("exact", 560, 560, None, None),
    "rfdetr_medium": ("exact", 576, 576, None, None),
    "yolos_tiny": ("shortest_edge", None, None, 512, 1333),
    **{
        key: ("shortest_edge", None, None, 800, 1333)
        for key in {
            "detr_r50", "conditional_detr_r50", "deformable_detr", "yolos_small",
            "fasterrcnn_r50", "retinanet_r50", "fcos_r50", "maskrcnn_r50_boxonly",
        }
    },
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
                if cfg.backend == "torchvision":
                    self.assertEqual(cfg.weights, "DEFAULT")
                preprocess = cfg.preprocess
                self.assertEqual(
                    (
                        preprocess.resize_mode,
                        preprocess.height,
                        preprocess.width,
                        preprocess.shortest_edge,
                        preprocess.longest_edge,
                    ),
                    EXPECTED_PREPROCESS[key],
                )
                self.assertEqual(model_adapter_from_config(cfg).key, key)

    def test_debug_experiment_config_loads(self):
        experiment_paths = sorted(Path("configs/experiments").glob("*_hazydet_controlled.yaml"))

        self.assertEqual([path.name for path in experiment_paths], ["yolo26m_hazydet_controlled.yaml"])

        cfg = load_experiment_config(experiment_paths[0])

        self.assertIsNotNone(cfg.model)
        self.assertIsNotNone(cfg.search_space)
        self.assertEqual(cfg.model.key, "yolo26m")
        self.assertEqual(cfg.train.run_key, "yolo26m_hazydet_controlled")
        self.assertEqual(cfg.tuning.study_name, "yolo26m_hazydet_controlled")
        self.assertEqual(cfg.tuning.n_trials, 8)
        self.assertEqual(cfg.tuning.trial_epochs, 10)
        self.assertEqual(cfg.tuning.sampler_params, {"n_startup_trials": 3})
        self.assertEqual(str(cfg.train.output_dir), "runs/yolo26m/hazydet/controlled")
        self.assertEqual(cfg.train.logging_steps, 100)
        self.assertEqual(cfg.train.optimizer.name, "adamw")
        self.assertEqual(cfg.train.scheduler.total_epochs, 50)
        self.assertEqual(str(cfg.tuning.output_dir), "runs/hpo/yolo26m_hazydet_controlled")
        self.assertEqual(cfg.logging.wandb.project, "yolo26m_hazydet_controlled")
        self.assertEqual(cfg.classes, ["person", "bicycle", "motorcycle", "car", "bus", "truck"])
        self.assertEqual(set(cfg.search_space.params), {"learning_rate"})

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
                adapter = TorchvisionDetectionAdapter(cfg)
                model = adapter._build_model(num_classes=7, preprocess=cfg.preprocess)
                self.assertIsNotNone(model)

    def test_torchvision_model_builder_uses_native_preprocess_size(self):
        cfg = load_model_config(Path("configs/models/fasterrcnn_r50.yaml"))
        adapter = TorchvisionDetectionAdapter(cfg)
        model = adapter._build_model(num_classes=7, preprocess=cfg.preprocess)

        self.assertEqual(model.transform.min_size, (800,))
        self.assertEqual(model.transform.max_size, 1333)


if __name__ == "__main__":
    unittest.main()
