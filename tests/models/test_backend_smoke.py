from __future__ import annotations

import os
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from datasets import Dataset

from obj_det.models.adapters.factory import model_adapter_from_config
from obj_det.models.schemas import DataLoaderConfig, EvalConfig, ModelConfig, PredictConfig, PreprocessConfig, TrainConfig

from .helpers import row


@contextmanager
def working_directory(path: Path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def tiny_dataset(size=(64, 64)):
    return Dataset.from_list([
        row(size=size, objects=[{
            "bbox": [8.0, 8.0, 20.0, 20.0],
            "native_label": "car",
            "native_label_id": "1",
            "meta_label": "car",
            "ignore": False,
            "iscrowd": False,
            "meta_json": "{}",
        }])
    ])


@unittest.skipUnless(os.environ.get("OBJ_DET_RUN_BACKEND_SMOKE") == "1", "set OBJ_DET_RUN_BACKEND_SMOKE=1")
class BackendSmokeTest(unittest.TestCase):
    def test_hf_trainer_smoke(self):
        ds = tiny_dataset()
        with TemporaryDirectory() as tmp:
            preprocess = PreprocessConfig(image_size=64)
            adapter = model_adapter_from_config(ModelConfig(
                key="tiny_detr",
                backend="hf_trainer",
                model_name_or_path="hf-internal-testing/tiny-random-detr",
            ))
            artifact = adapter.train(ds, ds, TrainConfig(
                run_key="hf_smoke",
                classes=["car"],
                output_dir=Path(tmp) / "hf",
                preprocess=preprocess,
                loader=DataLoaderConfig(pin_memory=False),
                max_epochs=1,
                max_steps=1,
                batch_size=1,
                amp=False,
                logging_steps=1,
            ))
            preds = list(adapter.predict(ds, artifact, PredictConfig(classes=["car"], preprocess=preprocess, batch_size=1, conf_threshold=0.0)))
            result = adapter.evaluate(ds, artifact, EvalConfig(classes=["car"], preprocess=preprocess, conf_threshold=0.0))

        self.assertIsNotNone(artifact.checkpoint_path)
        self.assertEqual(len(preds), 1)
        self.assertEqual(result.num_images, 1)

    def test_ultralytics_smoke(self):
        ds = tiny_dataset(size=(96, 96))
        with TemporaryDirectory() as tmp, working_directory(Path(tmp)):
            preprocess = PreprocessConfig(image_size=96)
            adapter = model_adapter_from_config(ModelConfig(
                key="yolo_smoke",
                backend="ultralytics",
                model_name_or_path=os.environ.get("OBJ_DET_YOLO_SMOKE_MODEL", "yolo11n.pt"),
            ))
            artifact = adapter.train(ds, ds, TrainConfig(
                run_key="yolo_smoke",
                classes=["car"],
                output_dir=Path(tmp) / "yolo",
                preprocess=preprocess,
                loader=DataLoaderConfig(pin_memory=False),
                max_epochs=1,
                max_steps=1,
                batch_size=1,
                amp=False,
                hparams={"learning_rate": 0.001},
                backend_params={"device": "cpu", "overrides": {"verbose": False}},
            ))
            preds = list(adapter.predict(ds, artifact, PredictConfig(
                classes=["car"], preprocess=preprocess, batch_size=1, conf_threshold=0.0, backend_params={"device": "cpu"}
            )))
            result = adapter.evaluate(ds, artifact, EvalConfig(
                classes=["car"], preprocess=preprocess, conf_threshold=0.0, backend_params={"device": "cpu"}
            ))

        self.assertIsNotNone(artifact.checkpoint_path)
        self.assertEqual(len(preds), 1)
        self.assertEqual(result.num_images, 1)

    def test_torchvision_smoke(self):
        ds = tiny_dataset()
        with TemporaryDirectory() as tmp:
            preprocess = PreprocessConfig(image_size=64)
            adapter = model_adapter_from_config(ModelConfig(
                key="torchvision_smoke",
                backend="torchvision",
                model_name_or_path="fasterrcnn_resnet50_fpn",
                params={"weights": None, "min_size": 64, "max_size": 64},
            ))
            artifact = adapter.train(ds, ds, TrainConfig(
                run_key="torchvision_smoke",
                classes=["car"],
                output_dir=Path(tmp) / "torchvision",
                preprocess=preprocess,
                loader=DataLoaderConfig(pin_memory=False),
                max_epochs=1,
                max_steps=1,
                batch_size=1,
                amp=False,
                logging_steps=1,
                hparams={"learning_rate": 0.001},
            ))
            preds = list(adapter.predict(ds, artifact, PredictConfig(
                classes=["car"], preprocess=preprocess, batch_size=1, conf_threshold=0.0
            )))
            result = adapter.evaluate(ds, artifact, EvalConfig(
                classes=["car"], preprocess=preprocess, conf_threshold=0.0
            ))

        self.assertIsNotNone(artifact.checkpoint_path)
        self.assertEqual(len(preds), 1)
        self.assertEqual(result.num_images, 1)


if __name__ == "__main__":
    unittest.main()
