from __future__ import annotations

import pathlib
import unittest


class BoundaryTest(unittest.TestCase):
    def test_model_layer_does_not_import_raw_sources(self):
        root = pathlib.Path("src/obj_det/models")
        forbidden = ("obj_det.datasets.sources", "BaseSourceDataset", "source_from_config")
        offenders = []
        for path in root.rglob("*.py"):
            text = path.read_text()
            for item in forbidden:
                if item in text:
                    offenders.append(f"{path}:{item}")
        self.assertEqual(offenders, [])

    def test_train_time_eval_uses_deterministic_preprocessing(self):
        hf_text = pathlib.Path("src/obj_det/models/adapters/hf_trainer.py").read_text()
        tv_text = pathlib.Path("src/obj_det/models/adapters/torchvision.py").read_text()

        self.assertIn("eval_transform = build_detection_transform(train_cfg.preprocess)", hf_text)
        self.assertIn("val_data = HFTrainerDetectionDataset(val_source, eval_transform)", hf_text)
        self.assertIn("eval_transform = build_detection_transform(train_cfg.preprocess)", tv_text)
        self.assertIn("eval_dataset=_TorchvisionTrainerDataset(val_source, eval_transform, spec)", tv_text)

    def test_all_training_adapters_enable_deterministic_execution(self):
        for name in ["hf_trainer.py", "torchvision.py", "ultralytics.py"]:
            with self.subTest(adapter=name):
                text = (pathlib.Path("src/obj_det/models/adapters") / name).read_text()
                self.assertIn("set_seed(train_cfg.seed, deterministic=True)", text)

    def test_ultralytics_train_time_eval_is_warn_only(self):
        text = pathlib.Path("src/obj_det/models/adapters/ultralytics.py").read_text()

        self.assertIn("warnings.warn", text)
        self.assertNotIn("raise NotImplementedError(\"Ultralytics train-time eval_strategy", text)

    def test_all_epoch_callbacks_log_validation_step_and_epoch_index(self):
        for name in ["hf_trainer.py", "torchvision.py", "ultralytics.py"]:
            with self.subTest(adapter=name):
                text = (pathlib.Path("src/obj_det/models/adapters") / name).read_text()
                self.assertIn("log_step=", text)
                self.assertIn('"val/epoch_index": epoch', text)

    def test_predict_paths_do_not_eagerly_list_dataset(self):
        offenders = []
        for path in pathlib.Path("src/obj_det/models/adapters").glob("*.py"):
            text = path.read_text()
            if "list(ds)" in text:
                offenders.append(str(path))

        self.assertEqual(offenders, [])

    def test_training_adapters_do_not_enable_parser_stats(self):
        offenders = []
        for path in pathlib.Path("src/obj_det/models/adapters").glob("*.py"):
            text = path.read_text()
            if "track_stats=True" in text:
                offenders.append(str(path))

        self.assertEqual(offenders, [])

    def test_only_base_adapter_owns_final_evaluate(self):
        evaluate_defs = []
        for path in pathlib.Path("src/obj_det/models/adapters").glob("*.py"):
            if "def evaluate(" in path.read_text():
                evaluate_defs.append(path.name)

        self.assertEqual(evaluate_defs, ["base.py"])

    def test_base_evaluate_uses_predictions_and_shared_evaluator(self):
        text = pathlib.Path("src/obj_det/models/adapters/base.py").read_text()

        self.assertIn("DetectionEvaluator", text)
        self.assertIn("self.predict(ds, artifact, predict_cfg)", text)
        self.assertIn("evaluator.evaluate(ds, predictions, eval_cfg", text)


if __name__ == "__main__":
    unittest.main()
