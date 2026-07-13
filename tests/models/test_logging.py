from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from obj_det.models.logging import (
    CompositeLogger,
    LocalJsonLogger,
    WandbLogger,
    flatten_eval_result,
    flatten_prefixed_scalar_mapping,
    flatten_scalar_mapping,
)
from obj_det.models.logging.base import BaseExperimentLogger
from obj_det.models.schemas.result import EvalResult


class RecordingLogger(BaseExperimentLogger):
    def __init__(self):
        self.events = []

    def start_run(self, name, config):
        self.events.append(("start_run", name, config))

    def finish_run(self, state="finished", error=None):
        self.events.append(("finish_run", state, error))

    def log_metrics(self, metrics, step=None):
        self.events.append(("metrics", metrics, step))


class LoggingTest(unittest.TestCase):
    def test_flatten_eval_result_logs_all_scalar_groups(self):
        result = EvalResult(
            model_key="m",
            dataset_key="tiny",
            split="test",
            primary_metric="map_50_95",
            primary_metric_value=0.5,
            metrics={"map_50_95": 0.5, "map_50": 0.7},
            per_class={"big car": {"map_50": 0.8}},
            per_condition={"low light": {"map_50": 0.6}},
            per_domain={"road": {"map_50": 0.4}},
            per_size={"small": {"ap": 0.3}},
            num_images=3,
            num_ground_truth_objects=4,
            num_predictions=5,
        )

        metrics = flatten_eval_result(result, prefix="eval/test")

        self.assertEqual(metrics["eval/test/map_50_95"], 0.5)
        self.assertEqual(metrics["eval/test/primary/map_50_95"], 0.5)
        self.assertEqual(metrics["eval/test/per_class/big_car/map_50"], 0.8)
        self.assertEqual(metrics["eval/test/per_condition/low_light/map_50"], 0.6)
        self.assertEqual(metrics["eval/test/per_domain/road/map_50"], 0.4)
        self.assertEqual(metrics["eval/test/per_size/small/ap"], 0.3)
        self.assertEqual(metrics["eval/test/num_images"], 3)

    def test_flatten_scalar_mapping_keeps_only_scalars(self):
        metrics, step = flatten_scalar_mapping("train", {"step": 2, "loss": "1.5", "name": "x", "ok": True})

        self.assertEqual(step, 2)
        self.assertEqual(metrics, {"train/loss": 1.5, "train/ok": True})

    def test_flatten_prefixed_scalar_mapping_does_not_duplicate_prefix(self):
        metrics, step = flatten_prefixed_scalar_mapping(
            "train",
            {"epoch": 3, "train/box_loss": "1.5", "lr/pg0": 0.01, "name": "x"},
        )

        self.assertEqual(step, 3)
        self.assertEqual(metrics["train/box_loss"], 1.5)
        self.assertEqual(metrics["train/lr/pg0"], 0.01)
        self.assertNotIn("train/train_box_loss", metrics)

    def test_local_json_logger_writes_events(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            logger = LocalJsonLogger(path)
            logger.start_run("run", {"a": 1})
            logger.log_metrics({"train/loss": 1.2}, step=4)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(rows[0]["event"], "start_run")
        self.assertEqual(rows[1]["metrics"], {"train/loss": 1.2})
        self.assertEqual(rows[1]["step"], 4)

    def test_composite_forwards_events(self):
        first = RecordingLogger()
        second = RecordingLogger()
        logger = CompositeLogger([first, second])

        logger.start_run("r", {})
        logger.log_metrics({"x": 1}, step=2)
        logger.finish_run()

        self.assertEqual(first.events, second.events)
        self.assertEqual(first.events[1], ("metrics", {"x": 1}, 2))

    def test_wandb_epoch_eval_defers_commit_for_same_step_state(self):
        calls = []
        wandb = type("FakeWandb", (), {"log": lambda self, data, **kwargs: calls.append((data, kwargs))})()
        logger = WandbLogger(project="test")
        logger._wandb = lambda: wandb
        result = EvalResult(
            model_key="m",
            primary_metric="map_50_95",
            primary_metric_value=0.5,
            metrics={"map_50_95": 0.5},
        )

        logger.log_eval_result(result, step=17, prefix="val/epoch")
        logger.log_metrics({"val/epoch_index": 2}, step=17)

        self.assertEqual(calls[0][1], {"step": 17, "commit": False})
        self.assertEqual(calls[1][1], {"step": 17})


if __name__ == "__main__":
    unittest.main()
