from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from datasets import Dataset

from obj_det.models.adapters.base import BaseModelAdapter
from obj_det.models.schemas import EvalConfig, ModelConfig, SearchSpace, TrainConfig, TransformConfig, TuningConfig
from obj_det.models.schemas.artifact import ModelArtifact
from obj_det.models.schemas.config import PredictConfig
from obj_det.models.schemas.prediction import PredictionRecord
from obj_det.models.schemas.result import EvalResult
from obj_det.models.tuning.final import run_final_seeds
from obj_det.models.tuning.runner import TuningRunner

from .helpers import row


class DummyAdapter(BaseModelAdapter):
    def __init__(self):
        super().__init__(ModelConfig(key="dummy", backend="torchvision", model_name_or_path="x"))
        self.trained = []
        self.evaluated = []

    def train(self, train_ds, val_ds, train_cfg):
        self.trained.append((train_cfg.seed, dict(train_cfg.hparams), train_cfg.output_dir))
        if train_cfg.hparams.get("fail"):
            raise RuntimeError("planned failure")
        return ModelArtifact(
            model_key=self.key,
            backend=self.backend,
            run_key=train_cfg.run_key,
            classes=train_cfg.classes,
            label_mode=train_cfg.label_mode,
            artifact_path=train_cfg.output_dir,
        )

    def evaluate(self, ds, artifact, eval_cfg):
        self.evaluated.append(ds)
        value = float(self.trained[-1][1].get("score", 0.0))
        return EvalResult(
            model_key=self.key,
            primary_metric="map_50_95",
            primary_metric_value=value,
            metrics={"map_50_95": value},
        )

    def predict(self, ds, artifact, predict_cfg: PredictConfig):
        return iter([
            PredictionRecord(
                image_id="img1",
                dataset="tiny",
                split="val",
                model_key=self.key,
                width=1,
                height=1,
            )
        ])


class FakeTrial:
    def __init__(self, number, values):
        self.number = number
        self.values = values
        self.params = {}

    def suggest_categorical(self, name, choices):
        value = self.values[name]
        self.params[name] = value
        return value

    def suggest_float(self, name, low, high, log=False):
        value = float(self.values[name])
        self.params[name] = value
        return value

    def suggest_int(self, name, low, high, log=False):
        value = int(self.values[name])
        self.params[name] = value
        return value

    def report(self, value, step):
        self.reported = (value, step)


class FakeStudy:
    def __init__(self, values):
        self.values = values
        self.trials = []
        self._best_trial = None
        self.best_value = None

    def optimize(self, objective, n_trials, timeout=None, catch=()):
        for i in range(n_trials):
            trial = FakeTrial(i, self.values[i])
            self.trials.append(trial)
            try:
                value = objective(trial)
            except catch:
                continue
            if self.best_value is None or value > self.best_value:
                self.best_value = value
                self._best_trial = trial

    @property
    def best_trial(self):
        if self._best_trial is None:
            raise ValueError("no completed trials")
        return self._best_trial


class FakeOptuna(types.ModuleType):
    def __init__(self, values):
        super().__init__("optuna")
        self.values = values
        self.samplers = types.SimpleNamespace(
            RandomSampler=lambda seed=None: object(),
            TPESampler=lambda seed=None: object(),
        )
        self.pruners = types.SimpleNamespace(
            NopPruner=lambda: object(),
            MedianPruner=lambda: object(),
            SuccessiveHalvingPruner=lambda: object(),
        )
        self.study = None

    def create_study(self, **kwargs):
        self.study = FakeStudy(self.values)
        return self.study


class TuningTest(unittest.TestCase):
    def setUp(self):
        self.old_optuna = sys.modules.get("optuna")

    def tearDown(self):
        if self.old_optuna is None:
            sys.modules.pop("optuna", None)
        else:
            sys.modules["optuna"] = self.old_optuna

    def test_hpo_records_failed_trial_and_selects_best(self):
        sys.modules["optuna"] = FakeOptuna([
            {"score": 0.1, "fail": True},
            {"score": 0.3, "fail": False},
            {"score": 0.2, "fail": False},
        ])
        ds = Dataset.from_list([row()])
        adapter = DummyAdapter()
        with TemporaryDirectory() as tmp:
            transform = TransformConfig(image_size=32)
            result = TuningRunner().optimize(
                adapter=adapter,
                train_ds=ds,
                val_ds=ds,
                base_train_cfg=TrainConfig(run_key="r", classes=["car"], output_dir=Path(tmp) / "base", transform=transform),
                eval_cfg=EvalConfig(classes=["car"], transform=transform),
                search_space=SearchSpace(params={
                    "score": {"type": "float", "low": 0.0, "high": 1.0},
                    "fail": {"type": "categorical", "choices": [True, False]},
                }),
                tuning_cfg=TuningConfig(study_name="s", n_trials=3, output_dir=Path(tmp)),
            )

        self.assertEqual(len(result.trials), 3)
        self.assertEqual(result.trials[0].state, "failed")
        self.assertIsNotNone(result.best_trial)
        self.assertEqual(result.best_trial.trial_number, 1)
        self.assertEqual(result.best_trial.metric_value, 0.3)

    def test_final_seeds_runs_all_seeds_without_picking_best(self):
        ds = Dataset.from_list([row()])
        adapter = DummyAdapter()
        with TemporaryDirectory() as tmp:
            transform = TransformConfig(image_size=32)
            runs = run_final_seeds(
                adapter=adapter,
                train_ds=ds,
                val_ds=ds,
                test_ds=ds,
                base_train_cfg=TrainConfig(run_key="final", classes=["car"], output_dir=Path(tmp), transform=transform),
                eval_cfg=EvalConfig(classes=["car"], transform=transform),
                hparams={"score": 0.4},
                seeds=[0, 1, 2],
            )

        self.assertEqual([run.seed for run in runs], [0, 1, 2])
        self.assertEqual([item[0] for item in adapter.trained], [0, 1, 2])
        self.assertEqual(len(adapter.evaluated), 6)


if __name__ == "__main__":
    unittest.main()
