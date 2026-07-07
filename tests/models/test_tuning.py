from __future__ import annotations

import sys
import types
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from datasets import Dataset

from obj_det.models.adapters.base import BaseModelAdapter
from obj_det.models.logging.factory import child_logger_factory_from_config
from obj_det.models.schemas import EvalConfig, ModelConfig, PreprocessConfig, SearchSpace, TrainConfig, TuningConfig
from obj_det.models.schemas.artifact import ModelArtifact
from obj_det.models.schemas.config import PredictConfig
from obj_det.models.schemas.logging import LoggingConfig
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

    def train(self, train_ds, val_ds, train_cfg, *, logger=None, log_prefix="train"):
        self.trained.append((train_cfg.seed, dict(train_cfg.hparams), train_cfg.output_dir))
        if logger is not None:
            logger.log_metrics({f"{log_prefix}/dummy_loss": 1.0})
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

    def evaluate(self, ds, artifact, eval_cfg, *, logger=None, log_prefix=None):
        self.evaluated.append(ds)
        value = float(self.trained[-1][1].get("score", 0.0))
        result = EvalResult(
            model_key=self.key,
            primary_metric="map_50_95",
            primary_metric_value=value,
            metrics={"map_50_95": value},
        )
        if logger is not None:
            logger.log_eval_result(result, prefix=log_prefix)
        return result

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


class RecordingLogger:
    def __init__(self, default_log_path: Path | None = None):
        self.default_log_path = default_log_path
        self.run_name = None
        self.events = []

    def start_run(self, name, config):
        self.run_name = name
        self.events.append(("start_run", name, config))

    def finish_run(self, state="finished", error=None):
        self.events.append(("finish_run", state, error))

    def start_trial(self, trial_number, hparams):
        self.events.append(("start_trial", trial_number, hparams))

    def finish_trial(self, state, error=None):
        self.events.append(("finish_trial", state, error))

    def log_metrics(self, metrics, step=None):
        self.events.append(("metrics", metrics, step))

    def log_eval_result(self, result, step=None, prefix=None):
        self.events.append(("eval_result", prefix, result.primary_metric_value))

    def log_artifact(self, path, name=None):
        self.events.append(("artifact", name, path))


class RecordingLoggerFactory:
    def __init__(self):
        self.loggers = []

    def __call__(self, run_name, default_log_path):
        logger = RecordingLogger(default_log_path=Path(default_log_path))
        self.loggers.append(logger)
        return logger


class TuningTest(unittest.TestCase):
    def setUp(self):
        self.old_optuna = sys.modules.get("optuna")

    def tearDown(self):
        if self.old_optuna is None:
            sys.modules.pop("optuna", None)
        else:
            sys.modules["optuna"] = self.old_optuna

    def test_search_space_validation_rejects_bad_specs(self):
        bad_specs = [
            {"lr": {"type": "FLOAT", "low": 0.0, "high": 1.0}},
            {"lr": {"type": "floot", "low": 0.0, "high": 1.0}},
            {"lr": {"type": "float", "low": 1.0, "high": 0.0}},
            {"steps": {"type": "int", "low": 10, "high": 1}},
            {"lr": {"type": "float", "low": 0.0}},
            {"opt": {"type": "categorical", "choices": []}},
            {"opt": {"type": "categorical", "choices": ["a"], "log": True}},
        ]

        for params in bad_specs:
            with self.subTest(params=params), self.assertRaises(ValueError):
                SearchSpace(params=params)

    def test_hpo_records_failed_trial_and_selects_best(self):
        sys.modules["optuna"] = FakeOptuna([
            {"score": 0.1, "fail": True},
            {"score": 0.3, "fail": False},
            {"score": 0.2, "fail": False},
        ])
        ds = Dataset.from_list([row()])
        adapter = DummyAdapter()
        with TemporaryDirectory() as tmp:
            preprocess = PreprocessConfig(image_size=32)
            result = TuningRunner().optimize(
                adapter=adapter,
                train_ds=ds,
                val_ds=ds,
                base_train_cfg=TrainConfig(run_key="r", classes=["car"], output_dir=Path(tmp) / "base", preprocess=preprocess),
                eval_cfg=EvalConfig(classes=["car"], preprocess=preprocess),
                search_space=SearchSpace(params={
                    "score": {"type": "float", "low": 0.0, "high": 1.0},
                    "fail": {"type": "categorical", "choices": [True, False]},
                }),
                tuning_cfg=TuningConfig(study_name="s", n_trials=3, output_dir=Path(tmp), catch_trial_errors=True),
            )

        self.assertEqual(len(result.trials), 3)
        self.assertEqual(result.trials[0].state, "failed")
        self.assertIsNotNone(result.best_trial)
        self.assertEqual(result.best_trial.trial_number, 1)
        self.assertEqual(result.best_trial.metric_value, 0.3)

    def test_hpo_fails_fast_by_default(self):
        sys.modules["optuna"] = FakeOptuna([
            {"score": 0.1, "fail": True},
            {"score": 0.3, "fail": False},
        ])
        ds = Dataset.from_list([row()])
        adapter = DummyAdapter()
        with TemporaryDirectory() as tmp:
            preprocess = PreprocessConfig(image_size=32)
            with self.assertRaises(RuntimeError):
                TuningRunner().optimize(
                    adapter=adapter,
                    train_ds=ds,
                    val_ds=ds,
                    base_train_cfg=TrainConfig(run_key="r", classes=["car"], output_dir=Path(tmp) / "base", preprocess=preprocess),
                    eval_cfg=EvalConfig(classes=["car"], preprocess=preprocess),
                    search_space=SearchSpace(params={
                        "score": {"type": "float", "low": 0.0, "high": 1.0},
                        "fail": {"type": "categorical", "choices": [True, False]},
                    }),
                    tuning_cfg=TuningConfig(study_name="s", n_trials=2, output_dir=Path(tmp)),
                )

        self.assertEqual(len(adapter.trained), 1)

    def test_hpo_stores_merged_hparams_for_trials_and_best(self):
        sys.modules["optuna"] = FakeOptuna([
            {"score": 0.3, "fail": False},
        ])
        ds = Dataset.from_list([row()])
        adapter = DummyAdapter()
        with TemporaryDirectory() as tmp:
            preprocess = PreprocessConfig(image_size=32)
            result = TuningRunner().optimize(
                adapter=adapter,
                train_ds=ds,
                val_ds=ds,
                base_train_cfg=TrainConfig(
                    run_key="r",
                    classes=["car"],
                    output_dir=Path(tmp) / "base",
                    preprocess=preprocess,
                    hparams={"warmup_epochs": 3},
                ),
                eval_cfg=EvalConfig(classes=["car"], preprocess=preprocess),
                search_space=SearchSpace(params={
                    "score": {"type": "float", "low": 0.0, "high": 1.0},
                    "fail": {"type": "categorical", "choices": [True, False]},
                }),
                tuning_cfg=TuningConfig(study_name="s", n_trials=1, output_dir=Path(tmp)),
            )

        expected = {"warmup_epochs": 3, "score": 0.3, "fail": False}
        self.assertEqual(adapter.trained[0][1], expected)
        self.assertEqual(result.trials[0].hparams, expected)
        self.assertEqual(result.best_trial.hparams, expected)

    def test_hpo_uses_lightweight_eval_by_default(self):
        class EvalConfigRecordingAdapter(DummyAdapter):
            def __init__(self):
                super().__init__()
                self.eval_configs = []

            def evaluate(self, ds, artifact, eval_cfg, *, logger=None, log_prefix=None):
                self.eval_configs.append(eval_cfg)
                return super().evaluate(ds, artifact, eval_cfg, logger=logger, log_prefix=log_prefix)

        sys.modules["optuna"] = FakeOptuna([
            {"score": 0.3, "fail": False},
        ])
        ds = Dataset.from_list([row()])
        adapter = EvalConfigRecordingAdapter()
        with TemporaryDirectory() as tmp:
            preprocess = PreprocessConfig(image_size=32)
            TuningRunner().optimize(
                adapter=adapter,
                train_ds=ds,
                val_ds=ds,
                base_train_cfg=TrainConfig(run_key="r", classes=["car"], output_dir=Path(tmp) / "base", preprocess=preprocess),
                eval_cfg=EvalConfig(
                    classes=["car"],
                    preprocess=preprocess,
                    compute_per_class=True,
                    compute_per_condition=True,
                    compute_per_domain=True,
                    compute_per_size=True,
                ),
                search_space=SearchSpace(params={
                    "score": {"type": "float", "low": 0.0, "high": 1.0},
                    "fail": {"type": "categorical", "choices": [True, False]},
                }),
                tuning_cfg=TuningConfig(study_name="s", n_trials=1, output_dir=Path(tmp)),
            )

        self.assertEqual(len(adapter.eval_configs), 1)
        used_cfg = adapter.eval_configs[0]
        self.assertFalse(used_cfg.compute_per_class)
        self.assertFalse(used_cfg.compute_per_condition)
        self.assertFalse(used_cfg.compute_per_domain)
        self.assertFalse(used_cfg.compute_per_size)

    def test_hpo_can_opt_into_detailed_eval(self):
        class EvalConfigRecordingAdapter(DummyAdapter):
            def __init__(self):
                super().__init__()
                self.eval_configs = []

            def evaluate(self, ds, artifact, eval_cfg, *, logger=None, log_prefix=None):
                self.eval_configs.append(eval_cfg)
                return super().evaluate(ds, artifact, eval_cfg, logger=logger, log_prefix=log_prefix)

        sys.modules["optuna"] = FakeOptuna([
            {"score": 0.3, "fail": False},
        ])
        ds = Dataset.from_list([row()])
        adapter = EvalConfigRecordingAdapter()
        with TemporaryDirectory() as tmp:
            preprocess = PreprocessConfig(image_size=32)
            TuningRunner().optimize(
                adapter=adapter,
                train_ds=ds,
                val_ds=ds,
                base_train_cfg=TrainConfig(run_key="r", classes=["car"], output_dir=Path(tmp) / "base", preprocess=preprocess),
                eval_cfg=EvalConfig(classes=["car"], preprocess=preprocess, compute_per_class=True),
                search_space=SearchSpace(params={
                    "score": {"type": "float", "low": 0.0, "high": 1.0},
                    "fail": {"type": "categorical", "choices": [True, False]},
                }),
                tuning_cfg=TuningConfig(study_name="s", n_trials=1, output_dir=Path(tmp), detailed_eval=True),
            )

        self.assertTrue(adapter.eval_configs[0].compute_per_class)

    def test_hpo_logs_trial_train_eval_and_objective(self):
        sys.modules["optuna"] = FakeOptuna([
            {"score": 0.3, "fail": False},
        ])
        ds = Dataset.from_list([row()])
        adapter = DummyAdapter()
        logger_factory = RecordingLoggerFactory()
        with TemporaryDirectory() as tmp:
            preprocess = PreprocessConfig(image_size=32)
            TuningRunner(logger_factory=logger_factory).optimize(
                adapter=adapter,
                train_ds=ds,
                val_ds=ds,
                base_train_cfg=TrainConfig(run_key="r", classes=["car"], output_dir=Path(tmp) / "base", preprocess=preprocess),
                eval_cfg=EvalConfig(classes=["car"], preprocess=preprocess),
                search_space=SearchSpace(params={
                    "score": {"type": "float", "low": 0.0, "high": 1.0},
                    "fail": {"type": "categorical", "choices": [True, False]},
                }),
                tuning_cfg=TuningConfig(study_name="s", n_trials=1, output_dir=Path(tmp)),
                run_config={"experiment": "cfg"},
            )

        self.assertEqual(len(logger_factory.loggers), 1)
        logger = logger_factory.loggers[0]
        self.assertEqual(logger.run_name, "s_trial_0000")
        self.assertEqual(logger.default_log_path, Path(tmp) / "trial_0000" / "logs" / "events.jsonl")
        self.assertEqual(logger.events[0][0], "start_run")
        self.assertEqual(logger.events[0][2]["trial"]["number"], 0)
        self.assertEqual(logger.events[0][2]["trial"]["hparams"]["score"], 0.3)
        metric_events = [event for event in logger.events if event[0] == "metrics"]
        self.assertTrue(any("train/dummy_loss" in event[1] for event in metric_events))
        self.assertTrue(any("objective/map_50_95" in event[1] for event in metric_events))
        self.assertTrue(any(event[:2] == ("eval_result", "val") for event in logger.events))
        self.assertEqual(logger.events[-1][:2], ("finish_run", "finished"))

    def test_hpo_failed_trial_gets_own_failed_run(self):
        sys.modules["optuna"] = FakeOptuna([
            {"score": 0.1, "fail": True},
            {"score": 0.3, "fail": False},
        ])
        ds = Dataset.from_list([row()])
        adapter = DummyAdapter()
        logger_factory = RecordingLoggerFactory()
        with TemporaryDirectory() as tmp:
            preprocess = PreprocessConfig(image_size=32)
            TuningRunner(logger_factory=logger_factory).optimize(
                adapter=adapter,
                train_ds=ds,
                val_ds=ds,
                base_train_cfg=TrainConfig(run_key="r", classes=["car"], output_dir=Path(tmp) / "base", preprocess=preprocess),
                eval_cfg=EvalConfig(classes=["car"], preprocess=preprocess),
                search_space=SearchSpace(params={
                    "score": {"type": "float", "low": 0.0, "high": 1.0},
                    "fail": {"type": "categorical", "choices": [True, False]},
                }),
                tuning_cfg=TuningConfig(study_name="s", n_trials=2, output_dir=Path(tmp), catch_trial_errors=True),
            )

        self.assertEqual([logger.run_name for logger in logger_factory.loggers], ["s_trial_0000", "s_trial_0001"])
        self.assertEqual(logger_factory.loggers[0].events[-1][0], "finish_run")
        self.assertEqual(logger_factory.loggers[0].events[-1][1], "failed")
        self.assertEqual(logger_factory.loggers[1].events[-1][1], "finished")

    def test_hpo_writes_local_child_log_file(self):
        sys.modules["optuna"] = FakeOptuna([
            {"score": 0.3, "fail": False},
        ])
        ds = Dataset.from_list([row()])
        adapter = DummyAdapter()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            preprocess = PreprocessConfig(image_size=32)
            factory = child_logger_factory_from_config(LoggingConfig(backends=["local"]), wandb_group="s")
            TuningRunner(logger_factory=factory).optimize(
                adapter=adapter,
                train_ds=ds,
                val_ds=ds,
                base_train_cfg=TrainConfig(run_key="r", classes=["car"], output_dir=root / "base", preprocess=preprocess),
                eval_cfg=EvalConfig(classes=["car"], preprocess=preprocess),
                search_space=SearchSpace(params={
                    "score": {"type": "float", "low": 0.0, "high": 1.0},
                    "fail": {"type": "categorical", "choices": [True, False]},
                }),
                tuning_cfg=TuningConfig(study_name="s", n_trials=1, output_dir=root / "hpo"),
            )
            log_path = root / "hpo" / "trial_0000" / "logs" / "events.jsonl"
            rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(rows[0]["event"], "start_run")
        self.assertTrue(any(row["event"] == "metrics" and "run/started" in row["metrics"] for row in rows))
        self.assertTrue(any(row["event"] == "metrics" and "train/dummy_loss" in row["metrics"] for row in rows))
        self.assertTrue(any(row["event"] == "metrics" and "objective/map_50_95" in row["metrics"] for row in rows))
        self.assertEqual(rows[-1]["event"], "finish_run")

    def test_final_seeds_runs_all_seeds_without_picking_best(self):
        ds = Dataset.from_list([row()])
        adapter = DummyAdapter()
        with TemporaryDirectory() as tmp:
            preprocess = PreprocessConfig(image_size=32)
            runs = run_final_seeds(
                adapter=adapter,
                train_ds=ds,
                val_ds=ds,
                test_ds=ds,
                base_train_cfg=TrainConfig(run_key="final", classes=["car"], output_dir=Path(tmp), preprocess=preprocess),
                eval_cfg=EvalConfig(classes=["car"], preprocess=preprocess),
                hparams={"score": 0.4},
                seeds=[0, 1, 2],
            )

        self.assertEqual([run.seed for run in runs], [0, 1, 2])
        self.assertEqual([item[0] for item in adapter.trained], [0, 1, 2])
        self.assertEqual(len(adapter.evaluated), 6)

    def test_final_seeds_merge_base_and_best_hparams(self):
        ds = Dataset.from_list([row()])
        adapter = DummyAdapter()
        with TemporaryDirectory() as tmp:
            preprocess = PreprocessConfig(image_size=32)
            run_final_seeds(
                adapter=adapter,
                train_ds=ds,
                val_ds=ds,
                test_ds=ds,
                base_train_cfg=TrainConfig(
                    run_key="final",
                    classes=["car"],
                    output_dir=Path(tmp),
                    preprocess=preprocess,
                    hparams={"warmup_epochs": 3},
                ),
                eval_cfg=EvalConfig(classes=["car"], preprocess=preprocess),
                hparams={"score": 0.4},
                seeds=[0],
            )

        self.assertEqual(adapter.trained[0][1], {"warmup_epochs": 3, "score": 0.4})

    def test_final_seeds_create_one_logger_run_per_seed(self):
        ds = Dataset.from_list([row()])
        adapter = DummyAdapter()
        logger_factory = RecordingLoggerFactory()
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "final"
            preprocess = PreprocessConfig(image_size=32)
            run_final_seeds(
                adapter=adapter,
                train_ds=ds,
                val_ds=ds,
                test_ds=ds,
                base_train_cfg=TrainConfig(run_key="final", classes=["car"], output_dir=Path(tmp), preprocess=preprocess),
                eval_cfg=EvalConfig(classes=["car"], preprocess=preprocess),
                hparams={"score": 0.4},
                seeds=[0, 1],
                output_dir=output_dir,
                logger_factory=logger_factory,
                run_config={"experiment": "cfg"},
            )

        self.assertEqual([logger.run_name for logger in logger_factory.loggers], ["final_final_seed0", "final_final_seed1"])
        self.assertEqual(
            [logger.default_log_path for logger in logger_factory.loggers],
            [
                output_dir / "final_seed0" / "logs" / "events.jsonl",
                output_dir / "final_seed1" / "logs" / "events.jsonl",
            ],
        )
        first_events = logger_factory.loggers[0].events
        self.assertTrue(any(event[0] == "metrics" and "train/dummy_loss" in event[1] for event in first_events))
        self.assertTrue(any(event[:2] == ("eval_result", "val") for event in first_events))
        self.assertTrue(any(event[:2] == ("eval_result", "test") for event in first_events))
        self.assertEqual(first_events[-1][:2], ("finish_run", "finished"))


if __name__ == "__main__":
    unittest.main()
