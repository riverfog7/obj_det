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
from obj_det.models.runner import write_final_results
from obj_det.models.tuning.final import FinalSeedRun, run_final_seeds
from obj_det.models.tuning.runner import TuningRunner

from .helpers import row


class DummyAdapter(BaseModelAdapter):
    def __init__(self, *, fail_train_calls: set[int] | None = None):
        super().__init__(
            ModelConfig(
                key="dummy",
                backend="torchvision",
                model_name_or_path="x",
                preprocess=PreprocessConfig(resize_mode="letterbox", height=32, width=32),
            )
        )
        self.fail_train_calls = set(fail_train_calls or ())
        self.trained = []
        self.train_configs = []
        self.epoch_eval_configs = []
        self.evaluated = []

    def train(self, train_ds, val_ds, train_cfg, *, epoch_eval_cfg=None, logger=None, log_prefix="train"):
        self.trained.append((train_cfg.seed, dict(train_cfg.hparams), train_cfg.output_dir))
        self.train_configs.append(train_cfg)
        self.epoch_eval_configs.append(epoch_eval_cfg)
        if logger is not None:
            logger.log_metrics({f"{log_prefix}/dummy_loss": 1.0})
        if len(self.trained) - 1 in self.fail_train_calls:
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
        value = float(self.trained[-1][1].get("learning_rate", 0.0))
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
        self.sampler_calls = []
        self.create_study_kwargs = None

        def random_sampler(seed=None):
            self.sampler_calls.append(("random", seed, {}))
            return object()

        def tpe_sampler(seed=None, **kwargs):
            self.sampler_calls.append(("tpe", seed, kwargs))
            return object()

        self.samplers = types.SimpleNamespace(
            RandomSampler=random_sampler,
            TPESampler=tpe_sampler,
        )
        self.pruners = types.SimpleNamespace(
            NopPruner=lambda: object(),
            MedianPruner=lambda: object(),
            SuccessiveHalvingPruner=lambda: object(),
        )
        self.study = None

    def create_study(self, **kwargs):
        self.create_study_kwargs = kwargs
        self.study = FakeStudy(self.values)
        return self.study


def learning_rate_search_space() -> SearchSpace:
    return SearchSpace(
        params={
            "learning_rate": {
                "type": "float",
                "low": 1.0e-6,
                "high": 1.0,
                "log": True,
            }
        }
    )


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
            {"learning_rate": 0.1},
            {"learning_rate": 0.3},
            {"learning_rate": 0.2},
        ])
        ds = Dataset.from_list([row()])
        adapter = DummyAdapter(fail_train_calls={0})
        with TemporaryDirectory() as tmp:
            preprocess = PreprocessConfig(resize_mode="letterbox", height=32, width=32)
            result = TuningRunner().optimize(
                adapter=adapter,
                train_ds=ds,
                val_ds=ds,
                base_train_cfg=TrainConfig(run_key="r", classes=["car"], output_dir=Path(tmp) / "base", preprocess=preprocess),
                eval_cfg=EvalConfig(classes=["car"], preprocess=preprocess),
                search_space=learning_rate_search_space(),
                tuning_cfg=TuningConfig(study_name="s", n_trials=3, output_dir=Path(tmp), catch_trial_errors=True),
            )

        self.assertEqual(len(result.trials), 3)
        self.assertEqual(result.trials[0].state, "failed")
        self.assertIsNotNone(result.best_trial)
        self.assertEqual(result.best_trial.trial_number, 1)
        self.assertEqual(result.best_trial.metric_value, 0.3)

    def test_hpo_fails_fast_by_default(self):
        sys.modules["optuna"] = FakeOptuna([
            {"learning_rate": 0.1},
            {"learning_rate": 0.3},
        ])
        ds = Dataset.from_list([row()])
        adapter = DummyAdapter(fail_train_calls={0})
        with TemporaryDirectory() as tmp:
            preprocess = PreprocessConfig(resize_mode="letterbox", height=32, width=32)
            with self.assertRaises(RuntimeError):
                TuningRunner().optimize(
                    adapter=adapter,
                    train_ds=ds,
                    val_ds=ds,
                    base_train_cfg=TrainConfig(run_key="r", classes=["car"], output_dir=Path(tmp) / "base", preprocess=preprocess),
                    eval_cfg=EvalConfig(classes=["car"], preprocess=preprocess),
                    search_space=learning_rate_search_space(),
                    tuning_cfg=TuningConfig(study_name="s", n_trials=2, output_dir=Path(tmp)),
                )

        self.assertEqual(len(adapter.trained), 1)

    def test_hpo_stores_sampled_hparams_and_resolved_training_config(self):
        sys.modules["optuna"] = FakeOptuna([
            {"learning_rate": 0.3},
        ])
        ds = Dataset.from_list([row()])
        adapter = DummyAdapter()
        with TemporaryDirectory() as tmp:
            preprocess = PreprocessConfig(resize_mode="letterbox", height=32, width=32)
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
                search_space=learning_rate_search_space(),
                tuning_cfg=TuningConfig(study_name="s", n_trials=1, output_dir=Path(tmp)),
            )

        expected = {"warmup_epochs": 3, "learning_rate": 0.3}
        self.assertEqual(adapter.trained[0][1], expected)
        sampled = {"learning_rate": 0.3}
        self.assertEqual(result.trials[0].hparams, sampled)
        self.assertEqual(result.best_trial.hparams, sampled)
        self.assertEqual(result.best_trial.resolved_train_config["hparams"], expected)

    def test_controlled_hpo_runs_eight_lr_only_trials_with_fixed_protocol(self):
        class CheckpointingAdapter(DummyAdapter):
            def train(self, train_ds, val_ds, train_cfg, *, epoch_eval_cfg=None, logger=None, log_prefix="train"):
                artifact = super().train(
                    train_ds,
                    val_ds,
                    train_cfg,
                    epoch_eval_cfg=epoch_eval_cfg,
                    logger=logger,
                    log_prefix=log_prefix,
                )
                checkpoint = train_cfg.output_dir / "checkpoints" / "epoch_010.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_text("checkpoint", encoding="utf-8")
                return artifact.model_copy(update={"checkpoint_path": checkpoint})

        learning_rates = [3.0e-6 * (1000.0 ** (index / 7.0)) for index in range(8)]
        fake_optuna = FakeOptuna([{"learning_rate": value} for value in learning_rates])
        sys.modules["optuna"] = fake_optuna
        ds = Dataset.from_list([row()])
        adapter = CheckpointingAdapter()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            preprocess = PreprocessConfig(resize_mode="letterbox", height=32, width=32)
            result = TuningRunner().optimize(
                adapter=adapter,
                train_ds=ds,
                val_ds=ds,
                base_train_cfg=TrainConfig(
                    run_key="r",
                    classes=["car"],
                    output_dir=root / "base",
                    preprocess=preprocess,
                    protocol="controlled",
                ),
                eval_cfg=EvalConfig(classes=["car"], preprocess=preprocess),
                search_space=SearchSpace(
                    params={
                        "learning_rate": {
                            "type": "float",
                            "low": 3.0e-6,
                            "high": 3.0e-3,
                            "log": True,
                        }
                    }
                ),
                tuning_cfg=TuningConfig(
                    study_name="s",
                    output_dir=root,
                    storage="sqlite:///ignored.db",
                ),
            )

            checkpoints = sorted(root.glob("trial_*/checkpoints/epoch_010.pt"))

        self.assertEqual(len(adapter.train_configs), 8)
        self.assertEqual(len(result.trials), 8)
        self.assertEqual(len(checkpoints), 8)
        self.assertEqual(fake_optuna.sampler_calls, [("tpe", 0, {"n_startup_trials": 3})])
        self.assertNotIn("storage", fake_optuna.create_study_kwargs)
        self.assertNotIn("load_if_exists", fake_optuna.create_study_kwargs)
        for train_cfg, epoch_eval_cfg, trial_result, fake_trial in zip(
            adapter.train_configs,
            adapter.epoch_eval_configs,
            result.trials,
            fake_optuna.study.trials,
        ):
            self.assertEqual(train_cfg.max_epochs, 10)
            self.assertEqual(train_cfg.seed, 0)
            self.assertFalse(train_cfg.eval_strategy.enabled)
            self.assertFalse(train_cfg.early_stopping.enabled)
            self.assertEqual(train_cfg.scheduler.total_epochs, 50)
            self.assertEqual(train_cfg.checkpoint.save_every_epochs, 10)
            self.assertFalse(train_cfg.checkpoint.save_best)
            self.assertTrue(train_cfg.checkpoint.save_last)
            self.assertFalse(train_cfg.checkpoint.keep_all_epoch_checkpoints)
            self.assertEqual(set(train_cfg.hparams), {"learning_rate"})
            self.assertIsNone(epoch_eval_cfg)
            self.assertEqual(set(trial_result.hparams), {"learning_rate"})
            self.assertEqual(trial_result.resolved_train_config["max_epochs"], 10)
            self.assertEqual(trial_result.checkpoint_meta["checkpoint_selection"], "trial_final")
            self.assertEqual(trial_result.checkpoint_meta["checkpoint_epoch"], 10)
            self.assertEqual(trial_result.checkpoint_meta["scheduler_total_epochs"], 50)
            self.assertEqual(fake_trial.reported[1], 10)
        self.assertEqual(set(result.best_trial.hparams), {"learning_rate"})
        self.assertEqual(result.best_trial.checkpoint_meta["checkpoint_selection"], "trial_final")
        self.assertEqual(result.best_trial.meta["boundary_warning"].split()[0], "Best")

    def test_hpo_requires_exact_objective_metric_key(self):
        class MissingObjectiveAdapter(DummyAdapter):
            def evaluate(self, ds, artifact, eval_cfg, *, logger=None, log_prefix=None):
                return EvalResult(
                    model_key=self.key,
                    primary_metric="map_50_95",
                    primary_metric_value=0.9,
                    metrics={"map_50": 0.9},
                )

        sys.modules["optuna"] = FakeOptuna([{"learning_rate": 3.0e-4}])
        ds = Dataset.from_list([row()])
        with TemporaryDirectory() as tmp:
            preprocess = PreprocessConfig(resize_mode="letterbox", height=32, width=32)
            with self.assertRaisesRegex(ValueError, "Missing required HPO objective metric 'map_50_95'"):
                TuningRunner().optimize(
                    adapter=MissingObjectiveAdapter(),
                    train_ds=ds,
                    val_ds=ds,
                    base_train_cfg=TrainConfig(
                        run_key="r",
                        classes=["car"],
                        output_dir=Path(tmp) / "base",
                        preprocess=preprocess,
                    ),
                    eval_cfg=EvalConfig(classes=["car"], preprocess=preprocess),
                    search_space=SearchSpace(
                        params={
                            "learning_rate": {
                                "type": "float",
                                "low": 3.0e-6,
                                "high": 3.0e-3,
                                "log": True,
                            }
                        }
                    ),
                    tuning_cfg=TuningConfig(study_name="s", n_trials=1, output_dir=Path(tmp)),
                )

    def test_lr_boundary_warning_uses_log_space(self):
        search_space = SearchSpace(
            params={
                "learning_rate": {
                    "type": "float",
                    "low": 3.0e-6,
                    "high": 3.0e-3,
                    "log": True,
                }
            }
        )
        runner = TuningRunner()

        self.assertIn("lower log-space boundary", runner._boundary_warning({"learning_rate": 3.0e-6}, search_space))
        self.assertIsNone(runner._boundary_warning({"learning_rate": (3.0e-6 * 3.0e-3) ** 0.5}, search_space))
        self.assertIn("upper log-space boundary", runner._boundary_warning({"learning_rate": 3.0e-3}, search_space))

    def test_hpo_uses_detailed_eval_by_default(self):
        class EvalConfigRecordingAdapter(DummyAdapter):
            def __init__(self):
                super().__init__()
                self.eval_configs = []

            def evaluate(self, ds, artifact, eval_cfg, *, logger=None, log_prefix=None):
                self.eval_configs.append(eval_cfg)
                return super().evaluate(ds, artifact, eval_cfg, logger=logger, log_prefix=log_prefix)

        sys.modules["optuna"] = FakeOptuna([
            {"learning_rate": 0.3},
        ])
        ds = Dataset.from_list([row()])
        adapter = EvalConfigRecordingAdapter()
        with TemporaryDirectory() as tmp:
            preprocess = PreprocessConfig(resize_mode="letterbox", height=32, width=32)
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
                search_space=learning_rate_search_space(),
                tuning_cfg=TuningConfig(study_name="s", n_trials=1, output_dir=Path(tmp)),
            )

        self.assertEqual(len(adapter.eval_configs), 1)
        used_cfg = adapter.eval_configs[0]
        self.assertTrue(used_cfg.compute_per_class)
        self.assertTrue(used_cfg.compute_per_condition)
        self.assertTrue(used_cfg.compute_per_domain)
        self.assertTrue(used_cfg.compute_per_size)

    def test_hpo_can_opt_out_of_detailed_eval(self):
        class EvalConfigRecordingAdapter(DummyAdapter):
            def __init__(self):
                super().__init__()
                self.eval_configs = []

            def evaluate(self, ds, artifact, eval_cfg, *, logger=None, log_prefix=None):
                self.eval_configs.append(eval_cfg)
                return super().evaluate(ds, artifact, eval_cfg, logger=logger, log_prefix=log_prefix)

        sys.modules["optuna"] = FakeOptuna([
            {"learning_rate": 0.3},
        ])
        ds = Dataset.from_list([row()])
        adapter = EvalConfigRecordingAdapter()
        with TemporaryDirectory() as tmp:
            preprocess = PreprocessConfig(resize_mode="letterbox", height=32, width=32)
            TuningRunner().optimize(
                adapter=adapter,
                train_ds=ds,
                val_ds=ds,
                base_train_cfg=TrainConfig(run_key="r", classes=["car"], output_dir=Path(tmp) / "base", preprocess=preprocess),
                eval_cfg=EvalConfig(classes=["car"], preprocess=preprocess, compute_per_class=True),
                search_space=learning_rate_search_space(),
                tuning_cfg=TuningConfig(study_name="s", n_trials=1, output_dir=Path(tmp), detailed_eval=False),
            )

        self.assertFalse(adapter.eval_configs[0].compute_per_class)

    def test_hpo_logs_trial_train_eval_and_objective(self):
        sys.modules["optuna"] = FakeOptuna([
            {"learning_rate": 0.3},
        ])
        ds = Dataset.from_list([row()])
        adapter = DummyAdapter()
        logger_factory = RecordingLoggerFactory()
        with TemporaryDirectory() as tmp:
            preprocess = PreprocessConfig(resize_mode="letterbox", height=32, width=32)
            TuningRunner(logger_factory=logger_factory).optimize(
                adapter=adapter,
                train_ds=ds,
                val_ds=ds,
                base_train_cfg=TrainConfig(run_key="r", classes=["car"], output_dir=Path(tmp) / "base", preprocess=preprocess),
                eval_cfg=EvalConfig(classes=["car"], preprocess=preprocess),
                search_space=learning_rate_search_space(),
                tuning_cfg=TuningConfig(study_name="s", n_trials=1, output_dir=Path(tmp)),
                run_config={"experiment": "cfg"},
            )

        self.assertEqual(len(logger_factory.loggers), 1)
        logger = logger_factory.loggers[0]
        self.assertEqual(logger.run_name, "s_trial_0000")
        self.assertEqual(logger.default_log_path, Path(tmp) / "trial_0000" / "logs" / "events.jsonl")
        self.assertEqual(logger.events[0][0], "start_run")
        self.assertEqual(logger.events[0][2]["trial"]["number"], 0)
        self.assertEqual(logger.events[0][2]["trial"]["hparams"]["learning_rate"], 0.3)
        metric_events = [event for event in logger.events if event[0] == "metrics"]
        self.assertTrue(any("train/dummy_loss" in event[1] for event in metric_events))
        self.assertTrue(any("objective/map_50_95" in event[1] for event in metric_events))
        self.assertTrue(any(event[:2] == ("eval_result", "val") for event in logger.events))
        self.assertEqual(logger.events[-1][:2], ("finish_run", "finished"))

    def test_hpo_failed_trial_gets_own_failed_run(self):
        sys.modules["optuna"] = FakeOptuna([
            {"learning_rate": 0.1},
            {"learning_rate": 0.3},
        ])
        ds = Dataset.from_list([row()])
        adapter = DummyAdapter(fail_train_calls={0})
        logger_factory = RecordingLoggerFactory()
        with TemporaryDirectory() as tmp:
            preprocess = PreprocessConfig(resize_mode="letterbox", height=32, width=32)
            TuningRunner(logger_factory=logger_factory).optimize(
                adapter=adapter,
                train_ds=ds,
                val_ds=ds,
                base_train_cfg=TrainConfig(run_key="r", classes=["car"], output_dir=Path(tmp) / "base", preprocess=preprocess),
                eval_cfg=EvalConfig(classes=["car"], preprocess=preprocess),
                search_space=learning_rate_search_space(),
                tuning_cfg=TuningConfig(study_name="s", n_trials=2, output_dir=Path(tmp), catch_trial_errors=True),
            )

        self.assertEqual([logger.run_name for logger in logger_factory.loggers], ["s_trial_0000", "s_trial_0001"])
        self.assertEqual(logger_factory.loggers[0].events[-1][0], "finish_run")
        self.assertEqual(logger_factory.loggers[0].events[-1][1], "failed")
        self.assertEqual(logger_factory.loggers[1].events[-1][1], "finished")

    def test_hpo_writes_local_child_log_file(self):
        sys.modules["optuna"] = FakeOptuna([
            {"learning_rate": 0.3},
        ])
        ds = Dataset.from_list([row()])
        adapter = DummyAdapter()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            preprocess = PreprocessConfig(resize_mode="letterbox", height=32, width=32)
            factory = child_logger_factory_from_config(LoggingConfig(backends=["local"]), wandb_group="s")
            TuningRunner(logger_factory=factory).optimize(
                adapter=adapter,
                train_ds=ds,
                val_ds=ds,
                base_train_cfg=TrainConfig(run_key="r", classes=["car"], output_dir=root / "base", preprocess=preprocess),
                eval_cfg=EvalConfig(classes=["car"], preprocess=preprocess),
                search_space=learning_rate_search_space(),
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
            preprocess = PreprocessConfig(resize_mode="letterbox", height=32, width=32)
            runs = run_final_seeds(
                adapter=adapter,
                train_ds=ds,
                val_ds=ds,
                test_ds=ds,
                base_train_cfg=TrainConfig(run_key="final", classes=["car"], output_dir=Path(tmp), preprocess=preprocess),
                eval_cfg=EvalConfig(
                    classes=["car"],
                    preprocess=preprocess,
                    compute_per_class=True,
                    compute_per_condition=True,
                    compute_per_domain=True,
                    compute_per_size=True,
                ),
                hparams={"learning_rate": 3.0e-4},
                seeds=[0, 1, 2],
            )

        self.assertEqual([run.seed for run in runs], [0, 1, 2])
        self.assertEqual([item[0] for item in adapter.trained], [0, 1, 2])
        self.assertEqual(len(adapter.evaluated), 6)
        self.assertEqual(len(adapter.epoch_eval_configs), 3)
        for epoch_eval_cfg in adapter.epoch_eval_configs:
            self.assertTrue(epoch_eval_cfg.compute_per_class)
            self.assertTrue(epoch_eval_cfg.compute_per_condition)
            self.assertTrue(epoch_eval_cfg.compute_per_domain)
            self.assertTrue(epoch_eval_cfg.compute_per_size)

    def test_final_seeds_merge_base_and_best_hparams(self):
        ds = Dataset.from_list([row()])
        adapter = DummyAdapter()
        with TemporaryDirectory() as tmp:
            preprocess = PreprocessConfig(resize_mode="letterbox", height=32, width=32)
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
                hparams={"learning_rate": 3.0e-4},
                seeds=[0],
            )

        self.assertEqual(
            adapter.trained[0][1],
            {"warmup_epochs": 3, "learning_rate": 3.0e-4},
        )

    def test_final_seeds_create_one_logger_run_per_seed(self):
        ds = Dataset.from_list([row()])
        adapter = DummyAdapter()
        logger_factory = RecordingLoggerFactory()
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "final"
            preprocess = PreprocessConfig(resize_mode="letterbox", height=32, width=32)
            run_final_seeds(
                adapter=adapter,
                train_ds=ds,
                val_ds=ds,
                test_ds=ds,
                base_train_cfg=TrainConfig(run_key="final", classes=["car"], output_dir=Path(tmp), preprocess=preprocess),
                eval_cfg=EvalConfig(classes=["car"], preprocess=preprocess),
                hparams={"learning_rate": 3.0e-4},
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

    def test_final_seeds_continue_after_middle_seed_failure(self):
        class MiddleSeedFailureAdapter(DummyAdapter):
            def __init__(self):
                super().__init__()
                self.attempted_seeds = []

            def train(self, train_ds, val_ds, train_cfg, *, epoch_eval_cfg=None, logger=None, log_prefix="train"):
                self.attempted_seeds.append(train_cfg.seed)
                if train_cfg.seed == 1:
                    raise RuntimeError("seed 1 failed")
                return super().train(
                    train_ds,
                    val_ds,
                    train_cfg,
                    epoch_eval_cfg=epoch_eval_cfg,
                    logger=logger,
                    log_prefix=log_prefix,
                )

        ds = Dataset.from_list([row()])
        adapter = MiddleSeedFailureAdapter()
        with TemporaryDirectory() as tmp:
            preprocess = PreprocessConfig(resize_mode="letterbox", height=32, width=32)
            runs = run_final_seeds(
                adapter=adapter,
                train_ds=ds,
                val_ds=ds,
                test_ds=ds,
                base_train_cfg=TrainConfig(
                    run_key="final",
                    classes=["car"],
                    output_dir=Path(tmp),
                    preprocess=preprocess,
                ),
                eval_cfg=EvalConfig(classes=["car"], preprocess=preprocess),
                hparams={"learning_rate": 3.0e-4},
                seeds=[0, 1, 2],
            )

        self.assertEqual(adapter.attempted_seeds, [0, 1, 2])
        self.assertEqual([run.state for run in runs], ["complete", "failed", "complete"])
        self.assertIn("seed 1 failed", runs[1].error)
        self.assertEqual(len(adapter.evaluated), 4)

    def test_controlled_final_rejects_legacy_provider_learning_rate_name(self):
        ds = Dataset.from_list([row()])
        with TemporaryDirectory() as tmp:
            preprocess = PreprocessConfig(resize_mode="letterbox", height=32, width=32)
            with self.assertRaisesRegex(ValueError, "canonical 'learning_rate'"):
                run_final_seeds(
                    adapter=DummyAdapter(),
                    train_ds=ds,
                    val_ds=ds,
                    test_ds=ds,
                    base_train_cfg=TrainConfig(
                        run_key="final",
                        classes=["car"],
                        output_dir=Path(tmp),
                        preprocess=preprocess,
                    ),
                    eval_cfg=EvalConfig(classes=["car"], preprocess=preprocess),
                    hparams={"lr0": 3.0e-4},
                    seeds=[0],
                )

    def test_failed_final_seed_preserves_completed_partial_outputs(self):
        class TestEvalFailureAdapter(DummyAdapter):
            def evaluate(self, ds, artifact, eval_cfg, *, logger=None, log_prefix=None):
                if log_prefix == "test":
                    raise RuntimeError("test evaluation failed")
                return super().evaluate(
                    ds,
                    artifact,
                    eval_cfg,
                    logger=logger,
                    log_prefix=log_prefix,
                )

        ds = Dataset.from_list([row()])
        with TemporaryDirectory() as tmp:
            preprocess = PreprocessConfig(resize_mode="letterbox", height=32, width=32)
            runs = run_final_seeds(
                adapter=TestEvalFailureAdapter(),
                train_ds=ds,
                val_ds=ds,
                test_ds=ds,
                base_train_cfg=TrainConfig(
                    run_key="final",
                    classes=["car"],
                    output_dir=Path(tmp),
                    preprocess=preprocess,
                ),
                eval_cfg=EvalConfig(classes=["car"], preprocess=preprocess),
                hparams={"learning_rate": 3.0e-4},
                seeds=[0],
            )

        self.assertEqual(runs[0].state, "failed")
        self.assertIsNotNone(runs[0].artifact)
        self.assertIsNotNone(runs[0].val_result)
        self.assertIsNone(runs[0].test_result)

    def test_final_result_aggregation_uses_successful_metric_intersection_and_sample_std(self):
        def result(metrics):
            return EvalResult(
                model_key="dummy",
                primary_metric="shared",
                primary_metric_value=metrics["shared"],
                metrics=metrics,
            )

        runs = [
            FinalSeedRun(
                seed=0,
                state="complete",
                val_result=result({"shared": 2.0, "seed0_only": 9.0}),
                test_result=result({"shared": 5.0}),
            ),
            FinalSeedRun(seed=1, state="failed", error="planned failure"),
            FinalSeedRun(
                seed=2,
                state="complete",
                val_result=result({"shared": 4.0, "seed2_only": 7.0}),
            ),
        ]

        with TemporaryDirectory() as tmp:
            path = write_final_results(runs, Path(tmp) / "final.json")
            payload = json.loads(path.read_text(encoding="utf-8"))

        aggregate = payload["aggregate"]
        self.assertEqual(aggregate["successful_seeds"], 2)
        self.assertEqual(aggregate["failed_seeds"], 1)
        self.assertEqual(set(aggregate["val"]), {"shared"})
        self.assertEqual(aggregate["val"]["shared"]["mean"], 3.0)
        self.assertAlmostEqual(aggregate["val"]["shared"]["std"], 2.0 ** 0.5)
        self.assertEqual(aggregate["test"]["shared"]["std"], 0.0)


if __name__ == "__main__":
    unittest.main()
