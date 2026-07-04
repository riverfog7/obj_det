from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from datasets import Dataset

from obj_det.models.schemas.artifact import ModelArtifact
from obj_det.models.schemas.config import EvalConfig, ModelConfig, PredictConfig, TrainConfig
from obj_det.models.schemas.prediction import PredictionRecord
from obj_det.models.schemas.result import EvalResult


class BaseModelAdapter(ABC):
    """Base interface for all HF-dataset-first detection backends."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.key = cfg.key
        self.backend = cfg.backend

    @abstractmethod
    def train(self, train_ds: Dataset, val_ds: Dataset, train_cfg: TrainConfig) -> ModelArtifact:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        ds: Dataset,
        artifact: ModelArtifact,
        predict_cfg: PredictConfig,
    ) -> Iterator[PredictionRecord]:
        raise NotImplementedError

    def evaluate(self, ds: Dataset, artifact: ModelArtifact, eval_cfg: EvalConfig) -> EvalResult:
        from obj_det.models.evaluation.evaluator import DetectionEvaluator

        predict_cfg = PredictConfig(
            classes=eval_cfg.classes,
            label_mode=eval_cfg.label_mode,
            batch_size=eval_cfg.batch_size,
            transform=eval_cfg.transform,
            conf_threshold=eval_cfg.conf_threshold,
            iou_threshold=eval_cfg.iou_threshold,
            backend_params=eval_cfg.backend_params,
        )
        predictions = list(self.predict(ds, artifact, predict_cfg))
        evaluator = DetectionEvaluator()
        return evaluator.evaluate(ds, predictions, eval_cfg, model_key=self.key)
