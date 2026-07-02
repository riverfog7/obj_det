from __future__ import annotations

from obj_det.models.adapters.base import BaseModelAdapter
from obj_det.models.adapters.hf_trainer import HFTrainerDetectionAdapter
from obj_det.models.adapters.torchvision import TorchvisionDetectionAdapter
from obj_det.models.adapters.ultralytics import UltralyticsDetectionAdapter
from obj_det.models.schemas.config import ModelConfig


MODEL_BACKENDS: dict[str, type[BaseModelAdapter]] = {
    "hf_trainer": HFTrainerDetectionAdapter,
    "ultralytics": UltralyticsDetectionAdapter,
    "torchvision": TorchvisionDetectionAdapter,
}


def model_adapter_from_config(cfg: ModelConfig) -> BaseModelAdapter:
    if cfg.backend not in MODEL_BACKENDS:
        raise ValueError(
            f"Unsupported backend={cfg.backend!r}. Supported backends: {sorted(MODEL_BACKENDS)}"
        )
    return MODEL_BACKENDS[cfg.backend](cfg)
