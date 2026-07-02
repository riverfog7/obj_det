from __future__ import annotations

from obj_det.models.schemas.tuning import SearchSpace


YOLO_CONTROLLED_SEARCH_SPACE = SearchSpace(
    params={
        "lr0": {"type": "float", "low": 0.001, "high": 0.01, "log": True},
        "weight_decay": {"type": "categorical", "choices": [0.0001, 0.0005, 0.001]},
        "momentum": {"type": "categorical", "choices": [0.9, 0.937]},
        "warmup_epochs": {"type": "categorical", "choices": [0, 3, 5]},
        "cos_lr": {"type": "categorical", "choices": [True, False]},
    }
)

HF_TRANSFORMER_SEARCH_SPACE = SearchSpace(
    params={
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-4, "log": True},
        "weight_decay": {"type": "categorical", "choices": [0.0, 0.0001, 0.01]},
        "warmup_ratio": {"type": "categorical", "choices": [0.0, 0.05, 0.1]},
        "lr_scheduler_type": {"type": "categorical", "choices": ["linear", "cosine"]},
        "max_grad_norm": {"type": "categorical", "choices": [0.1, 1.0]},
    }
)

TORCHVISION_SEARCH_SPACE = SearchSpace(
    params={
        "optimizer": {"type": "categorical", "choices": ["sgd", "adamw"]},
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "weight_decay": {"type": "categorical", "choices": [0.0001, 0.0005, 0.001]},
        "momentum": {"type": "categorical", "choices": [0.9]},
        "scheduler": {"type": "categorical", "choices": ["step", "cosine"]},
    }
)
