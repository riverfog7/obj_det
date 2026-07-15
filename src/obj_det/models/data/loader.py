from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

from obj_det.models.schemas.config import DataLoaderConfig


def dataloader_kwargs(cfg: DataLoaderConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
    }

    if cfg.num_workers > 0:
        if cfg.persistent_workers is not None:
            kwargs["persistent_workers"] = cfg.persistent_workers
        if cfg.prefetch_factor is not None:
            kwargs["prefetch_factor"] = cfg.prefetch_factor

    return kwargs


def seed_worker_transform(transform) -> None:
    worker = torch.utils.data.get_worker_info()
    if worker is None:
        return

    seed = torch.initial_seed() % (2**32)
    if getattr(transform, "_obj_det_worker_seed", None) == seed:
        return

    if hasattr(transform, "set_seed"):
        transform.set_seed(seed)
    transform._obj_det_worker_seed = seed


def seed_dataloader_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
