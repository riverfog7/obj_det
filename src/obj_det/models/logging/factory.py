from __future__ import annotations

from pathlib import Path

from obj_det.models.schemas.logging import LoggingConfig

from .base import BaseExperimentLogger
from .composite import CompositeLogger
from .local import LocalJsonLogger
from .null import NullLogger
from .wandb import WandbLogger


def logger_from_config(
    cfg: LoggingConfig,
    *,
    default_log_path: Path,
    run_name: str,
) -> BaseExperimentLogger | None:
    if cfg.backends == ["none"]:
        return None

    loggers: list[BaseExperimentLogger] = []
    for backend in cfg.backends:
        if backend == "local":
            loggers.append(LocalJsonLogger(cfg.local.path or default_log_path))
        elif backend == "wandb":
            loggers.append(
                WandbLogger(
                    project=cfg.wandb.project,
                    entity=cfg.wandb.entity,
                    group=cfg.wandb.group,
                    name=cfg.wandb.name or run_name,
                    mode=cfg.wandb.mode,
                    tags=cfg.wandb.tags,
                )
            )
        elif backend == "none":
            loggers.append(NullLogger())
        else:
            raise ValueError(f"Unknown logging backend: {backend}")

    if not loggers:
        return None
    if len(loggers) == 1:
        return loggers[0]
    return CompositeLogger(loggers)
