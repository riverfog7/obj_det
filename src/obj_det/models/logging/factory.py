from __future__ import annotations

from collections.abc import Callable
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
    wandb_group: str | None = None,
    force_run_name: bool = False,
    force_default_log_path: bool = False,
) -> BaseExperimentLogger | None:
    if cfg.backends == ["none"]:
        return None

    loggers: list[BaseExperimentLogger] = []
    for backend in cfg.backends:
        if backend == "local":
            loggers.append(LocalJsonLogger(default_log_path if force_default_log_path else cfg.local.path or default_log_path))
        elif backend == "wandb":
            loggers.append(
                WandbLogger(
                    project=cfg.wandb.project,
                    entity=cfg.wandb.entity,
                    group=wandb_group if wandb_group is not None else cfg.wandb.group,
                    name=run_name if force_run_name else cfg.wandb.name or run_name,
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


LoggerFactory = Callable[[str, Path], BaseExperimentLogger | None]


def child_logger_factory_from_config(cfg: LoggingConfig, *, wandb_group: str) -> LoggerFactory | None:
    if cfg.backends == ["none"]:
        return None

    def build(run_name: str, default_log_path: Path) -> BaseExperimentLogger | None:
        return logger_from_config(
            cfg,
            default_log_path=default_log_path,
            run_name=run_name,
            wandb_group=wandb_group,
            force_run_name=True,
            force_default_log_path=True,
        )

    return build
