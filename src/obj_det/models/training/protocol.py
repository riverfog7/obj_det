from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


_NORMALIZATION_MODULES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.LayerNorm,
    nn.GroupNorm,
)


def require_single_process(*, context: str) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        raise NotImplementedError(
            f"{context} requires a single process because shared checkpoint evaluation, "
            "Optuna state, and checkpoint manifests are intentionally one-writer"
        )


def build_adamw_param_groups(model: nn.Module, *, weight_decay: float) -> list[dict[str, Any]]:
    """Apply one decay rule across backends: no decay for biases or normalization parameters."""

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    seen: set[int] = set()

    for module in model.modules():
        is_normalization = isinstance(module, _NORMALIZATION_MODULES) or (
            "norm" in module.__class__.__name__.lower()
        )
        for name, parameter in module.named_parameters(recurse=False):
            parameter_id = id(parameter)
            if parameter_id in seen or not parameter.requires_grad:
                continue
            seen.add(parameter_id)
            if name == "bias" or is_normalization:
                no_decay.append(parameter)
            else:
                decay.append(parameter)

    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay)})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if not groups:
        raise ValueError("Cannot construct AdamW parameter groups: model has no trainable parameters")
    return groups


def optimizer_steps_per_epoch(num_batches: int) -> int:
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")
    return num_batches


def warmup_cosine_factor(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> float:
    if step < 0:
        raise ValueError("step cannot be negative")
    if warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if warmup_steps >= total_steps:
        raise ValueError("warmup_steps must be less than total_steps")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be between 0 and 1")

    if step < warmup_steps:
        return step / max(1, warmup_steps)

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_cosine_factor(
            step,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio,
        ),
    )


def require_metric(metrics: Mapping[str, float], name: str, *, context: str) -> float:
    if name not in metrics:
        available = ", ".join(sorted(metrics)) or "<none>"
        raise ValueError(f"Missing required {context} metric {name!r}; available metrics: {available}")
    value = float(metrics[name])
    if not math.isfinite(value):
        raise ValueError(f"Required {context} metric {name!r} is not finite: {value!r}")
    return value


@dataclass
class EarlyStoppingState:
    best_metric: float = -math.inf
    best_epoch: int | None = None
    bad_epochs: int = 0

    def update(self, epoch: int, metric: float, cfg: Any) -> bool:
        if epoch <= 0:
            raise ValueError("epoch must be one-based and positive")
        if not math.isfinite(metric):
            raise ValueError(f"Early-stopping metric is not finite: {metric!r}")

        if metric > self.best_metric + float(cfg.min_delta):
            self.best_metric = metric
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        return bool(
            cfg.enabled
            and epoch >= int(cfg.min_epochs)
            and self.bad_epochs >= int(cfg.patience)
        )


@dataclass
class CheckpointState:
    output_dir: Path
    best_checkpoint: Path | None = None
    last_checkpoint: Path | None = None
    best_metric: float = -math.inf
    best_epoch: int | None = None
    stopped_early: bool = False
    early_stopping: EarlyStoppingState | None = None

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.early_stopping is None:
            self.early_stopping = EarlyStoppingState()

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "checkpoints" / "manifest.json"

    def record_epoch(
        self,
        *,
        epoch: int,
        checkpoint_path: Path,
        metric: float | None,
        early_stopping_cfg: Any,
    ) -> bool:
        checkpoint_path = Path(checkpoint_path)
        self.last_checkpoint = checkpoint_path
        should_stop = False
        if metric is not None:
            if metric > self.best_metric:
                self.best_metric = metric
                self.best_epoch = epoch
                self.best_checkpoint = checkpoint_path
            should_stop = self.early_stopping.update(epoch, metric, early_stopping_cfg)
        self.stopped_early = self.stopped_early or should_stop
        self.write_manifest()
        return should_stop

    def write_manifest(self) -> Path:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        early = self.early_stopping or EarlyStoppingState()
        payload = {
            "best_checkpoint": str(self.best_checkpoint) if self.best_checkpoint is not None else None,
            "last_checkpoint": str(self.last_checkpoint) if self.last_checkpoint is not None else None,
            "best_epoch": self.best_epoch,
            "best_metric": None if self.best_epoch is None else self.best_metric,
            "bad_epochs": early.bad_epochs,
            "stopped_early": self.stopped_early,
            "early_stopping_state": (
                {
                    **asdict(early),
                    "best_metric": None if early.best_epoch is None else early.best_metric,
                }
            ),
        }
        self.manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return self.manifest_path

    def artifact_meta(self) -> dict[str, Any]:
        early = self.early_stopping or EarlyStoppingState()
        early_payload = asdict(early)
        if early.best_epoch is None:
            early_payload["best_metric"] = None
        return {
            "last_checkpoint": str(self.last_checkpoint) if self.last_checkpoint is not None else None,
            "best_epoch": self.best_epoch,
            "best_metric": None if self.best_epoch is None else self.best_metric,
            "stopped_early": self.stopped_early,
            "checkpoint_manifest": str(self.manifest_path),
            "early_stopping_state": early_payload,
        }
