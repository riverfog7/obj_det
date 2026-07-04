from __future__ import annotations

import math
import re
from typing import Any

from obj_det.models.schemas.result import EvalResult


_SAFE_PART = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize_metric_part(value: Any) -> str:
    text = str(value).strip().lower()
    text = _SAFE_PART.sub("_", text).strip("_")
    return text or "unknown"


def scalar_or_none(value: Any) -> int | float | bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        number = value
    elif hasattr(value, "item"):
        return scalar_or_none(value.item())
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            number = float(stripped)
        except ValueError:
            return None
    else:
        return None

    if isinstance(number, float) and not math.isfinite(number):
        return None
    return number


def flatten_scalar_mapping(prefix: str, row: dict[str, Any]) -> tuple[dict[str, int | float | bool], int | None]:
    step = None
    raw_step = row.get("step", row.get("epoch"))
    step_value = scalar_or_none(raw_step)
    if isinstance(step_value, bool):
        step = None
    elif step_value is not None:
        step = int(step_value)

    metrics: dict[str, int | float | bool] = {}
    for key, value in row.items():
        if key in {"step"}:
            continue
        scalar = scalar_or_none(value)
        if scalar is None:
            continue
        metrics[f"{prefix}/{sanitize_metric_part(key)}"] = scalar
    return metrics, step


def flatten_prefixed_scalar_mapping(prefix: str, row: dict[str, Any]) -> tuple[dict[str, int | float | bool], int | None]:
    """
    Flatten scalar training rows without duplicating an existing prefix.

    Ultralytics rows commonly already contain keys such as ``train/box_loss``.
    The generic flatten_scalar_mapping("train", row) would turn that into
    ``train/train_box_loss``. This helper keeps it as ``train/box_loss`` while
    still prefixing unqualified keys such as ``loss``.
    """
    step = None
    raw_step = row.get("step", row.get("epoch"))
    step_value = scalar_or_none(raw_step)
    if not isinstance(step_value, bool) and step_value is not None:
        step = int(step_value)

    base = sanitize_metric_part(prefix.strip("/") or "metrics")
    metrics: dict[str, int | float | bool] = {}
    for key, value in row.items():
        if key in {"step"}:
            continue
        scalar = scalar_or_none(value)
        if scalar is None:
            continue

        parts = [sanitize_metric_part(part) for part in str(key).split("/") if part.strip()]
        if not parts:
            continue
        metric_key = "/".join(parts) if parts[0] == base else f"{base}/{'/'.join(parts)}"
        metrics[metric_key] = scalar

    return metrics, step


def flatten_eval_result(result: EvalResult, prefix: str | None = None) -> dict[str, int | float | bool]:
    base = prefix or f"eval/{sanitize_metric_part(result.split or 'unknown')}"
    metrics: dict[str, int | float | bool] = {}

    for name, value in result.metrics.items():
        scalar = scalar_or_none(value)
        if scalar is not None:
            metrics[f"{base}/{sanitize_metric_part(name)}"] = scalar

    metrics[f"{base}/primary/{sanitize_metric_part(result.primary_metric)}"] = result.primary_metric_value
    metrics[f"{base}/num_images"] = result.num_images
    metrics[f"{base}/num_ground_truth_objects"] = result.num_ground_truth_objects
    metrics[f"{base}/num_predictions"] = result.num_predictions

    for class_name, class_metrics in result.per_class.items():
        group = sanitize_metric_part(class_name)
        for name, value in class_metrics.items():
            scalar = scalar_or_none(value)
            if scalar is not None:
                metrics[f"{base}/per_class/{group}/{sanitize_metric_part(name)}"] = scalar

    for condition, condition_metrics in result.per_condition.items():
        group = sanitize_metric_part(condition)
        for name, value in condition_metrics.items():
            scalar = scalar_or_none(value)
            if scalar is not None:
                metrics[f"{base}/per_condition/{group}/{sanitize_metric_part(name)}"] = scalar

    for domain, domain_metrics in result.per_domain.items():
        group = sanitize_metric_part(domain)
        for name, value in domain_metrics.items():
            scalar = scalar_or_none(value)
            if scalar is not None:
                metrics[f"{base}/per_domain/{group}/{sanitize_metric_part(name)}"] = scalar

    for size, size_metrics in result.per_size.items():
        group = sanitize_metric_part(size)
        for name, value in size_metrics.items():
            scalar = scalar_or_none(value)
            if scalar is not None:
                metrics[f"{base}/per_size/{group}/{sanitize_metric_part(name)}"] = scalar

    return metrics
