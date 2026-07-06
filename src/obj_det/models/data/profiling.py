from __future__ import annotations

import time
from typing import Any, Iterable

from obj_det.models.data.row_parser import HFDetectionRowParser
from obj_det.models.data.sample import DetectionSample
from obj_det.models.schemas.config import LabelMode


def measure_dataloader(loader: Iterable[Any], *, max_batches: int = 100) -> dict[str, float]:
    start = time.perf_counter()
    batches = 0
    for batches, _batch in enumerate(loader, start=1):
        if batches >= max_batches:
            break
    elapsed = time.perf_counter() - start
    return {
        "batches": float(batches),
        "seconds": elapsed,
        "batches_per_second": batches / elapsed if elapsed > 0 else 0.0,
    }


def measure_decode_backend(
    rows: Iterable[dict[str, Any]],
    *,
    classes: list[str],
    label_mode: LabelMode,
    decode_backend: str,
    max_images: int = 1000,
) -> dict[str, float]:
    parser = HFDetectionRowParser(classes=classes, label_mode=label_mode, decode_backend=decode_backend)
    start = time.perf_counter()
    images = 0
    for images, row in enumerate(rows, start=1):
        parser.decode_image(row["image"])
        if images >= max_images:
            break
    elapsed = time.perf_counter() - start
    return {
        "images": float(images),
        "seconds": elapsed,
        "images_per_second": images / elapsed if elapsed > 0 else 0.0,
    }


def measure_transform(
    samples: Iterable[DetectionSample],
    transform,
    *,
    max_samples: int = 1000,
) -> dict[str, float]:
    start = time.perf_counter()
    count = 0
    for count, sample in enumerate(samples, start=1):
        transform(sample)
        if count >= max_samples:
            break
    elapsed = time.perf_counter() - start
    return {
        "samples": float(count),
        "seconds": elapsed,
        "samples_per_second": count / elapsed if elapsed > 0 else 0.0,
    }
