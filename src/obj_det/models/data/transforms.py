from __future__ import annotations

from dataclasses import replace
from typing import Any

import albumentations as A
import cv2
import numpy as np

from obj_det.datasets.models import BBox
from obj_det.models.data.sample import DetectionSample, DetectionTarget
from obj_det.models.schemas.config import TransformConfig


class DetectionTransform:
    def __init__(self, cfg: TransformConfig, *, seed: int | None = None):
        self.cfg = cfg
        self.transform = A.Compose(
            self._transforms(cfg),
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["target_indices"],
                clip=True,
                filter_invalid_bboxes=True,
            ),
            seed=seed,
        )

    def set_seed(self, seed: int) -> None:
        self.transform.set_random_seed(seed)

    def __call__(self, sample: DetectionSample) -> DetectionSample:
        original_width = sample.width
        original_height = sample.height
        result = self.transform(
            image=np.ascontiguousarray(sample.image),
            bboxes=[target.bbox.xywh() for target in sample.targets],
            target_indices=list(range(len(sample.targets))),
        )

        image = np.asarray(result["image"], dtype=np.uint8)
        height, width = image.shape[:2]
        targets: list[DetectionTarget] = []

        for bbox_values, target_index in zip(result["bboxes"], result["target_indices"]):
            try:
                bbox = BBox.from_xywh(list(bbox_values)).clipped(width, height)
            except ValueError:
                continue
            if bbox is None:
                continue
            targets.append(replace(sample.targets[int(round(float(target_index)))], bbox=bbox))

        meta = dict(sample.meta)
        meta["preprocess"] = _preprocess_meta(original_width, original_height, self.cfg.image_size)

        return replace(sample, image=image, width=width, height=height, targets=targets, meta=meta)

    def _transforms(self, cfg: TransformConfig) -> list[Any]:
        transforms: list[Any] = []
        if cfg.horizontal_flip_p > 0:
            transforms.append(A.HorizontalFlip(p=cfg.horizontal_flip_p))
        if cfg.color_jitter_strength > 0:
            hue = min(0.5, cfg.color_jitter_strength * 0.5)
            transforms.append(
                A.ColorJitter(
                    brightness=cfg.color_jitter_strength,
                    contrast=cfg.color_jitter_strength,
                    saturation=cfg.color_jitter_strength,
                    hue=hue,
                    p=1.0,
                )
            )
        transforms.extend(
            [
                A.LongestMaxSize(max_size=cfg.image_size, interpolation=cv2.INTER_LINEAR, p=1.0),
                A.PadIfNeeded(
                    min_height=cfg.image_size,
                    min_width=cfg.image_size,
                    position="center",
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=(0, 0, 0),
                    p=1.0,
                ),
            ]
        )
        return transforms


def build_detection_transform(cfg: TransformConfig, *, seed: int | None = None) -> DetectionTransform:
    return DetectionTransform(cfg, seed=seed)


def _preprocess_meta(original_width: int, original_height: int, image_size: int) -> dict[str, Any]:
    scale = image_size / max(original_width, original_height)
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))
    pad_left = (image_size - resized_width) // 2
    pad_top = (image_size - resized_height) // 2

    return {
        "original_width": original_width,
        "original_height": original_height,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "scale": scale,
    }


def bbox_to_original(bbox: BBox, preprocess: dict[str, Any]) -> BBox | None:
    scale = float(preprocess["scale"])
    pad_left = float(preprocess["pad_left"])
    pad_top = float(preprocess["pad_top"])
    original_width = int(preprocess["original_width"])
    original_height = int(preprocess["original_height"])

    x1, y1, x2, y2 = bbox.xyxy()
    x1 = (x1 - pad_left) / scale
    y1 = (y1 - pad_top) / scale
    x2 = (x2 - pad_left) / scale
    y2 = (y2 - pad_top) / scale

    x1 = max(0.0, min(x1, float(original_width)))
    y1 = max(0.0, min(y1, float(original_height)))
    x2 = max(0.0, min(x2, float(original_width)))
    y2 = max(0.0, min(y2, float(original_height)))

    if x2 <= x1 or y2 <= y1:
        return None

    return BBox.from_xyxy([x1, y1, x2, y2])
