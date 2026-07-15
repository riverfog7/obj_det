from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Sequence

import albumentations as A
import cv2
import numpy as np

from obj_det.datasets.models import BBox
from obj_det.models.data.bbox import bbox_xywh, clip_xywh
from obj_det.models.data.sample import DetectionSample, DetectionTarget
from obj_det.models.schemas.config import AugmentationConfig, PreprocessConfig


class DetectionTransform:
    def __init__(
        self,
        preprocess: PreprocessConfig,
        augmentation: AugmentationConfig | None = None,
        *,
        seed: int | None = None,
    ):
        self.preprocess = preprocess
        self.augmentation = augmentation or AugmentationConfig()
        augmentations = self._augmentations(self.augmentation)
        self.transform = None
        if augmentations:
            self.transform = A.Compose(
                augmentations,
                bbox_params=A.BboxParams(
                    format="coco",
                    label_fields=["target_indices"],
                    clip=True,
                    filter_invalid_bboxes=True,
                ),
                seed=seed,
            )

    def set_seed(self, seed: int) -> None:
        if self.transform is not None:
            self.transform.set_random_seed(seed)

    def __call__(self, sample: DetectionSample) -> DetectionSample:
        if sample.image is None:
            raise ValueError("DetectionTransform requires decoded image data")

        original_width = sample.width
        original_height = sample.height
        result = {
            "image": np.ascontiguousarray(sample.image),
            "bboxes": [target.bbox_xywh for target in sample.targets],
            "target_indices": list(range(len(sample.targets))),
        }
        if self.transform is not None:
            result = self.transform(**result)

        image, resized_boxes, preprocess_meta = _resize_geometry(
            np.asarray(result["image"], dtype=np.uint8),
            result["bboxes"],
            self.preprocess,
            original_width=original_width,
            original_height=original_height,
        )
        height, width = image.shape[:2]
        targets: list[DetectionTarget] = []

        for bbox_values, target_index in zip(resized_boxes, result["target_indices"]):
            try:
                bbox = clip_xywh(bbox_xywh(bbox_values), width, height)
            except ValueError:
                continue
            if bbox is None:
                continue
            targets.append(replace(sample.targets[int(round(float(target_index)))], bbox_xywh=bbox))

        meta = dict(sample.meta)
        meta["preprocess"] = preprocess_meta

        return replace(sample, image=image, width=width, height=height, targets=targets, meta=meta)

    def _augmentations(self, augmentation: AugmentationConfig) -> list[Any]:
        transforms: list[Any] = []
        if augmentation.policy in {"basic", "weather"} and augmentation.horizontal_flip_p > 0:
            transforms.append(A.HorizontalFlip(p=augmentation.horizontal_flip_p))
        if (
            augmentation.policy in {"basic", "weather"}
            and augmentation.color_jitter_strength > 0
            and augmentation.color_jitter_p > 0
        ):
            hue = min(0.5, augmentation.color_jitter_strength * 0.5)
            transforms.append(
                A.ColorJitter(
                    brightness=augmentation.color_jitter_strength,
                    contrast=augmentation.color_jitter_strength,
                    saturation=augmentation.color_jitter_strength,
                    hue=hue,
                    p=augmentation.color_jitter_p,
                )
            )
        return transforms


def build_detection_transform(
    preprocess: PreprocessConfig,
    augmentation: AugmentationConfig | None = None,
    *,
    seed: int | None = None,
) -> DetectionTransform:
    return DetectionTransform(preprocess, augmentation, seed=seed)


def _resize_geometry(
    image: np.ndarray,
    boxes: Sequence[Sequence[float]],
    preprocess: PreprocessConfig,
    *,
    original_width: int,
    original_height: int,
) -> tuple[np.ndarray, list[list[float]], dict[str, Any]]:
    source_height, source_width = image.shape[:2]
    resized_width, resized_height, output_width, output_height, pad_left, pad_top = _target_geometry(
        source_width,
        source_height,
        preprocess,
    )
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    if pad_left or pad_top or output_width != resized_width or output_height != resized_height:
        output = np.zeros((output_height, output_width, image.shape[2]), dtype=np.uint8)
        output[pad_top : pad_top + resized_height, pad_left : pad_left + resized_width] = resized
    else:
        output = resized

    scale_x = resized_width / source_width
    scale_y = resized_height / source_height
    resized_boxes = [
        [
            float(x) * scale_x + pad_left,
            float(y) * scale_y + pad_top,
            float(w) * scale_x,
            float(h) * scale_y,
        ]
        for x, y, w, h in boxes
    ]
    meta = {
        "original_width": original_width,
        "original_height": original_height,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "scale": min(scale_x, scale_y),
        "scale_x": scale_x,
        "scale_y": scale_y,
        "resize_mode": preprocess.resize_mode,
    }
    return np.ascontiguousarray(output), resized_boxes, meta


def _target_geometry(
    width: int,
    height: int,
    preprocess: PreprocessConfig,
) -> tuple[int, int, int, int, int, int]:
    if preprocess.resize_mode == "exact":
        width_out = int(preprocess.width)
        height_out = int(preprocess.height)
        return width_out, height_out, width_out, height_out, 0, 0

    if preprocess.resize_mode == "letterbox":
        width_out = int(preprocess.width)
        height_out = int(preprocess.height)
        scale = min(width_out / width, height_out / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        pad_left = (width_out - resized_width) // 2
        pad_top = (height_out - resized_height) // 2
        return (
            resized_width,
            resized_height,
            width_out,
            height_out,
            pad_left,
            pad_top,
        )

    shortest_edge = int(preprocess.shortest_edge)
    longest_edge = int(preprocess.longest_edge)
    scale = min(shortest_edge / min(width, height), longest_edge / max(width, height))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return resized_width, resized_height, resized_width, resized_height, 0, 0


def bbox_to_original(bbox: BBox, preprocess: dict[str, Any]) -> BBox | None:
    scale = float(preprocess["scale"])
    scale_x = float(preprocess.get("scale_x", scale))
    scale_y = float(preprocess.get("scale_y", scale))
    pad_left = float(preprocess["pad_left"])
    pad_top = float(preprocess["pad_top"])
    original_width = int(preprocess["original_width"])
    original_height = int(preprocess["original_height"])

    x1, y1, x2, y2 = bbox.xyxy()
    x1 = (x1 - pad_left) / scale_x
    y1 = (y1 - pad_top) / scale_y
    x2 = (x2 - pad_left) / scale_x
    y2 = (y2 - pad_top) / scale_y

    x1 = max(0.0, min(x1, float(original_width)))
    y1 = max(0.0, min(y1, float(original_height)))
    x2 = max(0.0, min(x2, float(original_width)))
    y2 = max(0.0, min(y2, float(original_height)))

    if x2 <= x1 or y2 <= y1:
        return None

    return BBox.from_xyxy([x1, y1, x2, y2])


def canonicalize_prediction_bbox(
    xyxy: Sequence[float],
    *,
    image_width: int,
    image_height: int,
    preprocess: dict[str, Any] | None = None,
) -> BBox | None:
    if len(xyxy) != 4 or image_width <= 0 or image_height <= 0:
        return None
    try:
        x1, y1, x2, y2 = map(float, xyxy)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None

    x1 = max(0.0, min(x1, float(image_width)))
    y1 = max(0.0, min(y1, float(image_height)))
    x2 = max(0.0, min(x2, float(image_width)))
    y2 = max(0.0, min(y2, float(image_height)))
    if x2 <= x1 or y2 <= y1:
        return None

    bbox = BBox.from_xyxy([x1, y1, x2, y2])
    return bbox_to_original(bbox, preprocess) if preprocess is not None else bbox
