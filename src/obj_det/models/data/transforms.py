from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance

from obj_det.datasets.models import BBox
from obj_det.models.data.sample import DetectionSample, DetectionTarget


class BaseDetectionTransform(ABC):
    @abstractmethod
    def __call__(self, sample: DetectionSample) -> DetectionSample:
        raise NotImplementedError


class NoOpDetectionTransform(BaseDetectionTransform):
    def __call__(self, sample: DetectionSample) -> DetectionSample:
        return sample


class ResizePadDetectionTransform(BaseDetectionTransform):
    def __init__(self, image_size: int):
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        self.image_size = image_size

    def __call__(self, sample: DetectionSample) -> DetectionSample:
        return resize_pad_sample(sample, self.image_size)


class BasicDetectionTransform(BaseDetectionTransform):
    def __init__(
        self,
        image_size: int,
        *,
        horizontal_flip_p: float = 0.5,
        color_jitter_strength: float = 0.1,
    ):
        if not 0.0 <= horizontal_flip_p <= 1.0:
            raise ValueError("horizontal_flip_p must be in [0, 1]")
        if color_jitter_strength < 0.0:
            raise ValueError("color_jitter_strength must be non-negative")
        self.image_size = image_size
        self.horizontal_flip_p = horizontal_flip_p
        self.color_jitter_strength = color_jitter_strength

    def __call__(self, sample: DetectionSample) -> DetectionSample:
        if random.random() < self.horizontal_flip_p:
            sample = horizontal_flip_sample(sample)
        if self.color_jitter_strength > 0:
            sample = color_jitter_sample(sample, self.color_jitter_strength)
        return resize_pad_sample(sample, self.image_size)


class WeatherDetectionTransform(BaseDetectionTransform):
    def __init__(self, image_size: int, params: dict[str, Any] | None = None):
        self.image_size = image_size
        self.params = params or {}
        self.basic = BasicDetectionTransform(
            image_size=image_size,
            horizontal_flip_p=float(self.params.get("horizontal_flip_p", 0.5)),
            color_jitter_strength=float(self.params.get("color_jitter_strength", 0.1)),
        )

    def __call__(self, sample: DetectionSample) -> DetectionSample:
        sample = self.basic(sample)
        effect = str(self.params.get("effect") or sample.condition or "haze").lower()
        image = sample.image.astype(np.float32)
        strength = float(self.params.get("strength", 0.25))

        if effect in {"haze", "fog", "smoke"}:
            image = image * (1.0 - strength) + 255.0 * strength
        elif effect in {"low_light", "night"}:
            image = image * max(0.0, 1.0 - strength)
        elif effect == "rain":
            image = _add_rain(image, strength)
        elif effect == "snow":
            image = _add_snow(image, strength)

        return replace(sample, image=np.clip(image, 0, 255).astype(np.uint8))


def build_detection_transform(
    policy: str,
    image_size: int,
    params: dict[str, Any] | None = None,
) -> BaseDetectionTransform:
    params = params or {}
    if policy == "none":
        return ResizePadDetectionTransform(image_size=image_size)
    if policy == "basic":
        return BasicDetectionTransform(
            image_size=image_size,
            horizontal_flip_p=float(params.get("horizontal_flip_p", 0.5)),
            color_jitter_strength=float(params.get("color_jitter_strength", 0.1)),
        )
    if policy == "weather":
        return WeatherDetectionTransform(image_size=image_size, params=params)
    if policy == "provider":
        return NoOpDetectionTransform()
    raise ValueError(f"Unknown augmentation policy: {policy}")


def horizontal_flip_sample(sample: DetectionSample) -> DetectionSample:
    image = np.ascontiguousarray(np.flip(sample.image, axis=1))
    width = sample.width
    targets = [
        replace(
            target,
            bbox=BBox.from_xywh(
                [
                    width - target.bbox.x - target.bbox.w,
                    target.bbox.y,
                    target.bbox.w,
                    target.bbox.h,
                ]
            ),
        )
        for target in sample.targets
    ]
    meta = dict(sample.meta)
    meta["horizontal_flip"] = True
    return replace(sample, image=image, targets=targets, meta=meta)


def color_jitter_sample(sample: DetectionSample, strength: float) -> DetectionSample:
    image = Image.fromarray(sample.image)
    low = max(0.0, 1.0 - strength)
    high = 1.0 + strength

    for enhancer_cls in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        image = enhancer_cls(image).enhance(random.uniform(low, high))

    return replace(sample, image=np.asarray(image.convert("RGB")))


def resize_pad_sample(sample: DetectionSample, image_size: int) -> DetectionSample:
    image = Image.fromarray(sample.image)
    original_width = sample.width
    original_height = sample.height
    scale = image_size / max(original_width, original_height)
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))
    pad_left = (image_size - resized_width) // 2
    pad_top = (image_size - resized_height) // 2

    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (image_size, image_size), color=(0, 0, 0))
    canvas.paste(resized, (pad_left, pad_top))

    targets: list[DetectionTarget] = []
    for target in sample.targets:
        bbox = BBox.from_xywh(
            [
                target.bbox.x * scale + pad_left,
                target.bbox.y * scale + pad_top,
                target.bbox.w * scale,
                target.bbox.h * scale,
            ]
        ).clipped(image_size, image_size)
        if bbox is None:
            continue
        targets.append(replace(target, bbox=bbox))

    meta = dict(sample.meta)
    meta["preprocess"] = {
        "original_width": original_width,
        "original_height": original_height,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "scale": scale,
    }

    return replace(
        sample,
        image=np.asarray(canvas, dtype=np.uint8),
        width=image_size,
        height=image_size,
        targets=targets,
        meta=meta,
    )


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


def _add_rain(image: np.ndarray, strength: float) -> np.ndarray:
    out = image.copy()
    height, width = out.shape[:2]
    count = max(1, int(width * height * strength / 900))
    for _ in range(count):
        x = random.randrange(width)
        y = random.randrange(height)
        length = random.randint(4, max(5, height // 12))
        x2 = min(width - 1, x + length // 3)
        y2 = min(height - 1, y + length)
        out[y:y2, x:x2 + 1] = np.maximum(out[y:y2, x:x2 + 1], 200)
    return out * (1.0 - strength * 0.25)


def _add_snow(image: np.ndarray, strength: float) -> np.ndarray:
    out = image.copy()
    mask = np.random.random(out.shape[:2]) < min(0.5, strength * 0.2)
    out[mask] = 255
    return out * (1.0 + strength * 0.15)
