from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any

import albumentations as A
import numpy as np
from PIL import Image

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
        seed: int | None = None,
    ):
        if not 0.0 <= horizontal_flip_p <= 1.0:
            raise ValueError("horizontal_flip_p must be in [0, 1]")
        if color_jitter_strength < 0.0:
            raise ValueError("color_jitter_strength must be non-negative")
        self.image_size = image_size
        self.horizontal_flip_p = horizontal_flip_p
        self.color_jitter_strength = color_jitter_strength
        self.seed = seed
        transforms = []
        if self.horizontal_flip_p > 0:
            transforms.append(A.HorizontalFlip(p=self.horizontal_flip_p))
        if self.color_jitter_strength > 0:
            hue = min(0.5, self.color_jitter_strength * 0.5)
            transforms.append(
                A.ColorJitter(
                    brightness=self.color_jitter_strength,
                    contrast=self.color_jitter_strength,
                    saturation=self.color_jitter_strength,
                    hue=hue,
                    p=1.0,
                )
            )
        self.transform = _make_albumentations_transform(
            transforms,
            seed=self.seed,
            uses_bboxes=self.horizontal_flip_p > 0,
        )

    def __call__(self, sample: DetectionSample) -> DetectionSample:
        sample = _apply_albumentations(sample, self.transform)
        return resize_pad_sample(sample, self.image_size)


class WeatherDetectionTransform(BaseDetectionTransform):
    def __init__(self, image_size: int, params: dict[str, Any] | None = None):
        self.image_size = image_size
        self.params = params or {}
        self.basic = BasicDetectionTransform(
            image_size=image_size,
            horizontal_flip_p=float(self.params.get("horizontal_flip_p", 0.5)),
            color_jitter_strength=float(self.params.get("color_jitter_strength", 0.1)),
            seed=_optional_int(self.params.get("seed")),
        )
        self.seed = _optional_int(self.params.get("seed"))
        self._weather_cache: dict[tuple[str, float], Any] = {}

    def __call__(self, sample: DetectionSample) -> DetectionSample:
        sample = self.basic(sample)
        effect = str(self.params.get("effect") or sample.condition or "haze").lower()
        strength = float(self.params.get("strength", 0.25))
        transform = self._weather_transform(effect, strength)
        return _apply_albumentations(sample, transform)

    def _weather_transform(self, effect: str, strength: float):
        key = (effect, max(0.0, min(float(strength), 1.0)))
        if key not in self._weather_cache:
            self._weather_cache[key] = _make_albumentations_transform(
                _weather_transforms(*key),
                seed=self.seed,
                uses_bboxes=False,
            )
        return self._weather_cache[key]


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
            seed=_optional_int(params.get("seed")),
        )
    if policy == "weather":
        return WeatherDetectionTransform(image_size=image_size, params=params)
    if policy == "provider":
        return NoOpDetectionTransform()
    raise ValueError(f"Unknown augmentation policy: {policy}")


def _make_albumentations_transform(
    transforms: list[Any],
    *,
    seed: int | None = None,
    uses_bboxes: bool,
):
    if not transforms:
        return None
    bbox_params = None
    if uses_bboxes:
        bbox_params = A.BboxParams(
            format="coco",
            label_fields=["target_indices"],
            clip=True,
            filter_invalid_bboxes=True,
        )
    return A.Compose(transforms, bbox_params=bbox_params, seed=seed), uses_bboxes


def _apply_albumentations(sample: DetectionSample, transform) -> DetectionSample:
    if transform is None:
        return sample

    compose, uses_bboxes = transform
    if not uses_bboxes:
        result = compose(image=np.ascontiguousarray(sample.image))
        image = np.asarray(result["image"], dtype=np.uint8)
        height, width = image.shape[:2]
        return replace(sample, image=image, width=width, height=height)

    result = compose(
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

    return replace(sample, image=image, width=width, height=height, targets=targets)


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


def _weather_transforms(effect: str, strength: float) -> list[Any]:
    strength = max(0.0, min(float(strength), 1.0))
    if effect in {"haze", "fog", "smoke"}:
        fog = max(0.01, strength)
        return [A.RandomFog(alpha_coef=0.08, fog_coef_range=(fog, fog), p=1.0)]
    if effect in {"low_light", "night"}:
        brightness = -strength
        return [
            A.RandomBrightnessContrast(
                brightness_limit=(brightness, brightness),
                contrast_limit=(0.0, 0.0),
                p=1.0,
            )
        ]
    if effect == "rain":
        return [
            A.RandomRain(
                brightness_coefficient=max(0.0, 1.0 - strength * 0.7),
                rain_type="default",
                p=1.0,
            )
        ]
    if effect == "snow":
        low = min(0.5, strength * 0.2)
        high = max(low, min(0.8, strength * 0.3 + 0.1))
        return [A.RandomSnow(snow_point_range=(low, high), p=1.0)]
    return []


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
