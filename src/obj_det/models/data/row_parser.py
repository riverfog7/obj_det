from __future__ import annotations

import io
import json
import logging
from typing import Any

import cv2
import numpy as np
from PIL import Image

from obj_det.models.data.bbox import bbox_xywh
from obj_det.models.data.sample import DetectionSample, DetectionTarget
from obj_det.models.schemas.config import LabelMode


logger = logging.getLogger(__name__)


class HFDetectionRowParser:
    """Convert one standardized HF row into a runtime detection sample."""

    def __init__(self, classes: list[str], label_mode: LabelMode, *, decode_backend: str = "pil"):
        self.classes = classes
        self.label_mode = label_mode
        self.class_to_id = {name: idx for idx, name in enumerate(classes)}
        if decode_backend not in {"pil", "opencv"}:
            raise ValueError(f"Unknown decode_backend: {decode_backend}")
        self.decode_backend = decode_backend

    def parse(self, row: dict[str, Any], *, decode_image: bool = True) -> DetectionSample:
        image = self.decode_image(row["image"]) if decode_image else None
        targets: list[DetectionTarget] = []

        for obj in row.get("objects", []):
            if obj.get("ignore", False):
                continue

            label = obj.get("native_label") if self.label_mode == "native" else obj.get("meta_label")
            if label is None or label == "":
                continue

            if label not in self.class_to_id:
                logger.debug("Skipping object with label outside class list: %s", label)
                continue

            try:
                bbox = bbox_xywh(obj["bbox"])
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping object with invalid bbox: %s", exc)
                continue

            targets.append(
                DetectionTarget(
                    bbox_xywh=bbox,
                    label=label,
                    label_id=self.class_to_id[label],
                    iscrowd=bool(obj.get("iscrowd", False)),
                    meta=self._parse_json_field(obj.get("meta_json")),
                )
            )

        return DetectionSample(
            image=image,
            image_id=str(row["image_id"]),
            dataset=str(row["dataset"]),
            split=str(row["split"]),
            width=int(row["width"]),
            height=int(row["height"]),
            targets=targets,
            condition=(row.get("condition") or "unknown"),
            domain=(row.get("domain") or "unknown"),
            is_synthetic=row.get("is_synthetic"),
            meta=self._parse_json_field(row.get("meta_json")),
        )

    def parse_targets_only(self, row: dict[str, Any]) -> DetectionSample:
        return self.parse(row, decode_image=False)

    def decode_image(self, image_field: Any) -> np.ndarray:
        if isinstance(image_field, dict):
            image_bytes = image_field.get("bytes")
            image_path = image_field.get("path")

            if image_bytes is not None:
                return self._decode_bytes(image_bytes)

            if image_path is not None:
                return self._decode_path(image_path)

        if isinstance(image_field, Image.Image):
            return np.asarray(image_field.convert("RGB"))

        if isinstance(image_field, np.ndarray):
            return self._normalize_array(image_field)

        raise TypeError(f"Unsupported image field type: {type(image_field)}")

    def _decode_bytes(self, image_bytes: bytes) -> np.ndarray:
        if self.decode_backend == "opencv":
            try:
                buffer = np.frombuffer(image_bytes, dtype=np.uint8)
                bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                if bgr is not None:
                    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            except cv2.error as exc:
                logger.warning("OpenCV image decode failed; falling back to PIL: %s", exc)

        with Image.open(io.BytesIO(image_bytes)) as image:
            return np.asarray(image.convert("RGB"))

    def _decode_path(self, image_path: str) -> np.ndarray:
        if self.decode_backend == "opencv":
            try:
                bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if bgr is not None:
                    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            except cv2.error as exc:
                logger.warning("OpenCV image decode failed; falling back to PIL: %s", exc)

        with Image.open(image_path) as image:
            return np.asarray(image.convert("RGB"))

    def _normalize_array(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.ndim != 3:
            raise ValueError(f"Expected image array with 2 or 3 dimensions, got {image.ndim}")
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        if image.shape[-1] < 3:
            raise ValueError(f"Expected image array with at least 3 channels, got {image.shape[-1]}")
        return image[..., :3].astype(np.uint8, copy=False)

    def _parse_json_field(self, value: Any) -> dict[str, Any]:
        if value is None or value == "":
            return {}
        if isinstance(value, dict):
            return value
        return json.loads(value)
