from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from obj_det.models.data.bbox import BBoxXYWH


@dataclass
class DetectionTarget:
    bbox_xywh: BBoxXYWH
    label: str
    label_id: int
    iscrowd: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionSample:
    image: np.ndarray | None
    image_id: str
    dataset: str
    split: str
    width: int
    height: int
    targets: list[DetectionTarget]
    condition: str = "unknown"
    domain: str = "unknown"
    is_synthetic: bool | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionBatch:
    images: Any
    targets: Any
    samples: list[DetectionSample] | None = None
    meta: dict[str, Any] = field(default_factory=dict)
