from __future__ import annotations

from typing import Sequence


BBoxXYWH = tuple[float, float, float, float]
BBoxXYXY = tuple[float, float, float, float]


def bbox_xywh(values: Sequence[float]) -> BBoxXYWH:
    if len(values) != 4:
        raise ValueError(f"Expected 4 bbox values, got {len(values)}")
    x, y, w, h = map(float, values)
    if x < 0 or y < 0:
        raise ValueError(f"Invalid bbox origin: {values}")
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid bbox size: {values}")
    return x, y, w, h


def xywh_to_xyxy(box: BBoxXYWH) -> BBoxXYXY:
    x, y, w, h = box
    return x, y, x + w, y + h


def xyxy_to_xywh(box: Sequence[float]) -> BBoxXYWH:
    if len(box) != 4:
        raise ValueError(f"Expected 4 bbox values, got {len(box)}")
    x1, y1, x2, y2 = map(float, box)
    return bbox_xywh((x1, y1, x2 - x1, y2 - y1))


def area_xywh(box: BBoxXYWH) -> float:
    return box[2] * box[3]


def yolo_xywhn(box: BBoxXYWH, image_width: int, image_height: int) -> BBoxXYWH:
    x, y, w, h = box
    return (
        (x + w / 2.0) / image_width,
        (y + h / 2.0) / image_height,
        w / image_width,
        h / image_height,
    )


def clip_xywh(box: BBoxXYWH, image_width: int, image_height: int) -> BBoxXYWH | None:
    x1, y1, x2, y2 = xywh_to_xyxy(box)
    x1 = max(0.0, min(x1, float(image_width)))
    y1 = max(0.0, min(y1, float(image_height)))
    x2 = max(0.0, min(x2, float(image_width)))
    y2 = max(0.0, min(y2, float(image_height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return xyxy_to_xywh((x1, y1, x2, y2))
