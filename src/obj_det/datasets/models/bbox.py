from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from .base import SchemaModel


class BBox(SchemaModel):
    """
    Canonical bounding box.

    Format:
        x, y, w, h

    Coordinates:
        absolute pixel coordinates

    Meaning:
        x, y = top-left corner
        w, h = width and height
    """

    FORMAT: ClassVar[str] = "xywh_abs"

    x: float = Field(ge=0)
    y: float = Field(ge=0)
    w: float = Field(gt=0)
    h: float = Field(gt=0)

    @classmethod
    def from_xywh(cls, xywh: tuple[float, float, float, float] | list[float]) -> "BBox":
        if len(xywh) != 4:
            raise ValueError(f"Expected 4 bbox values, got {len(xywh)}")

        return cls(
            x=float(xywh[0]),
            y=float(xywh[1]),
            w=float(xywh[2]),
            h=float(xywh[3]),
        )

    @classmethod
    def from_xyxy(cls, xyxy: tuple[float, float, float, float] | list[float]) -> "BBox":
        if len(xyxy) != 4:
            raise ValueError(f"Expected 4 bbox values, got {len(xyxy)}")

        x1, y1, x2, y2 = map(float, xyxy)

        return cls(
            x=x1,
            y=y1,
            w=x2 - x1,
            h=y2 - y1,
        )

    @property
    def area(self) -> float:
        return self.w * self.h

    def xywh(self) -> tuple[float, float, float, float]:
        return self.x, self.y, self.w, self.h

    def xyxy(self) -> tuple[float, float, float, float]:
        return self.x, self.y, self.x + self.w, self.y + self.h

    def cxcywh(self) -> tuple[float, float, float, float]:
        return self.x + self.w / 2.0, self.y + self.h / 2.0, self.w, self.h

    def yolo_xywhn(self, image_width: int, image_height: int) -> tuple[float, float, float, float]:
        """
        YOLO normalized format:
            center_x, center_y, width, height
        """
        cx, cy, w, h = self.cxcywh()

        return (
            cx / image_width,
            cy / image_height,
            w / image_width,
            h / image_height,
        )

    def within_image(self, image_width: int, image_height: int, eps: float = 1e-6) -> bool:
        x1, y1, x2, y2 = self.xyxy()

        return (
            x1 >= -eps
            and y1 >= -eps
            and x2 <= image_width + eps
            and y2 <= image_height + eps
        )

    def clipped(self, image_width: int, image_height: int) -> "BBox | None":
        """
        Return clipped bbox.

        Returns None if the clipped box has no valid area.
        """
        x1, y1, x2, y2 = self.xyxy()

        x1 = max(0.0, min(x1, float(image_width)))
        y1 = max(0.0, min(y1, float(image_height)))
        x2 = max(0.0, min(x2, float(image_width)))
        y2 = max(0.0, min(y2, float(image_height)))

        if x2 <= x1 or y2 <= y1:
            return None

        return BBox.from_xyxy((x1, y1, x2, y2))
