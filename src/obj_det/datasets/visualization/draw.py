from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from PIL import Image, ImageColor, ImageDraw, ImageFont

from obj_det.datasets.models import ImageRecord, ObjectAnnotation


LabelMode = Literal["native", "meta"]
Color = tuple[int, int, int]


def draw_record(
    record: ImageRecord,
    *,
    label_mode: LabelMode = "native",
    include_ignored: bool = False,
    include_crowd: bool = True,
    show_label: bool = True,
    show_masks: bool = True,
) -> Image.Image:
    """Draw one canonical ImageRecord and return a PIL image."""
    image = Image.open(record.image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for obj in record.objects:
        if not _use_object(
            ignore=obj.ignore,
            iscrowd=obj.iscrowd,
            label=_object_label(obj, label_mode),
            include_ignored=include_ignored,
            include_crowd=include_crowd,
        ):
            continue

        label = _object_label(obj, label_mode)
        color = _label_color(label or obj.native_label)
        if show_masks:
            image = _overlay_segmentation(
                image,
                obj.meta.get("segmentation"),
                width=record.width,
                height=record.height,
                color=color,
            )
            draw = ImageDraw.Draw(image)

        _draw_box(
            draw,
            bbox=obj.bbox.xywh(),
            color=color,
            label=label if show_label else None,
        )

    return image


def draw_hf_row(
    row: Mapping[str, Any],
    *,
    label_mode: LabelMode = "native",
    include_ignored: bool = False,
    include_crowd: bool = True,
    show_label: bool = True,
    show_masks: bool = True,
) -> Image.Image:
    """Draw one converted Hugging Face dataset row and return a PIL image."""
    image = _open_hf_image(row).convert("RGB")
    draw = ImageDraw.Draw(image)
    width = int(row["width"])
    height = int(row["height"])

    for obj in row.get("objects", []):
        meta = _load_meta_json(obj.get("meta_json"))
        label = _hf_object_label(obj, label_mode)

        if not _use_object(
            ignore=bool(obj.get("ignore", False)),
            iscrowd=bool(obj.get("iscrowd", False)),
            label=label,
            include_ignored=include_ignored,
            include_crowd=include_crowd,
        ):
            continue

        color = _label_color(label or str(obj.get("native_label", "object")))
        if show_masks:
            image = _overlay_segmentation(
                image,
                meta.get("segmentation"),
                width=width,
                height=height,
                color=color,
            )
            draw = ImageDraw.Draw(image)

        _draw_box(
            draw,
            bbox=obj["bbox"],
            color=color,
            label=label if show_label else None,
        )

    return image


def make_grid(
    images: Sequence[Image.Image],
    *,
    columns: int = 4,
    padding: int = 8,
    background: str | Color = "white",
) -> Image.Image:
    """Compose PIL images into a simple grid."""
    if columns <= 0:
        raise ValueError("columns must be greater than 0")
    if padding < 0:
        raise ValueError("padding cannot be negative")
    if not images:
        raise ValueError("images cannot be empty")

    rgb_images = [image.convert("RGB") for image in images]
    cell_width = max(image.width for image in rgb_images)
    cell_height = max(image.height for image in rgb_images)
    rows = (len(rgb_images) + columns - 1) // columns

    grid_width = columns * cell_width + (columns + 1) * padding
    grid_height = rows * cell_height + (rows + 1) * padding
    grid = Image.new("RGB", (grid_width, grid_height), color=background)

    for idx, image in enumerate(rgb_images):
        row = idx // columns
        col = idx % columns
        x = padding + col * (cell_width + padding)
        y = padding + row * (cell_height + padding)
        grid.paste(image, (x, y))

    return grid


def _object_label(obj: ObjectAnnotation, mode: LabelMode) -> str | None:
    if mode == "native":
        return obj.native_label
    if mode == "meta":
        return obj.meta_label
    raise ValueError(f"Unknown label_mode: {mode}")


def _hf_object_label(obj: Mapping[str, Any], mode: LabelMode) -> str | None:
    if mode == "native":
        return obj.get("native_label")
    if mode == "meta":
        return obj.get("meta_label")
    raise ValueError(f"Unknown label_mode: {mode}")


def _use_object(
    *,
    ignore: bool,
    iscrowd: bool,
    label: str | None,
    include_ignored: bool,
    include_crowd: bool,
) -> bool:
    if ignore and not include_ignored:
        return False
    if iscrowd and not include_crowd:
        return False
    return label is not None


def _draw_box(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    color: Color,
    label: str | None,
) -> None:
    x, y, w, h = map(float, bbox)
    x1, y1, x2, y2 = x, y, x + w, y + h
    draw.rectangle((x1, y1, x2, y2), outline=color, width=3)

    if not label:
        return

    font = ImageFont.load_default()
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    text_width = right - left
    text_height = bottom - top
    text_x = x1
    text_y = max(0.0, y1 - text_height - 4)

    draw.rectangle(
        (text_x, text_y, text_x + text_width + 6, text_y + text_height + 4),
        fill=color,
    )
    draw.text((text_x + 3, text_y + 2), label, fill="white", font=font)


def _label_color(label: str) -> Color:
    digest = hashlib.md5(label.encode("utf-8")).digest()
    return 64 + digest[0] // 2, 64 + digest[1] // 2, 64 + digest[2] // 2


def _open_hf_image(row: Mapping[str, Any]) -> Image.Image:
    image = row.get("image")

    if isinstance(image, Image.Image):
        return image.copy()

    if isinstance(image, Mapping):
        image_bytes = image.get("bytes")
        if image_bytes is not None:
            return Image.open(BytesIO(image_bytes))

        image_path = image.get("path")
        if image_path:
            return Image.open(image_path)

    image_path = row.get("image_path")
    if image_path:
        return Image.open(Path(image_path))

    raise ValueError("HF row does not contain an image, image path, or image bytes")


def _load_meta_json(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {}
    return {}


def _overlay_segmentation(
    image: Image.Image,
    segmentation: Any,
    *,
    width: int,
    height: int,
    color: Color,
    alpha: int = 80,
) -> Image.Image:
    if segmentation is None:
        return image

    mask = _decode_coco_segmentation(segmentation, width=width, height=height)
    if mask is None:
        return image

    mask_image = Image.fromarray((mask > 0).astype("uint8") * alpha)
    overlay = Image.new("RGBA", image.size, color + (0,))
    color_layer = Image.new("RGBA", image.size, color + (alpha,))
    overlay.paste(color_layer, mask=mask_image)

    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _decode_coco_segmentation(segmentation: Any, *, width: int, height: int):
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        return None

    try:
        if isinstance(segmentation, dict):
            rle = segmentation
            if isinstance(rle.get("counts"), list):
                rle = mask_utils.frPyObjects(rle, height, width)
            return mask_utils.decode(rle)

        if isinstance(segmentation, list):
            if not segmentation:
                return None

            polygons = segmentation
            if all(isinstance(value, (int, float)) for value in segmentation):
                polygons = [segmentation]

            rles = mask_utils.frPyObjects(polygons, height, width)
            decoded = mask_utils.decode(rles)
            if decoded.ndim == 3:
                return decoded.any(axis=2)
            return decoded
    except Exception:
        return None

    return None
