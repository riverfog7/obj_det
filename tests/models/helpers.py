from __future__ import annotations

import io
import json

from PIL import Image


def image_bytes(size=(32, 24), color=(128, 128, 128)) -> bytes:
    image = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def row(
    *,
    image_id="img1",
    dataset="tiny",
    split="val",
    size=(32, 24),
    objects=None,
    condition="clear",
    domain="general",
):
    objects = objects if objects is not None else [
        {
            "bbox": [4.0, 5.0, 8.0, 10.0],
            "native_label": "car",
            "native_label_id": "1",
            "meta_label": "car",
            "ignore": False,
            "iscrowd": False,
            "meta_json": json.dumps({"source": "unit"}),
        }
    ]
    return {
        "image": {"bytes": image_bytes(size), "path": None},
        "image_id": image_id,
        "dataset": dataset,
        "split": split,
        "image_path": "unused.png",
        "width": size[0],
        "height": size[1],
        "condition": condition,
        "domain": domain,
        "is_synthetic": False,
        "objects": objects,
        "meta_json": json.dumps({"row": image_id}),
    }
