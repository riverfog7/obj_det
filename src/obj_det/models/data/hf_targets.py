from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from obj_det.models.data.bbox import area_xywh
from obj_det.models.data.sample import DetectionSample


@dataclass
class HFProcessedDetectionItem:
    pixel_values: torch.Tensor
    labels: dict[str, Any]
    pixel_mask: torch.Tensor | None = None


def sample_to_coco_annotation(sample: DetectionSample, *, image_id: int) -> dict[str, Any]:
    annotations = []
    for idx, target in enumerate(sample.targets):
        annotations.append(
            {
                "id": idx,
                "image_id": image_id,
                "category_id": target.label_id,
                "bbox": list(target.bbox_xywh),
                "area": area_xywh(target.bbox_xywh),
                "iscrowd": int(target.iscrowd),
            }
        )
    return {"image_id": image_id, "annotations": annotations}


def make_hf_detection_item(
    *,
    processor,
    sample: DetectionSample,
    image_id: int,
    processor_kwargs: dict[str, Any] | None = None,
) -> HFProcessedDetectionItem:
    kwargs = {"do_resize": False}
    if processor_kwargs:
        kwargs.update(processor_kwargs)

    encoded = processor(
        images=np.ascontiguousarray(sample.image),
        annotations=sample_to_coco_annotation(sample, image_id=image_id),
        return_tensors="pt",
        **kwargs,
    )
    pixel_values = encoded["pixel_values"].squeeze(0)
    pixel_mask = encoded.get("pixel_mask")
    if pixel_mask is not None:
        pixel_mask = pixel_mask.squeeze(0)
    labels = encoded["labels"][0]
    return HFProcessedDetectionItem(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)


def hf_detection_collate(items: list[HFProcessedDetectionItem]) -> dict[str, Any]:
    batch: dict[str, Any] = {
        "pixel_values": torch.stack([item.pixel_values for item in items]),
        "labels": [item.labels for item in items],
    }
    if items and items[0].pixel_mask is not None:
        batch["pixel_mask"] = torch.stack([item.pixel_mask for item in items if item.pixel_mask is not None])
    return batch
