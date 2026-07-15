from __future__ import annotations

import inspect
from typing import Any, Mapping

import numpy as np

from obj_det.models.data.bbox import area_xywh
from obj_det.models.data.sample import DetectionSample
from obj_det.models.schemas.config import PreprocessConfig


def validate_hf_processor_size(processor, preprocess: PreprocessConfig) -> None:
    size = processor.size

    def value(name: str) -> int | None:
        raw = size.get(name) if isinstance(size, Mapping) else getattr(size, name, None)
        return int(raw) if raw is not None else None

    if preprocess.resize_mode == "shortest_edge":
        actual = (value("shortest_edge"), value("longest_edge"))
        expected = (preprocess.shortest_edge, preprocess.longest_edge)
    else:
        actual = (value("height"), value("width"))
        expected = (preprocess.height, preprocess.width)

    if actual != expected:
        raise ValueError(
            f"Configured preprocessing {expected} does not match processor size {actual}"
        )


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


def hf_processor_kwargs(preprocess: PreprocessConfig) -> dict[str, bool]:
    kwargs = {"do_resize": False}
    if preprocess.resize_mode == "shortest_edge":
        kwargs["do_pad"] = True
    return kwargs


def post_process_hf_detections(
    processor,
    outputs,
    *,
    threshold: float,
    target_sizes,
    max_detections: int,
):
    method = processor.post_process_object_detection
    kwargs = {"threshold": threshold, "target_sizes": target_sizes}
    if "top_k" in inspect.signature(method).parameters:
        kwargs["top_k"] = max_detections
    return method(outputs, **kwargs)


def make_hf_detection_collate(processor, preprocess: PreprocessConfig):
    kwargs = hf_processor_kwargs(preprocess)

    def collate(samples: list[DetectionSample]) -> dict[str, Any]:
        for sample in samples:
            if sample.image is None:
                raise ValueError("HF Trainer collator requires decoded image data")

        encoded = processor(
            images=[np.ascontiguousarray(sample.image) for sample in samples],
            annotations=[
                sample_to_coco_annotation(sample, image_id=image_idx)
                for image_idx, sample in enumerate(samples)
            ],
            return_tensors="pt",
            **kwargs,
        )
        if preprocess.resize_mode == "shortest_edge" and "pixel_mask" not in encoded:
            raise ValueError("Variable-size HF batches require a pixel_mask")
        return dict(encoded)

    return collate
