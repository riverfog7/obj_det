from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from obj_det.models.data.bbox import yolo_xywhn
from obj_det.models.data.loader import seed_worker_transform
from obj_det.models.data.sample_source import DetectionSampleSource


logger = logging.getLogger(__name__)


class HFUltralyticsDetectionDataset(TorchDataset):
    def __init__(
        self,
        source: DetectionSampleSource,
        transform,
        *,
        include_samples: bool = False,
        profile_every_n: int | None = None,
    ):
        self.source = source
        self.transform = transform
        self.include_samples = include_samples
        self.profile_every_n = profile_every_n

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        t0 = time.perf_counter()
        seed_worker_transform(self.transform)
        sample = self.source[idx]
        t1 = time.perf_counter()
        sample = self.transform(sample)
        t2 = time.perf_counter()
        if sample.image is None:
            raise ValueError("Ultralytics dataset requires decoded image data")
        image = torch.from_numpy(np.ascontiguousarray(sample.image)).permute(2, 0, 1).contiguous()
        cls = torch.tensor([[target.label_id] for target in sample.targets], dtype=torch.float32).reshape(-1, 1)
        bboxes = torch.tensor(
            [yolo_xywhn(target.bbox_xywh, sample.width, sample.height) for target in sample.targets],
            dtype=torch.float32,
        ).reshape(-1, 4)
        item = {
            "img": image,
            "cls": cls,
            "bboxes": bboxes,
            "batch_idx": torch.zeros((len(sample.targets),), dtype=torch.float32),
            "im_file": sample.image_id,
            "ori_shape": (sample.height, sample.width),
            "resized_shape": (sample.height, sample.width),
        }
        if self.include_samples:
            item["sample"] = sample
        self._profile(idx, t0, t1, t2)
        return item

    def _profile(self, idx: int, t0: float, t1: float, t2: float) -> None:
        if not self.profile_every_n or idx % self.profile_every_n != 0:
            return
        t3 = time.perf_counter()
        logger.info(
            "YOLO dataset item profile idx=%s source_parse=%.6fs transform=%.6fs tensorize=%.6fs total=%.6fs",
            idx,
            t1 - t0,
            t2 - t1,
            t3 - t2,
            t3 - t0,
        )


def ultralytics_detection_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([item["img"] for item in batch], dim=0)
    cls = []
    bboxes = []
    batch_idx = []
    for idx, item in enumerate(batch):
        cls.append(item["cls"])
        bboxes.append(item["bboxes"])
        batch_idx.append(torch.full((item["cls"].shape[0],), idx, dtype=torch.float32))

    collated = {
        "img": images,
        "cls": torch.cat(cls, dim=0) if cls else torch.zeros((0, 1), dtype=torch.float32),
        "bboxes": torch.cat(bboxes, dim=0) if bboxes else torch.zeros((0, 4), dtype=torch.float32),
        "batch_idx": torch.cat(batch_idx, dim=0) if batch_idx else torch.zeros((0,), dtype=torch.float32),
        "im_file": [item["im_file"] for item in batch],
        "ori_shape": [item["ori_shape"] for item in batch],
        "resized_shape": [item["resized_shape"] for item in batch],
    }
    if batch and "sample" in batch[0]:
        collated["samples"] = [item["sample"] for item in batch]
    return collated
