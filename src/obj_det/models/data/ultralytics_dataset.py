from __future__ import annotations

from typing import Any

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset

from obj_det.models.data.row_parser import HFDetectionRowParser
from obj_det.models.data.sample import DetectionSample


class HFUltralyticsDetectionDataset(TorchDataset):
    def __init__(self, ds: Dataset, parser: HFDetectionRowParser, transform):
        self.ds = ds
        self.parser = parser
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.transform(self.parser.parse(self.ds[idx]))
        image = torch.from_numpy(np.array(sample.image, copy=True)).permute(2, 0, 1).contiguous()
        cls = torch.tensor([[target.label_id] for target in sample.targets], dtype=torch.float32)
        bboxes = torch.tensor(
            [target.bbox.yolo_xywhn(sample.width, sample.height) for target in sample.targets],
            dtype=torch.float32,
        ).reshape(-1, 4)
        return {
            "img": image,
            "cls": cls,
            "bboxes": bboxes,
            "batch_idx": torch.zeros((len(sample.targets),), dtype=torch.float32),
            "im_file": sample.image_id,
            "ori_shape": (sample.height, sample.width),
            "resized_shape": (sample.height, sample.width),
            "sample": sample,
        }


def ultralytics_detection_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([item["img"] for item in batch], dim=0)
    cls = []
    bboxes = []
    batch_idx = []
    for idx, item in enumerate(batch):
        cls.append(item["cls"])
        bboxes.append(item["bboxes"])
        batch_idx.append(torch.full((item["cls"].shape[0],), idx, dtype=torch.float32))

    return {
        "img": images,
        "cls": torch.cat(cls, dim=0) if cls else torch.zeros((0, 1), dtype=torch.float32),
        "bboxes": torch.cat(bboxes, dim=0) if bboxes else torch.zeros((0, 4), dtype=torch.float32),
        "batch_idx": torch.cat(batch_idx, dim=0) if batch_idx else torch.zeros((0,), dtype=torch.float32),
        "im_file": [item["im_file"] for item in batch],
        "ori_shape": [item["ori_shape"] for item in batch],
        "resized_shape": [item["resized_shape"] for item in batch],
        "samples": [item["sample"] for item in batch],
    }
