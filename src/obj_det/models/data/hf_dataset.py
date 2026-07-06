from __future__ import annotations

from torch.utils.data import Dataset as TorchDataset

from obj_det.models.data.loader import seed_worker_transform
from obj_det.models.data.sample_source import DetectionSampleSource


class HFTrainerDetectionDataset(TorchDataset):
    def __init__(self, source: DetectionSampleSource, transform):
        self.source = source
        self.transform = transform

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, idx: int):
        seed_worker_transform(self.transform)
        return self.transform(self.source[idx])
