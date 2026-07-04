from __future__ import annotations

from torch.utils.data import Dataset as TorchDataset

from obj_det.models.data.hf_targets import make_hf_detection_item
from obj_det.models.data.loader import seed_worker_transform
from obj_det.models.data.sample_source import DetectionSampleSource


class HFTrainerDetectionDataset(TorchDataset):
    def __init__(self, source: DetectionSampleSource, transform, processor, processor_kwargs=None):
        self.source = source
        self.transform = transform
        self.processor = processor
        self.processor_kwargs = processor_kwargs or {}

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, idx: int):
        seed_worker_transform(self.transform)
        sample = self.transform(self.source[idx])
        return make_hf_detection_item(
            processor=self.processor,
            sample=sample,
            image_id=idx,
            processor_kwargs=self.processor_kwargs,
        )
