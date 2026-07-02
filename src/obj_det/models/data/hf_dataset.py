from __future__ import annotations

from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset

from obj_det.models.data.hf_targets import make_hf_detection_item
from obj_det.models.data.row_parser import HFDetectionRowParser


class HFTrainerDetectionDataset(TorchDataset):
    def __init__(self, ds: Dataset, parser: HFDetectionRowParser, transform, processor, processor_kwargs=None):
        self.ds = ds
        self.parser = parser
        self.transform = transform
        self.processor = processor
        self.processor_kwargs = processor_kwargs or {}

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        sample = self.transform(self.parser.parse(self.ds[idx]))
        return make_hf_detection_item(
            processor=self.processor,
            sample=sample,
            image_id=idx,
            processor_kwargs=self.processor_kwargs,
        )
