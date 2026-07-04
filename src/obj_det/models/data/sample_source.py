from __future__ import annotations

from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset

from obj_det.models.data.row_parser import HFDetectionRowParser
from obj_det.models.data.sample import DetectionSample


class DetectionSampleSource(TorchDataset):
    def __init__(self, ds: Dataset, parser: HFDetectionRowParser, *, predecode_images: bool = False):
        self.ds = ds
        self.parser = parser
        self.samples = [parser.parse(row) for row in ds] if predecode_images else None

    def __len__(self) -> int:
        return len(self.samples) if self.samples is not None else len(self.ds)

    def __getitem__(self, idx: int) -> DetectionSample:
        if self.samples is not None:
            return self.samples[idx]
        return self.parser.parse(self.ds[idx])
