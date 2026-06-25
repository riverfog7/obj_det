from __future__ import annotations

from obj_det.datasets.models.source_config import SourceDatasetConfig

from .base import BaseSourceDataset
from .coco import CocoSourceDataset
from .visdrone import VisDroneDetSourceDataset


SOURCE_FORMATS: dict[str, type[BaseSourceDataset]] = {
    "coco": CocoSourceDataset,
    "visdrone_det": VisDroneDetSourceDataset,
}


def source_from_config(cfg: SourceDatasetConfig) -> BaseSourceDataset:
    source_format = cfg.source_format

    if source_format not in SOURCE_FORMATS:
        raise ValueError(
            f"Unsupported source_format={source_format!r} for dataset={cfg.key!r}. "
            f"Supported source formats: {sorted(SOURCE_FORMATS)}"
        )

    return SOURCE_FORMATS[source_format](cfg)
