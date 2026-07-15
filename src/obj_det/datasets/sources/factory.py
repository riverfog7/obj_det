from __future__ import annotations

from obj_det.datasets.models.source_config import SourceDatasetConfig

from .base import BaseSourceDataset
from .bdd100k import Bdd100kSourceDataset
from .coco import CocoSourceDataset
from .exdark import ExDarkSourceDataset
from .visdrone import VisDroneDetSourceDataset
from .voc import PascalVocSourceDataset
from .yolo import YoloSourceDataset
from .yolo_noyml import YoloNoYamlSourceDataset


SOURCE_FORMATS: dict[str, type[BaseSourceDataset]] = {
    "bdd100k": Bdd100kSourceDataset,
    "coco": CocoSourceDataset,
    "exdark": ExDarkSourceDataset,
    "pascal_voc": PascalVocSourceDataset,
    "visdrone_det": VisDroneDetSourceDataset,
    "yolo": YoloSourceDataset,
    "yolo_noyml": YoloNoYamlSourceDataset,
}


def source_from_config(cfg: SourceDatasetConfig) -> BaseSourceDataset:
    source_format = cfg.source_format

    if source_format not in SOURCE_FORMATS:
        raise ValueError(
            f"Unsupported source_format={source_format!r} for dataset={cfg.key!r}. "
            f"Supported source formats: {sorted(SOURCE_FORMATS)}"
        )

    return SOURCE_FORMATS[source_format](cfg)
