from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

import yaml
from datasets import Dataset, DatasetDict, Features, Image, List, Value

from obj_det.datasets.models import ImageRecord
from obj_det.datasets.models.source_config import SourceDatasetConfig
from obj_det.datasets.sources import source_from_config


logger = logging.getLogger(__name__)


HF_FEATURES = Features(
    {
        "image": Image(decode=False),
        "image_id": Value("string"),
        "dataset": Value("string"),
        "split": Value("string"),
        "image_path": Value("string"),
        "width": Value("int32"),
        "height": Value("int32"),
        "condition": Value("string"),
        "domain": Value("string"),
        "is_synthetic": Value("bool"),
        "objects": List(
            {
                "bbox": List(Value("float32"), length=4),
                "native_label": Value("string"),
                "native_label_id": Value("string"),
                "meta_label": Value("string"),
                "ignore": Value("bool"),
                "iscrowd": Value("bool"),
                "meta_json": Value("string"),
            }
        ),
        "meta_json": Value("string"),
    }
)


def convert_config_to_dataset_dict(
    config_path: Path,
    *,
    splits: Iterable[str] | None = None,
) -> DatasetDict:
    data = yaml.safe_load(config_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected dataset config mapping: {config_path}")

    cfg = SourceDatasetConfig.model_validate(data)
    source = source_from_config(cfg)
    split_names = list(splits) if splits is not None else source.splits
    output_splits: dict[str, str] = {}

    for split in split_names:
        source.require_split(split)
        output_split = source.split_cfg(split).output_split or split
        if output_split in output_splits:
            raise ValueError(
                f"Duplicate output split {output_split!r} for source splits "
                f"{output_splits[output_split]!r} and {split!r}"
            )
        output_splits[output_split] = split

    datasets = DatasetDict()
    for output_split, split in output_splits.items():

        def rows(split: str = split):
            for record in source.iter_records(split):
                yield _record_to_row(record)

        try:
            datasets[output_split] = Dataset.from_generator(
                rows,
                features=HF_FEATURES,
                split=output_split,
            )
        except ValueError as exc:
            if "corresponds to no data" not in str(exc):
                raise
            datasets[output_split] = Dataset.from_dict(
                {column: [] for column in HF_FEATURES},
                features=HF_FEATURES,
            )

        logger.info(
            "Dataset import stats: dataset=%s source_split=%s output_split=%s stats=%s",
            source.key,
            split,
            output_split,
            source.import_stats_snapshot(split),
        )

    return datasets


def _record_to_row(record: ImageRecord) -> dict[str, Any]:
    return {
        "image": {
            "bytes": record.image_path.read_bytes(),
            "path": str(record.image_path),
        },
        "image_id": record.image_id,
        "dataset": record.dataset,
        "split": record.split,
        "image_path": str(record.image_path),
        "width": record.width,
        "height": record.height,
        "condition": record.condition,
        "domain": record.domain,
        "is_synthetic": record.is_synthetic,
        "objects": [
            {
                "bbox": list(obj.bbox.xywh()),
                "native_label": obj.native_label,
                "native_label_id": None
                if obj.native_label_id is None
                else str(obj.native_label_id),
                "meta_label": obj.meta_label,
                "ignore": obj.ignore,
                "iscrowd": obj.iscrowd,
                "meta_json": json.dumps(obj.meta, default=str, sort_keys=True),
            }
            for obj in record.objects
        ],
        "meta_json": json.dumps(record.meta, default=str, sort_keys=True),
    }
