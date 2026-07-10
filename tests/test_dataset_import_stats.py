from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from obj_det.datasets.converters.hf import convert_config_to_dataset_dict
from obj_det.datasets.models import ImageRecord
from obj_det.datasets.models.source_config import SourceDatasetConfig
from obj_det.datasets.sources.base import BaseSourceDataset


class _StatsSourceDataset(BaseSourceDataset):
    def _iter_records(self, split: str):
        image_path = self.cfg.root / "image.bin"

        valid = self.make_object(
            bbox_xywh=(1, 1, 4, 4),
            image_width=16,
            image_height=16,
            native_label="car",
        )
        yield self._record(split, "valid", image_path, [valid] if valid is not None else [])

        yield self._record(split, "empty", image_path, [])

        invalid = self.make_object(
            bbox_xywh=(1, 1, 0, 4),
            image_width=16,
            image_height=16,
            native_label="car",
        )
        yield self._record(split, "invalid", image_path, [invalid] if invalid is not None else [])

        unmapped = self.make_object(
            bbox_xywh=(2, 2, 4, 4),
            image_width=16,
            image_height=16,
            native_label="dog",
        )
        yield self._record(split, "unmapped", image_path, [unmapped] if unmapped is not None else [])

        ignored = self.make_object(
            bbox_xywh=(3, 3, 4, 4),
            image_width=16,
            image_height=16,
            native_label="ignored",
        )
        yield self._record(split, "ignored", image_path, [ignored] if ignored is not None else [])

    def _record(
        self,
        split: str,
        source_id: str,
        image_path: Path,
        objects,
    ) -> ImageRecord:
        return self.make_record(
            split=split,
            source_id=source_id,
            image_path=image_path,
            width=16,
            height=16,
            objects=objects,
        )


class DatasetImportStatsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "image.bin").write_bytes(b"not-decoded-by-conversion")
        self.cfg = SourceDatasetConfig.model_validate(
            {
                "key": "stats",
                "root": str(self.root),
                "source_format": "test",
                "splits": {"train": {"paths": {}}},
                "bbox_policy": "drop",
                "class_map": {"car": "car"},
                "ignore_labels": ["ignored"],
            }
        )
        self.source = _StatsSourceDataset(self.cfg)

    def tearDown(self):
        self.tmp.cleanup()

    def test_source_drops_empty_images_and_counts_central_reasons(self):
        records = list(self.source.iter_records("train"))

        self.assertEqual([record.image_id for record in records], ["stats:train:valid", "stats:train:unmapped"])
        self.assertEqual(
            self.source.import_stats_snapshot("train"),
            {
                "source_objects_seen": 4,
                "source_objects_kept": 2,
                "source_images_seen": 5,
                "source_images_kept": 2,
                "dropped_empty_or_ambiguous_images": 3,
                "dropped_invalid_boxes": 1,
                "dropped_unknown_labels": 0,
                "unmapped_meta_labels": 1,
                "dropped_ignored_labels": 1,
            },
        )

    def test_hf_conversion_logs_aggregate_stats_for_each_split(self):
        config_path = self.root / "dataset.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "key: stats",
                    f"root: {self.root}",
                    "source_format: test",
                    "bbox_policy: drop",
                    "splits:",
                    "  train:",
                    "    paths: {}",
                    "class_map:",
                    "  car: car",
                    "ignore_labels: [ignored]",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        with (
            patch("obj_det.datasets.converters.hf.source_from_config", return_value=self.source),
            patch("datasets.config.HF_DATASETS_CACHE", str(self.root / "hf-cache")),
            self.assertLogs("obj_det.datasets.converters.hf", level="INFO") as logs,
        ):
            dataset = convert_config_to_dataset_dict(config_path)

        self.assertEqual(len(dataset["train"]), 2)
        self.assertTrue(any("source_split=train" in message for message in logs.output))
        self.assertTrue(any("dropped_empty_or_ambiguous_images" in message for message in logs.output))
        self.assertTrue(any("dropped_invalid_boxes" in message for message in logs.output))


if __name__ == "__main__":
    unittest.main()
