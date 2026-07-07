from __future__ import annotations

import unittest
from pathlib import Path

from datasets import Dataset

from obj_det.datasets.models import BBox
from obj_det.models.adapters.torchvision import _TorchvisionTrainerDataset, _torchvision_collate
from obj_det.models.data.hf_dataset import HFTrainerDetectionDataset
from obj_det.models.data.sample import DetectionBatch
from obj_det.models.data.loader import dataloader_kwargs
from obj_det.models.data.profiling import measure_dataloader, measure_decode_backend, measure_transform
from obj_det.models.data.hf_targets import make_hf_detection_collate, sample_to_coco_annotation
from obj_det.models.data.row_batches import iter_hf_row_batches
from obj_det.models.data.row_parser import HFDetectionRowParser
from obj_det.models.data.sample_source import DetectionSampleSource
from obj_det.models.data.transforms import DetectionTransform
from obj_det.models.data.ultralytics_dataset import HFUltralyticsDetectionDataset, ultralytics_detection_collate
from obj_det.models.schemas import DataLoaderConfig, PreprocessConfig

from .helpers import row


class BackendDataTest(unittest.TestCase):
    def test_hf_coco_annotation_uses_contiguous_label_ids(self):
        sample = HFDetectionRowParser(["car"], "meta").parse(row())
        ann = sample_to_coco_annotation(sample, image_id=7)

        self.assertEqual(ann["image_id"], 7)
        self.assertEqual(ann["annotations"][0]["category_id"], 0)
        self.assertEqual(ann["annotations"][0]["bbox"], [4.0, 5.0, 8.0, 10.0])

    def test_runtime_targets_do_not_use_pydantic_bbox(self):
        sample = HFDetectionRowParser(["car"], "meta").parse(row())

        self.assertIsInstance(sample.targets[0].bbox_xywh, tuple)
        self.assertNotIsInstance(sample.targets[0].bbox_xywh, BBox)
        self.assertFalse(hasattr(sample.targets[0], "bbox"))

    def test_ultralytics_dataset_and_collate(self):
        ds = Dataset.from_list([row(), row(image_id="img2")])
        parser = HFDetectionRowParser(["car"], "meta")
        source = DetectionSampleSource(ds, parser)
        transform = DetectionTransform(PreprocessConfig(image_size=64))
        dataset = HFUltralyticsDetectionDataset(source, transform)
        item = dataset[0]
        batch = ultralytics_detection_collate([dataset[0], dataset[1]])

        self.assertEqual(tuple(item["img"].shape), (3, 64, 64))
        self.assertEqual(tuple(item["cls"].shape), (1, 1))
        self.assertEqual(tuple(item["bboxes"].shape), (1, 4))
        self.assertEqual(tuple(batch["img"].shape), (2, 3, 64, 64))
        self.assertEqual(batch["batch_idx"].tolist(), [0.0, 1.0])
        self.assertNotIn("sample", item)
        self.assertNotIn("samples", batch)

    def test_ultralytics_mixed_empty_and_non_empty_batch(self):
        ds = Dataset.from_list([row(objects=[]), row(image_id="img2")])
        parser = HFDetectionRowParser(["car"], "meta")
        source = DetectionSampleSource(ds, parser)
        transform = DetectionTransform(PreprocessConfig(image_size=64))
        dataset = HFUltralyticsDetectionDataset(source, transform)
        empty_item = dataset[0]
        batch = ultralytics_detection_collate([empty_item, dataset[1]])

        self.assertEqual(tuple(empty_item["cls"].shape), (0, 1))
        self.assertEqual(tuple(empty_item["bboxes"].shape), (0, 4))
        self.assertEqual(tuple(batch["cls"].shape), (1, 1))
        self.assertEqual(tuple(batch["bboxes"].shape), (1, 4))
        self.assertEqual(batch["batch_idx"].tolist(), [1.0])

    def test_ultralytics_profiling_is_disabled_by_default(self):
        ds = Dataset.from_list([row()])
        parser = HFDetectionRowParser(["car"], "meta")
        source = DetectionSampleSource(ds, parser)
        transform = DetectionTransform(PreprocessConfig(image_size=64))
        dataset = HFUltralyticsDetectionDataset(source, transform)

        self.assertIsNone(DataLoaderConfig().profile_every_n)
        self.assertIsNone(dataset.profile_every_n)

    def test_ultralytics_dataset_can_include_samples_for_debug(self):
        ds = Dataset.from_list([row(), row(image_id="img2")])
        parser = HFDetectionRowParser(["car"], "meta")
        source = DetectionSampleSource(ds, parser)
        transform = DetectionTransform(PreprocessConfig(image_size=64))
        dataset = HFUltralyticsDetectionDataset(source, transform, include_samples=True)
        batch = ultralytics_detection_collate([dataset[0], dataset[1]])

        self.assertIn("sample", dataset[0])
        self.assertEqual(len(batch["samples"]), 2)

    def test_detection_batch_does_not_require_samples(self):
        batch = DetectionBatch(images=[], targets=[])

        self.assertIsNone(batch.samples)

    def test_sample_source_can_predecode_images(self):
        ds = Dataset.from_list([row(), row(image_id="img2")])
        parser = HFDetectionRowParser(["car"], "meta")
        source = DetectionSampleSource(ds, parser, predecode_images=True)

        self.assertEqual(len(source), 2)
        self.assertIsNotNone(source.samples)
        self.assertEqual(source[0].image_id, "img1")
        self.assertEqual(source[1].targets[0].label, "car")

    def test_hf_trainer_collator_batches_processor_calls(self):
        class FakeProcessor:
            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                batch_size = len(kwargs["images"])
                return {
                    "pixel_values": torch.zeros((batch_size, 3, 64, 64)),
                    "labels": kwargs["annotations"],
                }

        import torch

        ds = Dataset.from_list([row(), row(image_id="img2")])
        parser = HFDetectionRowParser(["car"], "meta")
        source = DetectionSampleSource(ds, parser)
        transform = DetectionTransform(PreprocessConfig(image_size=64))
        dataset = HFTrainerDetectionDataset(source, transform)
        processor = FakeProcessor()
        collate = make_hf_detection_collate(processor)
        batch = collate([dataset[0], dataset[1]])

        self.assertEqual(len(processor.calls), 1)
        self.assertEqual(len(processor.calls[0]["images"]), 2)
        self.assertEqual(tuple(batch["pixel_values"].shape), (2, 3, 64, 64))

    def test_dataloader_kwargs_only_pass_worker_options_when_enabled(self):
        self.assertEqual(
            dataloader_kwargs(DataLoaderConfig(num_workers=0, persistent_workers=True, prefetch_factor=2)),
            {"num_workers": 0, "pin_memory": True},
        )
        self.assertEqual(
            dataloader_kwargs(DataLoaderConfig(num_workers=2, persistent_workers=True, prefetch_factor=2)),
            {"num_workers": 2, "pin_memory": True, "persistent_workers": True, "prefetch_factor": 2},
        )

    def test_data_profiling_helpers_return_rates(self):
        loader_stats = measure_dataloader([{"x": 1}, {"x": 2}], max_batches=2)
        decode_stats = measure_decode_backend(
            [row(), row(image_id="img2")],
            classes=["car"],
            label_mode="meta",
            decode_backend="pil",
            max_images=2,
        )

        self.assertEqual(loader_stats["batches"], 2.0)
        self.assertGreater(loader_stats["batches_per_second"], 0.0)
        opencv_decode_stats = measure_decode_backend(
            [row(), row(image_id="img2")],
            classes=["car"],
            label_mode="meta",
            decode_backend="opencv",
            max_images=2,
        )

        self.assertEqual(decode_stats["images"], 2.0)
        self.assertGreater(decode_stats["images_per_second"], 0.0)
        self.assertEqual(opencv_decode_stats["images"], 2.0)
        self.assertGreater(opencv_decode_stats["images_per_second"], 0.0)

    def test_hf_row_batches_use_indexing_not_whole_iteration(self):
        rows = [row(), row(image_id="img2"), row(image_id="img3")]

        class NoIterDataset:
            def __len__(self):
                return len(rows)

            def __getitem__(self, idx):
                return rows[idx]

            def __iter__(self):
                raise AssertionError("Do not iterate whole dataset")

        batches = list(iter_hf_row_batches(NoIterDataset(), batch_size=2))

        self.assertEqual([[item["image_id"] for item in batch] for batch in batches], [["img1", "img2"], ["img3"]])

    def test_transform_profiling_helper_returns_rate(self):
        parser = HFDetectionRowParser(["car"], "meta")
        transform = DetectionTransform(PreprocessConfig(image_size=64))
        samples = [parser.parse(row()), parser.parse(row(image_id="img2"))]

        stats = measure_transform(samples, transform, max_samples=2)

        self.assertEqual(stats["samples"], 2.0)
        self.assertGreater(stats["samples_per_second"], 0.0)

    def test_torchvision_dataset_and_collate_are_trainer_inputs(self):
        ds = Dataset.from_list([row(), row(image_id="img2")])
        parser = HFDetectionRowParser(["car"], "meta")
        source = DetectionSampleSource(ds, parser)
        transform = DetectionTransform(PreprocessConfig(image_size=64))
        dataset = _TorchvisionTrainerDataset(source, transform)
        item = dataset[0]
        batch = _torchvision_collate([dataset[0], dataset[1]])

        self.assertEqual(tuple(item["image"].shape), (3, 64, 64))
        self.assertEqual(tuple(item["target"]["boxes"].shape), (1, 4))
        self.assertEqual(len(batch["images"]), 2)
        self.assertEqual(len(batch["targets"]), 2)

    def test_torchvision_adapter_has_no_custom_training_loop(self):
        source = Path("src/obj_det/models/adapters/torchvision.py").read_text()

        self.assertNotIn("loss.backward", source)
        self.assertNotIn("optimizer.step", source)


if __name__ == "__main__":
    unittest.main()
