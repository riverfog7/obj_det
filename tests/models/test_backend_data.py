from __future__ import annotations

import unittest
from pathlib import Path

from datasets import Dataset

from obj_det.models.adapters.torchvision import _TorchvisionTrainerDataset, _torchvision_collate
from obj_det.models.data.loader import dataloader_kwargs
from obj_det.models.data.hf_targets import sample_to_coco_annotation
from obj_det.models.data.row_parser import HFDetectionRowParser
from obj_det.models.data.sample_source import DetectionSampleSource
from obj_det.models.data.transforms import DetectionTransform
from obj_det.models.data.ultralytics_dataset import HFUltralyticsDetectionDataset, ultralytics_detection_collate
from obj_det.models.schemas import DataLoaderConfig, TransformConfig

from .helpers import row


class BackendDataTest(unittest.TestCase):
    def test_hf_coco_annotation_uses_contiguous_label_ids(self):
        sample = HFDetectionRowParser(["car"], "meta").parse(row())
        ann = sample_to_coco_annotation(sample, image_id=7)

        self.assertEqual(ann["image_id"], 7)
        self.assertEqual(ann["annotations"][0]["category_id"], 0)
        self.assertEqual(ann["annotations"][0]["bbox"], [4.0, 5.0, 8.0, 10.0])

    def test_ultralytics_dataset_and_collate(self):
        ds = Dataset.from_list([row(), row(image_id="img2")])
        parser = HFDetectionRowParser(["car"], "meta")
        source = DetectionSampleSource(ds, parser)
        transform = DetectionTransform(TransformConfig(image_size=64))
        dataset = HFUltralyticsDetectionDataset(source, transform)
        item = dataset[0]
        batch = ultralytics_detection_collate([dataset[0], dataset[1]])

        self.assertEqual(tuple(item["img"].shape), (3, 64, 64))
        self.assertEqual(tuple(item["cls"].shape), (1, 1))
        self.assertEqual(tuple(item["bboxes"].shape), (1, 4))
        self.assertEqual(tuple(batch["img"].shape), (2, 3, 64, 64))
        self.assertEqual(batch["batch_idx"].tolist(), [0.0, 1.0])

    def test_sample_source_can_predecode_images(self):
        ds = Dataset.from_list([row(), row(image_id="img2")])
        parser = HFDetectionRowParser(["car"], "meta")
        source = DetectionSampleSource(ds, parser, predecode_images=True)

        self.assertEqual(len(source), 2)
        self.assertIsNotNone(source.samples)
        self.assertEqual(source[0].image_id, "img1")
        self.assertEqual(source[1].targets[0].label, "car")

    def test_dataloader_kwargs_only_pass_worker_options_when_enabled(self):
        self.assertEqual(
            dataloader_kwargs(DataLoaderConfig(num_workers=0, persistent_workers=True, prefetch_factor=2)),
            {"num_workers": 0, "pin_memory": True},
        )
        self.assertEqual(
            dataloader_kwargs(DataLoaderConfig(num_workers=2, persistent_workers=True, prefetch_factor=2)),
            {"num_workers": 2, "pin_memory": True, "persistent_workers": True, "prefetch_factor": 2},
        )

    def test_torchvision_dataset_and_collate_are_trainer_inputs(self):
        ds = Dataset.from_list([row(), row(image_id="img2")])
        parser = HFDetectionRowParser(["car"], "meta")
        source = DetectionSampleSource(ds, parser)
        transform = DetectionTransform(TransformConfig(image_size=64))
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
