from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from obj_det.datasets.models import BBox
from obj_det.models.adapters.base import BaseModelAdapter
from obj_det.models.data.row_parser import HFDetectionRowParser
from obj_det.models.data.sample import DetectionSample
from obj_det.models.data.transforms import bbox_to_original, build_detection_transform
from obj_det.models.schemas.artifact import ModelArtifact
from obj_det.models.schemas.config import PredictConfig, TrainConfig
from obj_det.models.schemas.prediction import PredictionObject, PredictionRecord
from obj_det.models.utils.repro import set_seed


class TorchvisionDetectionAdapter(BaseModelAdapter):
    def train(self, train_ds: Dataset, val_ds: Dataset, train_cfg: TrainConfig) -> ModelArtifact:
        set_seed(train_cfg.seed)
        device = self._device(train_cfg.backend_params)
        train_cfg.output_dir.mkdir(parents=True, exist_ok=True)

        model = self._build_model(num_classes=len(train_cfg.classes) + 1).to(device)
        parser = HFDetectionRowParser(classes=train_cfg.classes, label_mode=train_cfg.label_mode)
        transform = build_detection_transform(
            train_cfg.augmentation_policy,
            train_cfg.image_size,
            {**train_cfg.backend_params.get("transform_params", {}), "seed": train_cfg.seed},
        )
        batch_size = train_cfg.per_device_batch_size or train_cfg.effective_batch_size
        loader = DataLoader(
            _TorchvisionDataset(train_ds, parser, transform),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_collate,
        )

        optimizer = self._optimizer(model, train_cfg)
        max_epochs = train_cfg.max_epochs or 1
        max_steps = train_cfg.max_steps or math.inf
        step = 0
        last_loss = None
        model.train()

        for _epoch in range(max_epochs):
            for images, targets, _samples in loader:
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().cpu())
                step += 1
                if step >= max_steps:
                    break
            if step >= max_steps:
                break

        checkpoint_path = train_cfg.output_dir / "checkpoint.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": train_cfg.classes,
                "label_mode": train_cfg.label_mode,
                "model_name_or_path": str(self.cfg.model_name_or_path),
            },
            checkpoint_path,
        )

        return ModelArtifact(
            model_key=self.key,
            backend=self.backend,
            run_key=train_cfg.run_key,
            classes=train_cfg.classes,
            label_mode=train_cfg.label_mode,
            artifact_path=train_cfg.output_dir,
            checkpoint_path=checkpoint_path,
            best_metric_name="train_loss" if last_loss is not None else None,
            best_metric_value=last_loss,
        )

    def predict(
        self,
        ds: Dataset,
        artifact: ModelArtifact,
        predict_cfg: PredictConfig,
    ) -> Iterator[PredictionRecord]:
        if artifact.checkpoint_path is None:
            raise ValueError("Torchvision artifact is missing checkpoint_path")

        device = self._device(predict_cfg.backend_params)
        model = self._build_model(num_classes=len(predict_cfg.classes) + 1).to(device)
        checkpoint = torch.load(artifact.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        parser = HFDetectionRowParser(classes=predict_cfg.classes, label_mode=predict_cfg.label_mode)
        transform = build_detection_transform(
            predict_cfg.augmentation_policy,
            predict_cfg.image_size,
            predict_cfg.backend_params.get("transform_params", {}),
        )

        with torch.no_grad():
            for row in ds:
                original = parser.parse(row)
                sample = transform(original)
                image = _image_tensor(sample).to(device)
                output = model([image])[0]
                predictions = []
                preprocess = sample.meta.get("preprocess")

                for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
                    score_value = float(score.detach().cpu())
                    if score_value < predict_cfg.conf_threshold:
                        continue
                    raw_label = int(label.detach().cpu())
                    class_idx = raw_label - 1
                    if class_idx < 0 or class_idx >= len(predict_cfg.classes):
                        continue
                    bbox = BBox.from_xyxy([float(v) for v in box.detach().cpu().tolist()])
                    if preprocess is not None:
                        bbox = bbox_to_original(bbox, preprocess)
                    if bbox is None:
                        continue
                    predictions.append(
                        PredictionObject(
                            bbox=bbox,
                            label=predict_cfg.classes[class_idx],
                            label_id=class_idx,
                            score=score_value,
                        )
                    )

                yield PredictionRecord(
                    image_id=original.image_id,
                    dataset=original.dataset,
                    split=original.split,
                    model_key=self.key,
                    width=original.width,
                    height=original.height,
                    predictions=predictions,
                )

    def _build_model(self, *, num_classes: int):
        model_name = str(self.cfg.model_name_or_path)
        if model_name != "fasterrcnn_resnet50_fpn":
            raise ValueError("Torchvision backend currently supports only fasterrcnn_resnet50_fpn")

        weights_param = self.cfg.params.get("weights") or self.cfg.weights
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if str(weights_param).lower() == "default" else None
        min_size = int(self.cfg.params.get("min_size", 800))
        max_size = int(self.cfg.params.get("max_size", 1333))
        model = fasterrcnn_resnet50_fpn(
            weights=weights,
            weights_backbone=None,
            min_size=min_size,
            max_size=max_size,
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    def _optimizer(self, model, train_cfg: TrainConfig):
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer_name = str(train_cfg.hparams.get("optimizer", "sgd")).lower()
        lr = float(train_cfg.hparams.get("learning_rate", train_cfg.hparams.get("lr", 0.001)))
        weight_decay = float(train_cfg.hparams.get("weight_decay", 0.0005))
        if optimizer_name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        momentum = float(train_cfg.hparams.get("momentum", 0.9))
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    def _device(self, params: dict) -> torch.device:
        requested = params.get("device")
        if requested is not None:
            return torch.device(requested)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _TorchvisionDataset(torch.utils.data.Dataset):
    def __init__(self, ds: Dataset, parser: HFDetectionRowParser, transform):
        self.ds = ds
        self.parser = parser
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        sample = self.transform(self.parser.parse(self.ds[idx]))
        return _image_tensor(sample), _target_dict(sample), sample


def _image_tensor(sample: DetectionSample) -> torch.Tensor:
    array = sample.image.astype("float32") / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _target_dict(sample: DetectionSample) -> dict[str, torch.Tensor]:
    boxes = [target.bbox.xyxy() for target in sample.targets]
    labels = [target.label_id + 1 for target in sample.targets]
    area = [target.bbox.area for target in sample.targets]
    iscrowd = [int(target.iscrowd) for target in sample.targets]

    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "area": torch.tensor(area, dtype=torch.float32),
        "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
    }


def _collate(batch):
    images, targets, samples = zip(*batch)
    return list(images), list(targets), list(samples)
