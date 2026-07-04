from __future__ import annotations

from typing import Iterator

import torch
from datasets import Dataset
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from transformers import Trainer

from obj_det.datasets.models import BBox
from obj_det.models.adapters.base import BaseModelAdapter
from obj_det.models.data.loader import seed_worker_transform
from obj_det.models.data.row_parser import HFDetectionRowParser
from obj_det.models.data.sample import DetectionSample
from obj_det.models.data.sample_source import DetectionSampleSource
from obj_det.models.data.transforms import bbox_to_original, build_detection_transform
from obj_det.models.schemas.artifact import ModelArtifact
from obj_det.models.schemas.config import PredictConfig, TrainConfig
from obj_det.models.schemas.prediction import PredictionObject, PredictionRecord
from obj_det.models.utils.repro import set_seed


class TorchvisionDetectionAdapter(BaseModelAdapter):
    def train(self, train_ds: Dataset, val_ds: Dataset, train_cfg: TrainConfig) -> ModelArtifact:
        set_seed(train_cfg.seed)
        train_cfg.output_dir.mkdir(parents=True, exist_ok=True)

        model = self._build_model(num_classes=len(train_cfg.classes) + 1)
        parser = HFDetectionRowParser(classes=train_cfg.classes, label_mode=train_cfg.label_mode)
        transform = build_detection_transform(train_cfg.transform, seed=train_cfg.seed)
        train_source = DetectionSampleSource(train_ds, parser, predecode_images=train_cfg.loader.predecode_images)
        val_source = DetectionSampleSource(val_ds, parser, predecode_images=train_cfg.loader.predecode_images)

        trainer = _TorchvisionTrainer(
            model=model,
            args=self._training_args(train_cfg),
            train_dataset=_TorchvisionTrainerDataset(train_source, transform),
            eval_dataset=_TorchvisionTrainerDataset(val_source, transform),
            data_collator=_torchvision_collate,
            train_cfg=train_cfg,
        )
        trainer.train()

        checkpoint_path = train_cfg.output_dir / "checkpoint.pt"
        torch.save(
            {
                "model_state_dict": trainer.model.state_dict(),
                "classes": train_cfg.classes,
                "label_mode": train_cfg.label_mode,
                "model_name_or_path": str(self.cfg.model_name_or_path),
            },
            checkpoint_path,
        )

        train_loss = None
        if trainer.state.log_history:
            for row in reversed(trainer.state.log_history):
                if "train_loss" in row:
                    train_loss = float(row["train_loss"])
                    break

        return ModelArtifact(
            model_key=self.key,
            backend=self.backend,
            run_key=train_cfg.run_key,
            classes=train_cfg.classes,
            label_mode=train_cfg.label_mode,
            artifact_path=train_cfg.output_dir,
            checkpoint_path=checkpoint_path,
            best_metric_name="train_loss" if train_loss is not None else None,
            best_metric_value=train_loss,
            meta={"trainer_global_step": trainer.state.global_step},
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
        transform = build_detection_transform(predict_cfg.transform)

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

    def _training_args(self, train_cfg: TrainConfig):
        from transformers import TrainingArguments

        hparams = train_cfg.hparams
        max_steps = train_cfg.max_steps if train_cfg.max_steps is not None else -1
        epochs = float(train_cfg.max_epochs or 1)
        loader = train_cfg.loader
        loader_args = {
            "dataloader_num_workers": loader.num_workers,
            "dataloader_pin_memory": loader.pin_memory,
            "dataloader_persistent_workers": bool(loader.persistent_workers) if loader.num_workers > 0 else False,
        }
        if loader.num_workers > 0 and loader.prefetch_factor is not None:
            loader_args["dataloader_prefetch_factor"] = loader.prefetch_factor

        return TrainingArguments(
            output_dir=str(train_cfg.output_dir),
            num_train_epochs=epochs,
            max_steps=max_steps,
            per_device_train_batch_size=train_cfg.batch_size,
            per_device_eval_batch_size=train_cfg.batch_size,
            gradient_accumulation_steps=1,
            learning_rate=float(hparams.get("learning_rate", hparams.get("lr", 0.001))),
            weight_decay=float(hparams.get("weight_decay", 0.0005)),
            warmup_ratio=float(hparams.get("warmup_ratio", 0.0)),
            lr_scheduler_type=hparams.get("lr_scheduler_type", "linear"),
            max_grad_norm=float(hparams.get("max_grad_norm", 1.0)),
            seed=train_cfg.seed,
            fp16=bool(train_cfg.amp and torch.cuda.is_available()),
            eval_strategy=self._eval_strategy(train_cfg),
            save_strategy="epoch" if max_steps < 0 else "no",
            logging_strategy="steps",
            logging_steps=int(train_cfg.backend_params.get("logging_steps", 10)),
            report_to=[],
            remove_unused_columns=False,
            load_best_model_at_end=False,
            **loader_args,
            **train_cfg.backend_params.get("training_args", {}),
        )

    def _eval_strategy(self, train_cfg: TrainConfig) -> str:
        if not train_cfg.eval_strategy.enabled:
            return "no"
        if train_cfg.eval_strategy.every_epochs != 1:
            raise NotImplementedError("TorchVision HF Trainer eval_strategy currently supports every_epochs=1 only")
        return "epoch"

    def _device(self, params: dict) -> torch.device:
        requested = params.get("device")
        if requested is not None:
            return torch.device(requested)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _TorchvisionTrainer(Trainer):
    def __init__(self, *args, train_cfg: TrainConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_cfg = train_cfg

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss_dict = model(inputs["images"], inputs["targets"])
        loss = sum(loss for loss in loss_dict.values())
        return (loss, loss_dict) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        was_training = model.training
        model.train()
        with torch.no_grad():
            loss_dict = model(inputs["images"], inputs["targets"])
            loss = sum(loss for loss in loss_dict.values()).detach()
        model.train(was_training)
        return loss, None, None

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        params = [p for p in self.model.parameters() if p.requires_grad]
        hparams = self.train_cfg.hparams
        optimizer_name = str(hparams.get("optimizer", "sgd")).lower()
        lr = float(hparams.get("learning_rate", hparams.get("lr", 0.001)))
        weight_decay = float(hparams.get("weight_decay", 0.0005))

        if optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            momentum = float(hparams.get("momentum", 0.9))
            self.optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported TorchVision optimizer: {optimizer_name}")

        return self.optimizer


class _TorchvisionTrainerDataset(torch.utils.data.Dataset):
    def __init__(self, source: DetectionSampleSource, transform):
        self.source = source
        self.transform = transform

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, idx: int):
        seed_worker_transform(self.transform)
        sample = self.transform(self.source[idx])
        return {"image": _image_tensor(sample), "target": _target_dict(sample)}


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


def _torchvision_collate(batch):
    return {
        "images": [item["image"] for item in batch],
        "targets": [item["target"] for item in batch],
    }
