from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Literal

import torch
from datasets import Dataset
from torchvision.models.detection import (
    FCOS_ResNet50_FPN_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
    MaskRCNN_ResNet50_FPN_Weights,
    RetinaNet_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
    fcos_resnet50_fpn,
    maskrcnn_resnet50_fpn,
    retinanet_resnet50_fpn,
)
from torchvision.models.detection.fcos import FCOSClassificationHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from transformers import Trainer, TrainerCallback

from obj_det.models.adapters.base import BaseModelAdapter
from obj_det.models.data.bbox import area_xywh, xywh_to_xyxy
from obj_det.models.data.loader import seed_worker_transform
from obj_det.models.data.row_parser import HFDetectionRowParser
from obj_det.models.data.sample import DetectionSample
from obj_det.models.data.sample_source import DetectionSampleSource
from obj_det.models.data.transforms import canonicalize_prediction_bbox, build_detection_transform
from obj_det.models.logging.base import BaseExperimentLogger
from obj_det.models.logging.trainer_callback import make_transformers_logging_callback
from obj_det.models.schemas.artifact import ModelArtifact
from obj_det.models.schemas.config import EvalConfig, PredictConfig, PreprocessConfig, TrainConfig
from obj_det.models.schemas.prediction import PredictionObject, PredictionRecord
from obj_det.models.training import (
    CheckpointState,
    build_adamw_param_groups,
    build_warmup_cosine_scheduler,
    optimizer_steps_per_epoch,
    require_metric,
    require_single_process,
)
from obj_det.models.utils.repro import set_seed


@dataclass(frozen=True)
class _TorchvisionModelSpec:
    model_name: str
    builder: Callable[..., torch.nn.Module]
    weights_cls: type
    head_kind: Literal["roi", "anchor"]
    label_offset: int
    threshold_parent: tuple[str, ...]

    def model_num_classes(self, canonical_num_classes: int) -> int:
        return canonical_num_classes + self.label_offset

    def training_label(self, canonical_label: int) -> int:
        return canonical_label + self.label_offset

    def canonical_label(self, model_label: int) -> int:
        return model_label - self.label_offset


_TORCHVISION_MODEL_SPECS = {
    "fasterrcnn_resnet50_fpn": _TorchvisionModelSpec(
        model_name="fasterrcnn_resnet50_fpn",
        builder=fasterrcnn_resnet50_fpn,
        weights_cls=FasterRCNN_ResNet50_FPN_Weights,
        head_kind="roi",
        label_offset=1,
        threshold_parent=("roi_heads",),
    ),
    "maskrcnn_resnet50_fpn": _TorchvisionModelSpec(
        model_name="maskrcnn_resnet50_fpn",
        builder=maskrcnn_resnet50_fpn,
        weights_cls=MaskRCNN_ResNet50_FPN_Weights,
        head_kind="roi",
        label_offset=1,
        threshold_parent=("roi_heads",),
    ),
    "retinanet_resnet50_fpn": _TorchvisionModelSpec(
        model_name="retinanet_resnet50_fpn",
        builder=retinanet_resnet50_fpn,
        weights_cls=RetinaNet_ResNet50_FPN_Weights,
        head_kind="anchor",
        label_offset=0,
        threshold_parent=(),
    ),
    "fcos_resnet50_fpn": _TorchvisionModelSpec(
        model_name="fcos_resnet50_fpn",
        builder=fcos_resnet50_fpn,
        weights_cls=FCOS_ResNet50_FPN_Weights,
        head_kind="anchor",
        label_offset=0,
        threshold_parent=(),
    ),
}


class TorchvisionDetectionAdapter(BaseModelAdapter):
    def train(
        self,
        train_ds: Dataset,
        val_ds: Dataset,
        train_cfg: TrainConfig,
        *,
        epoch_eval_cfg: EvalConfig | None = None,
        logger: BaseExperimentLogger | None = None,
        log_prefix: str = "train",
    ) -> ModelArtifact:
        require_single_process(context="Controlled TorchVision training")
        self._require_model_preprocess(train_cfg.preprocess)
        set_seed(train_cfg.seed)
        train_cfg.output_dir.mkdir(parents=True, exist_ok=True)

        spec = self._model_spec()
        weights = self._resolve_weights(spec)
        pretraining_meta = self._pretraining_metadata(spec, weights)
        model = self._build_model(
            spec=spec,
            num_classes=spec.model_num_classes(len(train_cfg.classes)),
            preprocess=train_cfg.preprocess,
            weights=weights,
        )
        parser = HFDetectionRowParser(classes=train_cfg.classes, label_mode=train_cfg.label_mode, decode_backend=train_cfg.loader.decode_backend)
        transform = build_detection_transform(train_cfg.preprocess, train_cfg.augmentation, seed=train_cfg.seed)
        eval_transform = build_detection_transform(train_cfg.preprocess)
        train_source = DetectionSampleSource(train_ds, parser, predecode_images=train_cfg.loader.predecode_images)
        val_source = DetectionSampleSource(val_ds, parser, predecode_images=train_cfg.loader.predecode_images)
        checkpoint_state = CheckpointState(train_cfg.output_dir)
        def epoch_callback(protocol_trainer, epoch: int) -> bool:
            max_epochs = int(train_cfg.max_epochs or 1)
            should_save = (
                epoch % train_cfg.checkpoint.save_every_epochs == 0
                or epoch >= max_epochs
                or epoch_eval_cfg is not None
            )
            if not should_save:
                return False
            checkpoint_path = self._save_training_checkpoint(
                protocol_trainer,
                train_cfg=train_cfg,
                spec=spec,
                pretraining_meta=pretraining_meta,
                epoch=epoch,
            )
            metric = None
            if epoch_eval_cfg is not None:
                result = self.evaluate(
                    val_ds,
                    self._artifact_for_checkpoint(train_cfg, checkpoint_path, pretraining_meta),
                    epoch_eval_cfg,
                    logger=logger,
                    log_prefix="val/epoch",
                    log_step=protocol_trainer.state.global_step,
                )
                metric = require_metric(
                    result.metrics,
                    train_cfg.early_stopping.metric,
                    context="early-stopping",
                )
            should_stop = checkpoint_state.record_epoch(
                epoch=epoch,
                checkpoint_path=checkpoint_path,
                metric=metric,
                early_stopping_cfg=train_cfg.early_stopping,
            )
            if logger is not None:
                logger.log_artifact(checkpoint_path, name=f"checkpoint_epoch_{epoch:03d}")
                if metric is not None:
                    logger.log_metrics(
                        {
                            "val/epoch_index": epoch,
                            "early_stopping/bad_epochs": checkpoint_state.early_stopping.bad_epochs,
                            "early_stopping/best_metric": checkpoint_state.early_stopping.best_metric,
                            "early_stopping/best_epoch": checkpoint_state.early_stopping.best_epoch,
                        },
                        step=protocol_trainer.state.global_step,
                    )
            return should_stop

        trainer = _TorchvisionTrainer(
            model=model,
            args=self._training_args(train_cfg),
            train_dataset=_TorchvisionTrainerDataset(train_source, transform, spec),
            eval_dataset=_TorchvisionTrainerDataset(val_source, eval_transform, spec),
            data_collator=_torchvision_collate,
            train_cfg=train_cfg,
            epoch_callback=epoch_callback,
            callbacks=[make_transformers_logging_callback(TrainerCallback, logger, log_prefix)] if logger else None,
        )
        trainer.train()

        checkpoint_path = checkpoint_state.best_checkpoint if train_cfg.early_stopping.restore_best else None
        if checkpoint_path is None:
            checkpoint_path = checkpoint_state.last_checkpoint
        if checkpoint_path is None:
            epoch = int(round(float(trainer.state.epoch or train_cfg.max_epochs or 1)))
            checkpoint_path = self._save_training_checkpoint(
                trainer,
                train_cfg=train_cfg,
                spec=spec,
                pretraining_meta=pretraining_meta,
                epoch=epoch,
            )
            checkpoint_state.record_epoch(
                epoch=epoch,
                checkpoint_path=checkpoint_path,
                metric=None,
                early_stopping_cfg=train_cfg.early_stopping,
            )
        selected_is_best = (
            checkpoint_state.best_checkpoint is not None
            and checkpoint_path == checkpoint_state.best_checkpoint
        )
        optimizer_steps = int(trainer.state.global_step)
        if logger is not None:
            logger.log_metrics(
                {f"{log_prefix}/optimizer_steps": optimizer_steps},
                step=optimizer_steps,
            )

        train_loss = None
        if trainer.state.log_history:
            for row in reversed(trainer.state.log_history):
                if "train_loss" in row:
                    train_loss = float(row["train_loss"])
                    break
        optimizer_meta = {
            **train_cfg.optimizer.model_dump(mode="json"),
            "learning_rate": float(train_cfg.hparams.get("learning_rate", 0.001)),
        }
        scheduler_meta = train_cfg.scheduler.model_dump(mode="json")

        return ModelArtifact(
            model_key=self.key,
            backend=self.backend,
            run_key=train_cfg.run_key,
            classes=train_cfg.classes,
            label_mode=train_cfg.label_mode,
            artifact_path=train_cfg.output_dir,
            checkpoint_path=checkpoint_path,
            best_metric_name=(
                train_cfg.early_stopping.metric
                if checkpoint_state.best_epoch is not None
                else ("train_loss" if train_loss is not None else None)
            ),
            best_metric_value=(
                checkpoint_state.best_metric
                if checkpoint_state.best_epoch is not None
                else train_loss
            ),
            meta={
                "trainer_global_step": trainer.state.global_step,
                "optimizer_steps": optimizer_steps,
                "checkpoint_selection": "best_validation" if selected_is_best else "last",
                "model_spec": spec.model_name,
                "label_offset": spec.label_offset,
                "optimizer": optimizer_meta,
                "scheduler": scheduler_meta,
                **checkpoint_state.artifact_meta(),
                **pretraining_meta,
            },
        )

    def predict(
        self,
        ds: Dataset,
        artifact: ModelArtifact,
        predict_cfg: PredictConfig,
    ) -> Iterator[PredictionRecord]:
        if artifact.checkpoint_path is None:
            raise ValueError("Torchvision artifact is missing checkpoint_path")

        self._require_model_preprocess(predict_cfg.preprocess)
        device = self._device(predict_cfg.backend_params)
        spec = self._model_spec()
        model = self._build_model(
            spec=spec,
            num_classes=spec.model_num_classes(len(predict_cfg.classes)),
            preprocess=predict_cfg.preprocess,
            weights=None,
            predict_cfg=predict_cfg,
        ).to(device)
        checkpoint = torch.load(artifact.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        parser = HFDetectionRowParser(
            classes=predict_cfg.classes,
            label_mode=predict_cfg.label_mode,
            decode_backend=predict_cfg.backend_params.get("decode_backend", "pil"),
        )
        transform = build_detection_transform(predict_cfg.preprocess)

        with torch.no_grad():
            for row in ds:
                original = parser.parse(row)
                sample = transform(original)
                image = _image_tensor(sample).to(device)
                output = model([image])[0]
                predictions = []
                invalid_prediction_boxes_dropped = 0
                preprocess = sample.meta.get("preprocess")

                for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
                    score_value = float(score.detach().cpu())
                    if score_value < predict_cfg.conf_threshold:
                        continue
                    raw_label = int(label.detach().cpu())
                    class_idx = spec.canonical_label(raw_label)
                    if class_idx < 0 or class_idx >= len(predict_cfg.classes):
                        continue
                    bbox = canonicalize_prediction_bbox(
                        box.detach().cpu().tolist(),
                        image_width=sample.width,
                        image_height=sample.height,
                        preprocess=preprocess,
                    )
                    if bbox is None:
                        invalid_prediction_boxes_dropped += 1
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
                    meta={"invalid_prediction_boxes_dropped": invalid_prediction_boxes_dropped},
                )

    def _model_spec(self) -> _TorchvisionModelSpec:
        model_name = str(self.cfg.model_name_or_path)
        try:
            return _TORCHVISION_MODEL_SPECS[model_name]
        except KeyError as exc:
            supported = ", ".join(_TORCHVISION_MODEL_SPECS)
            raise ValueError(f"Torchvision backend supports: {supported}") from exc

    def _build_model(
        self,
        *,
        num_classes: int,
        preprocess: PreprocessConfig,
        spec: _TorchvisionModelSpec | None = None,
        weights=None,
        predict_cfg: PredictConfig | None = None,
    ):
        spec = spec or self._model_spec()
        if preprocess.resize_mode != "shortest_edge":
            raise ValueError("TorchVision models require shortest_edge preprocessing")

        model = spec.builder(
            weights=weights,
            weights_backbone=None,
            min_size=preprocess.shortest_edge,
            max_size=preprocess.longest_edge,
        )
        if spec.head_kind == "roi":
            self._replace_roi_box_predictor(model, num_classes)
            if spec.model_name == "maskrcnn_resnet50_fpn":
                self._disable_mask_head(model)
        else:
            self._replace_anchor_classification_head(model, num_classes)

        if predict_cfg is not None:
            self._apply_predict_config(model, spec, predict_cfg)
        return model

    def _resolve_weights(self, spec: _TorchvisionModelSpec):
        configured = self.cfg.params["weights"] if "weights" in self.cfg.params else self.cfg.weights
        if configured is None:
            return None
        if isinstance(configured, spec.weights_cls):
            return configured

        value = str(configured).strip()
        if value.lower() in {"none", "null", "false"}:
            return None
        if value.lower() == "default":
            return spec.weights_cls.DEFAULT

        member_name = value.rsplit(".", 1)[-1]
        member = getattr(spec.weights_cls, member_name, None)
        if member is None:
            choices = ", ".join(member.name for member in spec.weights_cls)
            raise ValueError(
                f"Unknown weights {value!r} for {spec.model_name}; use 'default', 'none', or one of: {choices}"
            )
        return member

    def _pretraining_metadata(self, spec: _TorchvisionModelSpec, weights) -> dict:
        configured = self.cfg.params["weights"] if "weights" in self.cfg.params else self.cfg.weights
        if weights is None:
            return {
                "pretrained": False,
                "pretrained_config": None if configured is None else str(configured),
                "pretrained_source": None,
            }

        value = getattr(weights, "value", None)
        url = getattr(weights, "url", None) or getattr(value, "url", None)
        name = getattr(weights, "name", str(weights))
        return {
            "pretrained": True,
            "pretrained_config": None if configured is None else str(configured),
            "pretrained_weights": name,
            "pretrained_source": url or f"torchvision:{spec.model_name}:{name}",
        }

    def _save_training_checkpoint(
        self,
        trainer,
        *,
        train_cfg: TrainConfig,
        spec: _TorchvisionModelSpec,
        pretraining_meta: dict,
        epoch: int,
    ) -> Path:
        checkpoint_dir = train_cfg.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        scaler = getattr(getattr(trainer, "accelerator", None), "scaler", None)
        payload = {
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict() if trainer.optimizer is not None else None,
            "scheduler_state_dict": trainer.lr_scheduler.state_dict() if trainer.lr_scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "global_optimizer_step": trainer.state.global_step,
            "classes": train_cfg.classes,
            "label_mode": train_cfg.label_mode,
            "model_name_or_path": str(self.cfg.model_name_or_path),
            "model_spec": spec.model_name,
            "label_offset": spec.label_offset,
            "pretraining": pretraining_meta,
            "resolved_train_config": train_cfg.model_dump(mode="json"),
            "rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        torch.save(payload, checkpoint_path)
        return checkpoint_path

    def _artifact_for_checkpoint(
        self,
        train_cfg: TrainConfig,
        checkpoint_path: Path,
        pretraining_meta: dict,
    ) -> ModelArtifact:
        return ModelArtifact(
            model_key=self.key,
            backend=self.backend,
            run_key=train_cfg.run_key,
            classes=train_cfg.classes,
            label_mode=train_cfg.label_mode,
            artifact_path=train_cfg.output_dir,
            checkpoint_path=checkpoint_path,
            meta=pretraining_meta,
        )

    def _apply_predict_config(
        self,
        model,
        spec: _TorchvisionModelSpec,
        predict_cfg: PredictConfig,
    ) -> None:
        threshold_owner = model
        for attribute in spec.threshold_parent:
            threshold_owner = getattr(threshold_owner, attribute)

        threshold_owner.score_thresh = float(predict_cfg.conf_threshold)
        threshold_owner.nms_thresh = float(predict_cfg.iou_threshold)

        max_detections = getattr(predict_cfg, "max_detections_per_image", None)
        if max_detections is None:
            max_detections = predict_cfg.backend_params.get("max_detections_per_image")
        if max_detections is not None:
            max_detections = int(max_detections)
            if max_detections <= 0:
                raise ValueError("max_detections_per_image must be positive")
            threshold_owner.detections_per_img = max_detections

    def _replace_roi_box_predictor(self, model, num_classes: int) -> None:
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def _replace_anchor_classification_head(self, model, num_classes: int) -> None:
        head = model.head.classification_head
        in_channels = head.cls_logits.in_channels
        num_anchors = head.num_anchors
        if isinstance(head, RetinaNetClassificationHead):
            model.head.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, num_classes)
            return
        if isinstance(head, FCOSClassificationHead):
            model.head.classification_head = FCOSClassificationHead(in_channels, num_anchors, num_classes)
            return
        raise TypeError(f"Unsupported TorchVision classification head: {type(head)!r}")

    def _disable_mask_head(self, model) -> None:
        model.roi_heads.mask_roi_pool = None
        model.roi_heads.mask_head = None
        model.roi_heads.mask_predictor = None

    def _training_args(self, train_cfg: TrainConfig):
        from transformers import TrainingArguments

        hparams = train_cfg.hparams
        max_steps = train_cfg.max_steps if train_cfg.max_steps is not None else -1
        epochs = float(train_cfg.max_epochs or 1)
        backend_args = {
            key: value
            for key, value in train_cfg.backend_params.get("training_args", {}).items()
            if key != "logging_steps"
        }
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
            learning_rate=float(hparams.get("learning_rate", 0.001)),
            weight_decay=float(train_cfg.optimizer.weight_decay),
            warmup_ratio=0.0,
            lr_scheduler_type="constant",
            max_grad_norm=float(hparams.get("max_grad_norm", 1.0)),
            seed=train_cfg.seed,
            fp16=bool(train_cfg.amp and torch.cuda.is_available()),
            eval_strategy="no",
            save_strategy="no",
            logging_strategy="steps",
            logging_steps=train_cfg.logging_steps,
            report_to=[],
            remove_unused_columns=False,
            load_best_model_at_end=False,
            **loader_args,
            **backend_args,
        )

    def _device(self, params: dict) -> torch.device:
        requested = params.get("device")
        if requested is not None:
            return torch.device(requested)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _TorchvisionTrainer(Trainer):
    def __init__(self, *args, train_cfg: TrainConfig, epoch_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_cfg = train_cfg
        self.epoch_callback = epoch_callback
        self.protocol_last_epoch = 0

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

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        super()._maybe_log_save_evaluate(*args, **kwargs)
        if self.epoch_callback is None or self.state.epoch is None:
            return
        epoch = int(round(float(self.state.epoch)))
        if epoch <= self.protocol_last_epoch or abs(float(self.state.epoch) - epoch) > 1.0e-6:
            return
        self.protocol_last_epoch = epoch
        if self.epoch_callback(self, epoch):
            self.control.should_training_stop = True

    def create_optimizer(self, model=None):
        if self.optimizer is not None:
            return self.optimizer

        model = self.model if model is None else model
        hparams = self.train_cfg.hparams
        lr = float(hparams.get("learning_rate", 0.001))

        cfg = self.train_cfg.optimizer
        params = build_adamw_param_groups(model, weight_decay=cfg.weight_decay)
        self.optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.epsilon,
        )

        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is not None:
            return self.lr_scheduler

        optimizer = optimizer or self.optimizer
        if optimizer is None:
            raise ValueError("TorchVision scheduler requires an initialized optimizer")

        num_batches = len(self.get_train_dataloader())
        steps_per_epoch = optimizer_steps_per_epoch(
            num_batches,
        )
        cfg = self.train_cfg.scheduler
        warmup_steps = int(round(cfg.warmup_epochs * steps_per_epoch))
        total_steps = int(cfg.total_epochs * steps_per_epoch)
        self.lr_scheduler = build_warmup_cosine_scheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=cfg.min_lr_ratio,
        )
        return self.lr_scheduler


class _TorchvisionTrainerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        source: DetectionSampleSource,
        transform,
        spec: _TorchvisionModelSpec | None = None,
    ):
        self.source = source
        self.transform = transform
        self.spec = spec or _TORCHVISION_MODEL_SPECS["fasterrcnn_resnet50_fpn"]

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, idx: int):
        seed_worker_transform(self.transform)
        sample = self.transform(self.source[idx])
        return {"image": _image_tensor(sample), "target": _target_dict(sample, self.spec)}


def _image_tensor(sample: DetectionSample) -> torch.Tensor:
    if sample.image is None:
        raise ValueError("TorchVision tensor conversion requires decoded image data")
    array = sample.image.astype("float32") / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _target_dict(sample: DetectionSample, spec: _TorchvisionModelSpec) -> dict[str, torch.Tensor]:
    boxes = [xywh_to_xyxy(target.bbox_xywh) for target in sample.targets]
    labels = [spec.training_label(target.label_id) for target in sample.targets]
    area = [area_xywh(target.bbox_xywh) for target in sample.targets]
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
