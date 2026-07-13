from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from datasets import Dataset

from obj_det.models.adapters.base import BaseModelAdapter
from obj_det.models.data.hf_dataset import HFTrainerDetectionDataset
from obj_det.models.data.hf_targets import make_hf_detection_collate
from obj_det.models.data.row_parser import HFDetectionRowParser
from obj_det.models.data.row_batches import iter_hf_row_batches
from obj_det.models.data.sample_source import DetectionSampleSource
from obj_det.models.data.transforms import canonicalize_prediction_bbox, build_detection_transform
from obj_det.models.logging.base import BaseExperimentLogger
from obj_det.models.logging.trainer_callback import make_transformers_logging_callback
from obj_det.models.schemas.artifact import ModelArtifact
from obj_det.models.schemas.config import EvalConfig, PredictConfig, TrainConfig
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


class HFTrainerDetectionAdapter(BaseModelAdapter):
    """Transformers Trainer backend for HF Dataset object-detection rows."""

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
        require_single_process(context="Controlled HF Trainer training")
        try:
            from transformers import AutoImageProcessor, AutoModelForObjectDetection, Trainer, TrainerCallback, TrainingArguments
        except ImportError as exc:
            raise ImportError("Install the models extra to use backend='hf_trainer'.") from exc

        set_seed(train_cfg.seed)
        train_cfg.output_dir.mkdir(parents=True, exist_ok=True)

        processor = AutoImageProcessor.from_pretrained(
            str(self.cfg.model_name_or_path),
            **self.cfg.params.get("processor_from_pretrained_kwargs", {}),
        )
        id2label, label2id = self._label_maps(train_cfg.classes)
        model = AutoModelForObjectDetection.from_pretrained(
            str(self.cfg.weights or self.cfg.model_name_or_path),
            num_labels=len(train_cfg.classes),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            **self.cfg.params.get("model_from_pretrained_kwargs", {}),
        )

        parser = HFDetectionRowParser(classes=train_cfg.classes, label_mode=train_cfg.label_mode, decode_backend=train_cfg.loader.decode_backend)
        transform = build_detection_transform(train_cfg.preprocess, train_cfg.augmentation, seed=train_cfg.seed)
        eval_transform = build_detection_transform(train_cfg.preprocess)
        processor_kwargs = train_cfg.backend_params.get("processor_kwargs", {"do_resize": False})
        train_source = DetectionSampleSource(train_ds, parser, predecode_images=train_cfg.loader.predecode_images)
        val_source = DetectionSampleSource(val_ds, parser, predecode_images=train_cfg.loader.predecode_images)
        train_data = HFTrainerDetectionDataset(train_source, transform)
        val_data = HFTrainerDetectionDataset(val_source, eval_transform)

        args = self._training_args(train_cfg, epoch_eval_enabled=epoch_eval_cfg is not None)
        checkpoint_state = CheckpointState(train_cfg.output_dir)
        adapter = self
        class ProtocolTrainer(Trainer):
            def __init__(self, *args, protocol_processor, **kwargs):
                self.protocol_processor = protocol_processor
                self.protocol_last_epoch = 0
                super().__init__(*args, **kwargs)

            def create_optimizer(self, model=None):
                if self.optimizer is not None:
                    return self.optimizer
                model = self.model if model is None else model
                cfg = train_cfg.optimizer
                params = build_adamw_param_groups(model, weight_decay=cfg.weight_decay)
                self.optimizer = torch.optim.AdamW(
                    params,
                    lr=float(train_cfg.hparams.get("learning_rate", 5.0e-5)),
                    betas=(cfg.beta1, cfg.beta2),
                    eps=cfg.epsilon,
                )
                return self.optimizer

            def create_scheduler(self, num_training_steps: int, optimizer=None):
                if self.lr_scheduler is not None:
                    return self.lr_scheduler
                optimizer = optimizer or self.optimizer
                if optimizer is None:
                    raise ValueError("HF scheduler requires an initialized optimizer")
                steps_per_epoch = optimizer_steps_per_epoch(
                    len(self.get_train_dataloader()),
                    train_cfg.gradient_accumulation_steps,
                )
                scheduler_cfg = train_cfg.scheduler
                self.lr_scheduler = build_warmup_cosine_scheduler(
                    optimizer,
                    warmup_steps=int(round(scheduler_cfg.warmup_epochs * steps_per_epoch)),
                    total_steps=int(scheduler_cfg.total_epochs * steps_per_epoch),
                    min_lr_ratio=scheduler_cfg.min_lr_ratio,
                )
                return self.lr_scheduler

            def _maybe_log_save_evaluate(self, *method_args, **method_kwargs):
                super()._maybe_log_save_evaluate(*method_args, **method_kwargs)
                if epoch_eval_cfg is None or self.state.epoch is None:
                    return
                epoch = int(round(float(self.state.epoch)))
                if epoch <= self.protocol_last_epoch or abs(float(self.state.epoch) - epoch) > 1.0e-6:
                    return
                self.protocol_last_epoch = epoch
                checkpoint_path = train_cfg.output_dir / f"checkpoint-{self.state.global_step}"
                if not checkpoint_path.exists():
                    model_arg = method_args[2] if len(method_args) > 2 else method_kwargs.get("model", self.model)
                    trial_arg = method_args[3] if len(method_args) > 3 else method_kwargs.get("trial")
                    self._save_checkpoint(model_arg, trial_arg)
                self.protocol_processor.save_pretrained(str(checkpoint_path))
                artifact = adapter._artifact_for_checkpoint(train_cfg, checkpoint_path)
                result = adapter.evaluate(
                    val_ds,
                    artifact,
                    epoch_eval_cfg,
                    logger=logger,
                    log_prefix="val/epoch",
                    log_step=self.state.global_step,
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
                    logger.log_metrics(
                        {
                            "val/epoch_index": epoch,
                            "early_stopping/bad_epochs": checkpoint_state.early_stopping.bad_epochs,
                            "early_stopping/best_metric": checkpoint_state.early_stopping.best_metric,
                            "early_stopping/best_epoch": checkpoint_state.early_stopping.best_epoch,
                        },
                        step=self.state.global_step,
                    )
                if should_stop:
                    self.control.should_training_stop = True

        trainer = ProtocolTrainer(
            model=model,
            args=args,
            train_dataset=train_data,
            eval_dataset=val_data,
            data_collator=make_hf_detection_collate(processor, processor_kwargs),
            processing_class=processor,
            protocol_processor=processor,
            callbacks=[make_transformers_logging_callback(TrainerCallback, logger, log_prefix)] if logger else None,
        )
        trainer.train()

        artifact_path = train_cfg.output_dir
        checkpoint_path = checkpoint_state.best_checkpoint if train_cfg.early_stopping.restore_best else None
        if checkpoint_path is None:
            checkpoint_path = checkpoint_state.last_checkpoint
        if checkpoint_path is None:
            trainer._save_checkpoint(trainer.model, None)
            checkpoint_path = artifact_path / f"checkpoint-{trainer.state.global_step}"
            processor.save_pretrained(str(checkpoint_path))
            checkpoint_state.record_epoch(
                epoch=int(round(float(trainer.state.epoch or train_cfg.max_epochs or 1))),
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
            "learning_rate": float(train_cfg.hparams.get("learning_rate", 5.0e-5)),
        }
        scheduler_meta = train_cfg.scheduler.model_dump(mode="json")

        return ModelArtifact(
            model_key=self.key,
            backend=self.backend,
            run_key=train_cfg.run_key,
            classes=train_cfg.classes,
            label_mode=train_cfg.label_mode,
            artifact_path=artifact_path,
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
                "pretrained_source": str(self.cfg.weights or self.cfg.model_name_or_path),
                "optimizer": optimizer_meta,
                "scheduler": scheduler_meta,
                **checkpoint_state.artifact_meta(),
            },
        )

    def predict(
        self,
        ds: Dataset,
        artifact: ModelArtifact,
        predict_cfg: PredictConfig,
    ) -> Iterator[PredictionRecord]:
        try:
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
        except ImportError as exc:
            raise ImportError("Install the models extra to use backend='hf_trainer'.") from exc

        checkpoint = artifact.checkpoint_path or artifact.artifact_path or Path(str(self.cfg.model_name_or_path))
        processor = AutoImageProcessor.from_pretrained(str(checkpoint))
        model = AutoModelForObjectDetection.from_pretrained(str(checkpoint))
        device = torch.device(predict_cfg.backend_params.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
        model.to(device)
        model.eval()

        parser = HFDetectionRowParser(
            classes=predict_cfg.classes,
            label_mode=predict_cfg.label_mode,
            decode_backend=predict_cfg.backend_params.get("decode_backend", "pil"),
        )
        transform = build_detection_transform(predict_cfg.preprocess)
        processor_kwargs = predict_cfg.backend_params.get("processor_kwargs", {"do_resize": False})

        with torch.no_grad():
            for rows in iter_hf_row_batches(ds, predict_cfg.batch_size):
                original_samples = [parser.parse(row) for row in rows]
                samples = [transform(sample) for sample in original_samples]
                inputs = processor(
                    images=[np.ascontiguousarray(sample.image) for sample in samples],
                    return_tensors="pt",
                    **processor_kwargs,
                )
                inputs = {key: value.to(device) for key, value in inputs.items() if isinstance(value, torch.Tensor)}
                outputs = model(**inputs)
                target_sizes = torch.tensor(
                    [[sample.height, sample.width] for sample in samples],
                    dtype=torch.float32,
                    device=device,
                )
                results = processor.post_process_object_detection(
                    outputs,
                    threshold=predict_cfg.conf_threshold,
                    target_sizes=target_sizes,
                )

                for original, sample, result in zip(original_samples, samples, results):
                    preprocess = sample.meta.get("preprocess")
                    predictions: list[PredictionObject] = []
                    invalid_prediction_boxes_dropped = 0
                    order = torch.argsort(result["scores"], descending=True)[: predict_cfg.max_detections_per_image]
                    boxes = result["boxes"][order]
                    scores = result["scores"][order]
                    labels = result["labels"][order]
                    for box, score, label in zip(boxes, scores, labels):
                        class_idx = int(label.detach().cpu())
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
                                score=float(score.detach().cpu()),
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

    def _training_args(self, train_cfg: TrainConfig, *, epoch_eval_enabled: bool = False):
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

        save_strategy = (
            "epoch"
            if max_steps < 0
            and train_cfg.checkpoint.save_every_epochs == 1
            and (epoch_eval_enabled or train_cfg.checkpoint.keep_all_epoch_checkpoints)
            else "no"
        )

        return TrainingArguments(
            output_dir=str(train_cfg.output_dir),
            num_train_epochs=epochs,
            max_steps=max_steps,
            per_device_train_batch_size=train_cfg.batch_size,
            per_device_eval_batch_size=train_cfg.batch_size,
            gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
            learning_rate=float(hparams.get("learning_rate", 5e-5)),
            weight_decay=float(train_cfg.optimizer.weight_decay),
            warmup_ratio=0.0,
            lr_scheduler_type="constant",
            max_grad_norm=float(hparams.get("max_grad_norm", 1.0)),
            seed=train_cfg.seed,
            fp16=bool(train_cfg.amp and torch.cuda.is_available()),
            eval_strategy="no",
            save_strategy=save_strategy,
            logging_strategy="steps",
            logging_steps=train_cfg.logging_steps,
            report_to=[],
            remove_unused_columns=False,
            load_best_model_at_end=False,
            **loader_args,
            **backend_args,
        )

    def _label_maps(self, classes: list[str]) -> tuple[dict[int, str], dict[str, int]]:
        id2label = {idx: label for idx, label in enumerate(classes)}
        label2id = {label: idx for idx, label in enumerate(classes)}
        return id2label, label2id

    def _artifact_for_checkpoint(self, train_cfg: TrainConfig, checkpoint_path: Path) -> ModelArtifact:
        return ModelArtifact(
            model_key=self.key,
            backend=self.backend,
            run_key=train_cfg.run_key,
            classes=train_cfg.classes,
            label_mode=train_cfg.label_mode,
            artifact_path=train_cfg.output_dir,
            checkpoint_path=checkpoint_path,
        )
