from __future__ import annotations

import os
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from obj_det.models.adapters.base import BaseModelAdapter
from obj_det.models.data.loader import dataloader_kwargs
from obj_det.models.data.row_parser import HFDetectionRowParser
from obj_det.models.data.row_batches import iter_hf_row_batches
from obj_det.models.data.sample_source import DetectionSampleSource
from obj_det.models.data.ultralytics_dataset import HFUltralyticsDetectionDataset, ultralytics_detection_collate
from obj_det.models.data.transforms import canonicalize_prediction_bbox, build_detection_transform
from obj_det.models.logging.base import BaseExperimentLogger
from obj_det.models.logging.metrics import flatten_prefixed_scalar_mapping
from obj_det.models.schemas.artifact import ModelArtifact
from obj_det.models.schemas.config import EvalConfig, PredictConfig, TrainConfig
from obj_det.models.schemas.prediction import PredictionObject, PredictionRecord
from obj_det.models.training import (
    CheckpointState,
    build_adamw_param_groups,
    build_warmup_cosine_scheduler,
    optimizer_steps_per_epoch,
    require_metric,
)
from obj_det.models.utils.repro import set_seed


_CONTROLLED_AUG_OFF = {
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "cutmix": 0.0,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.0,
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "erasing": 0.0,
    "close_mosaic": 0,
    "multi_scale": 0.0,
}


class UltralyticsDetectionAdapter(BaseModelAdapter):
    """Ultralytics backend using HF-backed dataloaders, not YOLO folders."""

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
        try:
            from ultralytics.models.yolo.detect import DetectionTrainer
        except ImportError as exc:
            raise ImportError("Install the models extra to use backend='ultralytics'.") from exc

        self._validate_single_process_device(train_cfg)

        if train_cfg.eval_strategy.enabled and epoch_eval_cfg is None:
            warnings.warn(
                "Ultralytics eval_strategy is enabled but no epoch_eval_cfg was supplied; "
                "epoch validation will not run.",
                RuntimeWarning,
                stacklevel=2,
            )

        set_seed(train_cfg.seed)
        train_cfg.output_dir.mkdir(parents=True, exist_ok=True)
        parser = HFDetectionRowParser(
            classes=train_cfg.classes,
            label_mode=train_cfg.label_mode,
            decode_backend=train_cfg.loader.decode_backend,
        )
        train_transform = build_detection_transform(train_cfg.preprocess, train_cfg.augmentation, seed=train_cfg.seed)
        eval_transform = build_detection_transform(train_cfg.preprocess)
        train_source = DetectionSampleSource(train_ds, parser, predecode_images=train_cfg.loader.predecode_images)
        val_source = None
        if train_cfg.backend_params.get("use_internal_val", False):
            warnings.warn(
                "Ultralytics internal validation is not implemented; validation source was not built.",
                RuntimeWarning,
                stacklevel=2,
            )
        overrides = self._train_overrides(train_cfg)
        checkpoint_state = CheckpointState(train_cfg.output_dir)
        trainer_cls = self._trainer_class(DetectionTrainer)
        trainer = trainer_cls(
            overrides=overrides,
            train_source=train_source,
            val_source=val_source,
            train_transform=train_transform,
            eval_transform=eval_transform,
            loader_cfg=train_cfg.loader,
            classes=train_cfg.classes,
            max_steps=train_cfg.max_steps,
            logger=logger,
            log_prefix=log_prefix,
            logging_steps=train_cfg.logging_steps,
            stop_after_epochs=int(train_cfg.max_epochs or train_cfg.scheduler.total_epochs),
            gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
            optimizer_cfg=train_cfg.optimizer,
            scheduler_cfg=train_cfg.scheduler,
            checkpoint_state=checkpoint_state,
            trial_final_only=(
                epoch_eval_cfg is None
                and not train_cfg.checkpoint.keep_all_epoch_checkpoints
            ),
            adapter=self,
            val_dataset=val_ds,
            epoch_eval_cfg=epoch_eval_cfg,
            train_cfg=train_cfg,
        )
        trainer.train()

        if logger is not None:
            logger.log_metrics(
                {f"{log_prefix}/optimizer_steps": trainer._protocol_optimizer_steps},
                step=trainer._protocol_optimizer_steps,
            )

        return self._artifact_from_trainer(trainer, train_cfg, checkpoint_state=checkpoint_state)

    def predict(
        self,
        ds: Dataset,
        artifact: ModelArtifact,
        predict_cfg: PredictConfig,
    ) -> Iterator[PredictionRecord]:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("Install the models extra to use backend='ultralytics'.") from exc

        checkpoint = artifact.checkpoint_path or artifact.artifact_path or Path(str(self.cfg.model_name_or_path))
        model = YOLO(str(checkpoint))
        device = predict_cfg.backend_params.get("device")
        parser = HFDetectionRowParser(
            classes=predict_cfg.classes,
            label_mode=predict_cfg.label_mode,
            decode_backend=predict_cfg.backend_params.get("decode_backend", "pil"),
        )
        transform = build_detection_transform(predict_cfg.preprocess)

        for rows in iter_hf_row_batches(ds, predict_cfg.batch_size):
            originals = [parser.parse(row) for row in rows]
            samples = [transform(sample) for sample in originals]
            results = model.predict(
                source=[sample.image for sample in samples],
                imgsz=predict_cfg.preprocess.image_size,
                conf=predict_cfg.conf_threshold,
                iou=predict_cfg.iou_threshold,
                max_det=predict_cfg.max_detections_per_image,
                device=device,
                verbose=False,
                save=False,
            )

            for original, sample, result in zip(originals, samples, results):
                preprocess = sample.meta.get("preprocess")
                predictions: list[PredictionObject] = []
                invalid_prediction_boxes_dropped = 0
                boxes = result.boxes
                if boxes is not None:
                    for xyxy, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
                        class_idx = int(cls.detach().cpu())
                        if class_idx < 0 or class_idx >= len(predict_cfg.classes):
                            continue
                        bbox = canonicalize_prediction_bbox(
                            xyxy.detach().cpu().tolist(),
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
                                score=float(conf.detach().cpu()),
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

    def _train_overrides(self, train_cfg: TrainConfig) -> dict:
        hparams = train_cfg.hparams
        project = train_cfg.output_dir.parent if train_cfg.output_dir.parent != Path("") else Path(".")
        overrides = {
            "task": "detect",
            "mode": "train",
            "model": str(self.cfg.weights or self.cfg.model_name_or_path),
            "data": "hf://obj-det",
            "project": str(project),
            "name": train_cfg.output_dir.name,
            "exist_ok": True,
            "epochs": int(train_cfg.scheduler.total_epochs),
            "imgsz": int(train_cfg.preprocess.image_size),
            "batch": int(train_cfg.batch_size),
            "seed": int(train_cfg.seed),
            "amp": bool(train_cfg.amp),
            "workers": int(train_cfg.loader.num_workers),
            "val": False,
            "plots": False,
            "save": bool(train_cfg.checkpoint.keep_all_epoch_checkpoints),
            "save_period": (
                int(train_cfg.checkpoint.save_every_epochs)
                if train_cfg.checkpoint.keep_all_epoch_checkpoints
                else -1
            ),
            "device": train_cfg.backend_params.get("device"),
            "optimizer": "AdamW",
            "lr0": float(hparams.get("learning_rate", 0.01)),
            "lrf": float(train_cfg.scheduler.min_lr_ratio),
            "weight_decay": float(train_cfg.optimizer.weight_decay),
            "momentum": float(train_cfg.optimizer.beta1),
            "warmup_epochs": 0.0,
            "cos_lr": False,
            "patience": 0,
            "nbs": int(train_cfg.batch_size * train_cfg.gradient_accumulation_steps),
        }
        overrides.update(train_cfg.backend_params.get("overrides", {}))
        overrides.update(
            {
                "epochs": int(train_cfg.scheduler.total_epochs),
                "optimizer": "AdamW",
                "lr0": float(hparams.get("learning_rate", 0.01)),
                "lrf": float(train_cfg.scheduler.min_lr_ratio),
                "weight_decay": float(train_cfg.optimizer.weight_decay),
                "momentum": float(train_cfg.optimizer.beta1),
                "warmup_epochs": 0.0,
                "cos_lr": False,
                "patience": 0,
                "nbs": int(train_cfg.batch_size * train_cfg.gradient_accumulation_steps),
            }
        )
        overrides.update(_CONTROLLED_AUG_OFF)
        return overrides

    def _validate_single_process_device(self, train_cfg: TrainConfig) -> None:
        device = train_cfg.backend_params.get("device")
        if isinstance(device, (list, tuple)):
            multi_device = len(device) > 1
        elif isinstance(device, str):
            normalized = device.strip().strip("[]")
            multi_device = len([item for item in normalized.split(",") if item.strip()]) > 1
        else:
            multi_device = False
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if multi_device or world_size > 1:
            raise NotImplementedError(
                "The Ultralytics HF-backed adapter requires a single training process; "
                "multi-device Ultralytics would lose its in-memory dataset and shared epoch controller"
            )

    def _artifact_from_trainer(
        self,
        trainer,
        train_cfg: TrainConfig,
        *,
        checkpoint_state: CheckpointState | None = None,
    ) -> ModelArtifact:
        last_value = getattr(trainer, "last", None)
        best_value = getattr(trainer, "best", None)
        last = Path(last_value) if last_value else None
        best = Path(best_value) if best_value else None
        selected = None
        if checkpoint_state is not None and train_cfg.early_stopping.restore_best:
            selected = checkpoint_state.best_checkpoint
        if selected is None and checkpoint_state is not None:
            selected = checkpoint_state.last_checkpoint
        selected_is_best = checkpoint_state is not None and checkpoint_state.best_checkpoint is not None and selected == checkpoint_state.best_checkpoint
        checkpoint_path = selected or (last if last is not None and last.exists() else None)
        optimizer_meta = {
            **train_cfg.optimizer.model_dump(mode="json"),
            "learning_rate": float(train_cfg.hparams.get("learning_rate", 0.01)),
        }
        scheduler_meta = train_cfg.scheduler.model_dump(mode="json")
        meta = {
            "protocol": train_cfg.protocol,
            "ultralytics_args": dict(vars(trainer.args)),
            "ultralytics_last": str(last) if last is not None else None,
            "ultralytics_best": str(best) if best is not None else None,
            "checkpoint_selection": "best_validation" if selected_is_best else "last",
            "optimizer_steps": int(getattr(trainer, "_protocol_optimizer_steps", 0)),
            "pretrained_source": str(self.cfg.weights or self.cfg.model_name_or_path),
            "optimizer": optimizer_meta,
            "scheduler": scheduler_meta,
        }
        if checkpoint_state is not None:
            meta.update(checkpoint_state.artifact_meta())
        return ModelArtifact(
            model_key=self.key,
            backend=self.backend,
            run_key=train_cfg.run_key,
            classes=train_cfg.classes,
            label_mode=train_cfg.label_mode,
            artifact_path=Path(trainer.save_dir),
            checkpoint_path=checkpoint_path,
            best_metric_name=(
                train_cfg.early_stopping.metric
                if checkpoint_state is not None
                and checkpoint_state.best_epoch is not None
                else None
            ),
            best_metric_value=(
                checkpoint_state.best_metric
                if checkpoint_state is not None
                and checkpoint_state.best_epoch is not None
                else None
            ),
            meta=meta,
        )

    def _trainer_class(self, detection_trainer_cls):
        class HFBackedDetectionTrainer(detection_trainer_cls):
            def __init__(
                self,
                *args,
                train_source,
                val_source,
                train_transform,
                eval_transform,
                loader_cfg,
                classes,
                max_steps,
                logger,
                log_prefix,
                logging_steps,
                stop_after_epochs=1,
                gradient_accumulation_steps=1,
                optimizer_cfg=None,
                scheduler_cfg=None,
                checkpoint_state=None,
                trial_final_only=False,
                adapter=None,
                val_dataset=None,
                epoch_eval_cfg=None,
                train_cfg=None,
                **kwargs,
            ):
                self._hf_train_source = train_source
                self._hf_val_source = val_source
                self._hf_train_transform = train_transform
                self._hf_eval_transform = eval_transform
                self._hf_loader_cfg = loader_cfg
                self._hf_classes = classes
                self._hf_max_steps = max_steps
                self._hf_seen_steps = 0
                self._hf_last_logged_step = 0
                self._hf_logger = logger
                self._hf_log_prefix = log_prefix
                self._hf_logging_steps = logging_steps
                self._protocol_stop_after_epochs = stop_after_epochs
                self._protocol_optimizer_cfg = optimizer_cfg
                self._protocol_scheduler_cfg = scheduler_cfg
                self._protocol_checkpoint_state = checkpoint_state
                self._protocol_trial_final_only = trial_final_only
                self._protocol_adapter = adapter
                self._protocol_val_dataset = val_dataset
                self._protocol_epoch_eval_cfg = epoch_eval_cfg
                self._protocol_train_cfg = train_cfg
                self._protocol_scheduler = None
                self._protocol_optimizer_steps = 0
                self._protocol_nominal_accumulate = gradient_accumulation_steps
                self._protocol_epoch_batches = 0
                self._protocol_last_eval_epoch = 0
                self._protocol_stop_at_epoch_end = False
                super().__init__(*args, **kwargs)

            def run_callbacks(self, event: str):
                super().run_callbacks(event)
                if event == "on_train_epoch_start":
                    self._protocol_epoch_batches = 0
                    self.accumulate = self._protocol_nominal_accumulate
                elif event == "on_train_batch_start":
                    current_batch = self._protocol_epoch_batches + 1
                    if current_batch == len(self.train_loader):
                        remainder = len(self.train_loader) % self._protocol_nominal_accumulate
                        if remainder:
                            self.accumulate = remainder
                elif event == "on_train_batch_end":
                    self._protocol_epoch_batches += 1
                    self._hf_seen_steps += 1
                    if self._hf_seen_steps % self._hf_logging_steps == 0:
                        self._log_step_metrics()
                    if self._hf_max_steps is None:
                        return
                    if self._hf_seen_steps >= self._hf_max_steps:
                        self.stop = True
                elif event == "on_train_end":
                    self._log_step_metrics(force=True)
                elif event == "on_train_epoch_end":
                    if self.epoch + 1 >= self._protocol_stop_after_epochs:
                        self._protocol_stop_at_epoch_end = True
                elif event == "on_fit_epoch_end":
                    self._protocol_epoch_end()

            def _log_step_metrics(self, *, force: bool = False):
                if self._hf_logger is None:
                    return
                if self._hf_seen_steps <= 0:
                    return
                if force and self._hf_last_logged_step == self._hf_seen_steps:
                    return

                row = {"step": self._hf_seen_steps}
                if getattr(self, "tloss", None) is not None:
                    row.update(self.label_loss_items(self.tloss, prefix=self._hf_log_prefix))
                optimizer = getattr(self, "optimizer", None)
                if optimizer is not None:
                    row.update(
                        {f"lr/pg{idx}": group["lr"] for idx, group in enumerate(optimizer.param_groups) if "lr" in group}
                    )

                metrics, step = flatten_prefixed_scalar_mapping(self._hf_log_prefix, row)
                if metrics:
                    self._hf_logger.log_metrics(metrics, step=step if step is not None else self._hf_seen_steps)
                    self._hf_last_logged_step = self._hf_seen_steps

            def get_dataset(self):
                return {
                    "train": "hf_train",
                    "val": "hf_val",
                    "nc": len(self._hf_classes),
                    "names": {idx: name for idx, name in enumerate(self._hf_classes)},
                    "channels": 3,
                }

            def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
                if mode == "train":
                    source = self._hf_train_source
                    transform = self._hf_train_transform
                else:
                    source = self._hf_val_source
                    transform = self._hf_eval_transform
                    if source is None:
                        return DataLoader(
                            [],
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=ultralytics_detection_collate,
                            **dataloader_kwargs(self._hf_loader_cfg),
                        )

                dataset = HFUltralyticsDetectionDataset(
                    source,
                    transform,
                    include_samples=bool(getattr(self._hf_loader_cfg, "include_samples_in_batch", False)),
                    profile_every_n=getattr(self._hf_loader_cfg, "profile_every_n", None),
                )
                return DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=mode == "train",
                    collate_fn=ultralytics_detection_collate,
                    **dataloader_kwargs(self._hf_loader_cfg),
                )

            def get_validator(self):
                return _NoOpUltralyticsValidator()

            def validate(self):
                fitness = -float(self.loss.detach().cpu()) if getattr(self, "loss", None) is not None else 0.0
                if not self.best_fitness or self.best_fitness < fitness:
                    self.best_fitness = fitness
                return {}, fitness

            def build_optimizer(self, model, name="AdamW", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
                cfg = self._protocol_optimizer_cfg
                groups = build_adamw_param_groups(model, weight_decay=cfg.weight_decay)
                return torch.optim.AdamW(
                    groups,
                    lr=float(lr),
                    betas=(cfg.beta1, cfg.beta2),
                    eps=cfg.epsilon,
                )

            def _setup_scheduler(self):
                self._protocol_nominal_accumulate = int(self.accumulate)
                steps_per_epoch = optimizer_steps_per_epoch(
                    len(self.train_loader),
                    self._protocol_nominal_accumulate,
                )
                cfg = self._protocol_scheduler_cfg
                self._protocol_scheduler = build_warmup_cosine_scheduler(
                    self.optimizer,
                    warmup_steps=int(round(cfg.warmup_epochs * steps_per_epoch)),
                    total_steps=int(cfg.total_epochs * steps_per_epoch),
                    min_lr_ratio=cfg.min_lr_ratio,
                )
                self.scheduler = _NoOpEpochScheduler(self._protocol_scheduler)

            def optimizer_step(self):
                scaler = getattr(self, "scaler", None)
                scale_before = scaler.get_scale() if scaler is not None else None
                super().optimizer_step()
                if (
                    scale_before is not None
                    and scaler.get_scale() < scale_before
                ):
                    return
                self._protocol_optimizer_steps += 1
                if self._protocol_scheduler is not None:
                    self._protocol_scheduler.step()

            def _protocol_epoch_end(self):
                epoch = int(self.epoch) + 1
                if epoch <= self._protocol_last_eval_epoch:
                    return

                needs_manual_save = self._protocol_stop_at_epoch_end or self.stop
                if self._protocol_epoch_eval_cfg is None and not needs_manual_save:
                    return
                if needs_manual_save and (not self.last.exists() or not self.args.save):
                    if self._protocol_trial_final_only:
                        self._save_trial_final_checkpoint()
                    else:
                        self.save_model()

                epoch_checkpoint = self.wdir / f"epoch{self.epoch}.pt"
                checkpoint_path = epoch_checkpoint if epoch_checkpoint.exists() else self.last
                if not checkpoint_path.exists():
                    raise FileNotFoundError(
                        f"Ultralytics epoch {epoch} checkpoint was not created: {checkpoint_path}"
                    )
                metric = None
                if self._protocol_epoch_eval_cfg is not None:
                    artifact = self._protocol_adapter._artifact_for_checkpoint(
                        self._protocol_train_cfg,
                        checkpoint_path,
                        artifact_path=Path(self.save_dir),
                    )
                    result = self._protocol_adapter.evaluate(
                        self._protocol_val_dataset,
                        artifact,
                        self._protocol_epoch_eval_cfg,
                        logger=self._hf_logger,
                        log_prefix="val/epoch",
                    )
                    metric = require_metric(
                        result.metrics,
                        self._protocol_train_cfg.early_stopping.metric,
                        context="early-stopping",
                    )

                should_stop = self._protocol_checkpoint_state.record_epoch(
                    epoch=epoch,
                    checkpoint_path=checkpoint_path,
                    metric=metric,
                    early_stopping_cfg=self._protocol_train_cfg.early_stopping,
                )
                self._protocol_last_eval_epoch = epoch
                if self._hf_logger is not None:
                    self._hf_logger.log_artifact(checkpoint_path, name=f"checkpoint_epoch_{epoch:03d}")
                    if metric is not None:
                        self._hf_logger.log_metrics(
                            {
                                "early_stopping/bad_epochs": self._protocol_checkpoint_state.early_stopping.bad_epochs,
                                "early_stopping/best_metric": self._protocol_checkpoint_state.early_stopping.best_metric,
                                "early_stopping/best_epoch": self._protocol_checkpoint_state.early_stopping.best_epoch,
                            },
                            step=self._protocol_optimizer_steps,
                        )
                if should_stop or self._protocol_stop_at_epoch_end:
                    self.stop = True

            def _save_trial_final_checkpoint(self):
                original_best = self.best
                self.best = self.last
                try:
                    return self.save_model()
                finally:
                    self.best = original_best

            def final_eval(self):
                return None

        return HFBackedDetectionTrainer

    def _artifact_for_checkpoint(
        self,
        train_cfg: TrainConfig,
        checkpoint_path: Path,
        *,
        artifact_path: Path,
    ) -> ModelArtifact:
        return ModelArtifact(
            model_key=self.key,
            backend=self.backend,
            run_key=train_cfg.run_key,
            classes=train_cfg.classes,
            label_mode=train_cfg.label_mode,
            artifact_path=artifact_path,
            checkpoint_path=checkpoint_path,
        )


class _NoOpEpochScheduler:
    """Ignore Ultralytics' epoch-level scheduler step; the shared scheduler advances per optimizer update."""

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def step(self, *args, **kwargs):
        return None

    @property
    def last_epoch(self):
        return self.scheduler.last_epoch

    @last_epoch.setter
    def last_epoch(self, value):
        # Ultralytics resets its epoch scheduler to ``start_epoch - 1`` after
        # construction. The shared scheduler is optimizer-step based and has
        # already applied factor(0), so forwarding that reset would make the
        # first two optimizer updates use zero LR instead of only update zero.
        return None

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        return self.scheduler.load_state_dict(state_dict)

    def get_last_lr(self):
        return self.scheduler.get_last_lr()


class _NoOpUltralyticsValidator:
    def __init__(self):
        self.metrics = SimpleNamespace(keys=[])
        self.args = SimpleNamespace(plots=False, compile=False)

    def __call__(self, *args, **kwargs):
        return {"fitness": 0.0}
