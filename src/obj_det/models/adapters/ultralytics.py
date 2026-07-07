from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

from datasets import Dataset
from torch.utils.data import DataLoader

from obj_det.datasets.models import BBox
from obj_det.models.adapters.base import BaseModelAdapter
from obj_det.models.data.loader import dataloader_kwargs
from obj_det.models.data.row_parser import HFDetectionRowParser
from obj_det.models.data.row_batches import iter_hf_row_batches
from obj_det.models.data.sample_source import DetectionSampleSource
from obj_det.models.data.ultralytics_dataset import HFUltralyticsDetectionDataset, ultralytics_detection_collate
from obj_det.models.data.transforms import bbox_to_original, build_detection_transform
from obj_det.models.logging.base import BaseExperimentLogger
from obj_det.models.logging.metrics import flatten_prefixed_scalar_mapping
from obj_det.models.schemas.artifact import ModelArtifact
from obj_det.models.schemas.config import PredictConfig, TrainConfig
from obj_det.models.schemas.prediction import PredictionObject, PredictionRecord
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
        logger: BaseExperimentLogger | None = None,
        log_prefix: str = "train",
    ) -> ModelArtifact:
        try:
            from ultralytics.models.yolo.detect import DetectionTrainer
        except ImportError as exc:
            raise ImportError("Install the models extra to use backend='ultralytics'.") from exc

        if train_cfg.eval_strategy.enabled:
            warnings.warn(
                "Ultralytics train-time eval_strategy is ignored; final metrics come from adapter.evaluate().",
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
        )
        trainer.train()

        return self._artifact_from_trainer(trainer, train_cfg)

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
                device=device,
                verbose=False,
                save=False,
            )

            for original, sample, result in zip(originals, samples, results):
                preprocess = sample.meta.get("preprocess")
                predictions: list[PredictionObject] = []
                boxes = result.boxes
                if boxes is not None:
                    for xyxy, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
                        class_idx = int(cls.detach().cpu())
                        if class_idx < 0 or class_idx >= len(predict_cfg.classes):
                            continue
                        bbox = BBox.from_xyxy([float(value) for value in xyxy.detach().cpu().tolist()])
                        if preprocess is not None:
                            bbox = bbox_to_original(bbox, preprocess)
                        if bbox is None:
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
                )

    def _train_overrides(self, train_cfg: TrainConfig) -> dict:
        hparams = train_cfg.hparams
        project = train_cfg.output_dir.parent if train_cfg.output_dir.parent != Path("") else Path(".")
        overrides = {
            "task": "detect",
            "mode": "train",
            "model": str(self.cfg.model_name_or_path),
            "data": "hf://obj-det",
            "project": str(project),
            "name": train_cfg.output_dir.name,
            "exist_ok": True,
            "epochs": int(train_cfg.max_epochs or 1),
            "imgsz": int(train_cfg.preprocess.image_size),
            "batch": int(train_cfg.batch_size),
            "seed": int(train_cfg.seed),
            "amp": bool(train_cfg.amp),
            "workers": int(train_cfg.loader.num_workers),
            "val": False,
            "plots": False,
            "save": True,
            "device": train_cfg.backend_params.get("device"),
            "optimizer": hparams.get("optimizer", train_cfg.backend_params.get("optimizer", "auto")),
            "lr0": float(hparams.get("lr0", hparams.get("learning_rate", 0.01))),
            "weight_decay": float(hparams.get("weight_decay", 0.0005)),
            "momentum": float(hparams.get("momentum", 0.937)),
            "warmup_epochs": float(hparams.get("warmup_epochs", 3.0)),
            "cos_lr": bool(hparams.get("cos_lr", False)),
        }
        if train_cfg.max_steps is not None:
            overrides["epochs"] = max(1, int(train_cfg.max_epochs or 1))
        if train_cfg.protocol in {"controlled", "equal_hpo"}:
            overrides.update(_CONTROLLED_AUG_OFF)
        overrides.update(train_cfg.backend_params.get("overrides", {}))
        return overrides

    def _artifact_from_trainer(self, trainer, train_cfg: TrainConfig) -> ModelArtifact:
        last_value = getattr(trainer, "last", None)
        best_value = getattr(trainer, "best", None)
        last = Path(last_value) if last_value else None
        best = Path(best_value) if best_value else None
        checkpoint_path = last if last is not None and last.exists() else None
        meta = {
            "ultralytics_args": dict(vars(trainer.args)),
            "ultralytics_last": str(last) if last is not None else None,
            "ultralytics_best": str(best) if best is not None else None,
            "checkpoint_selection": "last_external_eval",
        }
        return ModelArtifact(
            model_key=self.key,
            backend=self.backend,
            run_key=train_cfg.run_key,
            classes=train_cfg.classes,
            label_mode=train_cfg.label_mode,
            artifact_path=Path(trainer.save_dir),
            checkpoint_path=checkpoint_path,
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
                super().__init__(*args, **kwargs)

            def run_callbacks(self, event: str):
                super().run_callbacks(event)
                if event == "on_train_batch_end":
                    self._hf_seen_steps += 1
                    if self._hf_seen_steps % self._hf_logging_steps == 0:
                        self._log_step_metrics()
                    if self._hf_max_steps is None:
                        return
                    if self._hf_seen_steps >= self._hf_max_steps:
                        self.stop = True
                elif event == "on_train_end":
                    self._log_step_metrics(force=True)

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
                        raise RuntimeError("Ultralytics internal validation source is disabled")

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

            def final_eval(self):
                return None

        return HFBackedDetectionTrainer


class _NoOpUltralyticsValidator:
    def __init__(self):
        self.metrics = SimpleNamespace(keys=[])
        self.args = SimpleNamespace(plots=False, compile=False)

    def __call__(self, *args, **kwargs):
        return {"fitness": 0.0}
