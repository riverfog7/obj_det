from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from datasets import Dataset

from obj_det.datasets.models import BBox
from obj_det.models.adapters.base import BaseModelAdapter
from obj_det.models.data.hf_dataset import HFTrainerDetectionDataset
from obj_det.models.data.hf_targets import hf_detection_collate
from obj_det.models.data.row_parser import HFDetectionRowParser
from obj_det.models.data.sample_source import DetectionSampleSource
from obj_det.models.data.transforms import bbox_to_original, build_detection_transform
from obj_det.models.logging.base import BaseExperimentLogger
from obj_det.models.logging.trainer_callback import make_transformers_logging_callback
from obj_det.models.schemas.artifact import ModelArtifact
from obj_det.models.schemas.config import PredictConfig, TrainConfig
from obj_det.models.schemas.prediction import PredictionObject, PredictionRecord
from obj_det.models.utils.repro import set_seed


class HFTrainerDetectionAdapter(BaseModelAdapter):
    """Transformers Trainer backend for HF Dataset object-detection rows."""

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

        parser = HFDetectionRowParser(classes=train_cfg.classes, label_mode=train_cfg.label_mode)
        transform = build_detection_transform(train_cfg.preprocess, train_cfg.augmentation, seed=train_cfg.seed)
        processor_kwargs = train_cfg.backend_params.get("processor_kwargs", {"do_resize": False})
        train_source = DetectionSampleSource(train_ds, parser, predecode_images=train_cfg.loader.predecode_images)
        val_source = DetectionSampleSource(val_ds, parser, predecode_images=train_cfg.loader.predecode_images)
        train_data = HFTrainerDetectionDataset(train_source, transform, processor, processor_kwargs)
        val_data = HFTrainerDetectionDataset(val_source, transform, processor, processor_kwargs)

        args = self._training_args(train_cfg)
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_data,
            eval_dataset=val_data,
            data_collator=hf_detection_collate,
            processing_class=processor,
            callbacks=[make_transformers_logging_callback(TrainerCallback, logger, log_prefix)] if logger else None,
        )
        trainer.train()

        artifact_path = train_cfg.output_dir
        checkpoint_path = artifact_path / "final"
        trainer.save_model(str(checkpoint_path))
        processor.save_pretrained(str(checkpoint_path))

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
            artifact_path=artifact_path,
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

        parser = HFDetectionRowParser(classes=predict_cfg.classes, label_mode=predict_cfg.label_mode)
        transform = build_detection_transform(predict_cfg.preprocess)
        processor_kwargs = predict_cfg.backend_params.get("processor_kwargs", {"do_resize": False})
        rows = list(ds)

        with torch.no_grad():
            for start in range(0, len(rows), predict_cfg.batch_size):
                original_samples = [parser.parse(row) for row in rows[start : start + predict_cfg.batch_size]]
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
                    for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
                        class_idx = int(label.detach().cpu())
                        if class_idx < 0 or class_idx >= len(predict_cfg.classes):
                            continue
                        bbox = BBox.from_xyxy([float(value) for value in box.detach().cpu().tolist()])
                        if preprocess is not None:
                            bbox = bbox_to_original(bbox, preprocess)
                        if bbox is None:
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
                    )

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
        eval_args = self._eval_strategy_args(train_cfg)
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
            learning_rate=float(hparams.get("learning_rate", 5e-5)),
            weight_decay=float(hparams.get("weight_decay", 0.0)),
            warmup_ratio=float(hparams.get("warmup_ratio", 0.0)),
            lr_scheduler_type=hparams.get("lr_scheduler_type", "linear"),
            max_grad_norm=float(hparams.get("max_grad_norm", 1.0)),
            seed=train_cfg.seed,
            fp16=bool(train_cfg.amp and torch.cuda.is_available()),
            eval_strategy=eval_args["eval_strategy"],
            save_strategy="epoch" if max_steps < 0 else "no",
            logging_strategy="steps",
            logging_steps=train_cfg.logging_steps,
            report_to=[],
            remove_unused_columns=False,
            load_best_model_at_end=False,
            **loader_args,
            **backend_args,
        )

    def _eval_strategy_args(self, train_cfg: TrainConfig) -> dict[str, str]:
        if not train_cfg.eval_strategy.enabled:
            return {"eval_strategy": "no"}

        if train_cfg.eval_strategy.every_epochs != 1:
            raise NotImplementedError("HF Trainer eval_strategy currently supports every_epochs=1 only")

        return {"eval_strategy": "epoch"}

    def _label_maps(self, classes: list[str]) -> tuple[dict[int, str], dict[str, int]]:
        id2label = {idx: label for idx, label in enumerate(classes)}
        label2id = {label: idx for idx, label in enumerate(classes)}
        return id2label, label2id
