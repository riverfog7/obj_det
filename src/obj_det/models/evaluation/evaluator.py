from __future__ import annotations

import contextlib
import io
import logging
from typing import Iterable

from datasets import Dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from obj_det.models.data.bbox import area_xywh
from obj_det.models.data.row_parser import HFDetectionRowParser
from obj_det.models.data.sample import DetectionSample
from obj_det.models.evaluation.grouping import image_ids_by_field
from obj_det.models.schemas.config import EvalConfig
from obj_det.models.schemas.prediction import PredictionRecord
from obj_det.models.schemas.result import EvalResult


logger = logging.getLogger(__name__)


class DetectionEvaluator:
    def evaluate(
        self,
        ds: Dataset,
        predictions: Iterable[PredictionRecord],
        eval_cfg: EvalConfig,
        *,
        model_key: str,
    ) -> EvalResult:
        parser = HFDetectionRowParser(classes=eval_cfg.classes, label_mode=eval_cfg.label_mode)
        samples = [parser.parse_targets_only(row) for row in ds]
        stats = parser.stats_snapshot()
        if stats:
            logger.info("Evaluation object filtering stats: %s", stats)
        pred_list = list(predictions)
        metrics = self._evaluate_subset(samples, pred_list, eval_cfg.classes)

        per_class: dict[str, dict[str, float]] = {}
        if eval_cfg.compute_per_class:
            for idx, class_name in enumerate(eval_cfg.classes):
                per_class[class_name] = self._evaluate_subset(
                    samples,
                    pred_list,
                    eval_cfg.classes,
                    cat_ids=[idx + 1],
                )

        per_condition: dict[str, dict[str, float]] = {}
        if eval_cfg.compute_per_condition:
            for condition, image_ids in image_ids_by_field(samples, "condition").items():
                per_condition[condition] = self._evaluate_subset(
                    samples,
                    pred_list,
                    eval_cfg.classes,
                    image_ids=image_ids,
                )

        per_domain: dict[str, dict[str, float]] = {}
        if eval_cfg.compute_per_domain:
            for domain, image_ids in image_ids_by_field(samples, "domain").items():
                per_domain[domain] = self._evaluate_subset(
                    samples,
                    pred_list,
                    eval_cfg.classes,
                    image_ids=image_ids,
                )

        per_size = self._size_metrics(metrics) if eval_cfg.compute_per_size else {}
        primary_value = metrics.get(eval_cfg.primary_metric, 0.0)

        dataset_key = samples[0].dataset if samples else None
        split = samples[0].split if samples else None

        return EvalResult(
            model_key=model_key,
            dataset_key=dataset_key,
            split=split,
            primary_metric=eval_cfg.primary_metric,
            primary_metric_value=primary_value,
            metrics=metrics,
            per_class=per_class,
            per_condition=per_condition,
            per_domain=per_domain,
            per_size=per_size,
            num_images=len(samples),
            num_ground_truth_objects=sum(len(sample.targets) for sample in samples),
            num_predictions=sum(len(record.predictions) for record in pred_list),
        )

    def _evaluate_subset(
        self,
        samples: list[DetectionSample],
        predictions: list[PredictionRecord],
        classes: list[str],
        *,
        image_ids: set[str] | None = None,
        cat_ids: list[int] | None = None,
    ) -> dict[str, float]:
        selected_samples = [s for s in samples if image_ids is None or s.image_id in image_ids]
        selected_ids = {s.image_id for s in selected_samples}
        selected_predictions = [p for p in predictions if p.image_id in selected_ids]

        gt_count = sum(len(s.targets) for s in selected_samples)
        pred_count = sum(len(p.predictions) for p in selected_predictions)
        if not selected_samples or gt_count == 0 or pred_count == 0:
            return self._zero_metrics()

        image_id_to_int = {sample.image_id: idx + 1 for idx, sample in enumerate(selected_samples)}
        coco_gt = self._build_coco_gt(selected_samples, classes, image_id_to_int)
        coco_preds = self._build_coco_predictions(selected_predictions, classes, image_id_to_int)
        if not coco_preds:
            return self._zero_metrics()

        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = coco_gt.loadRes(coco_preds)
            evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
            evaluator.params.imgIds = list(image_id_to_int.values())
            if cat_ids is not None:
                evaluator.params.catIds = cat_ids
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()

        return self._stats_to_metrics(evaluator.stats)

    def _build_coco_gt(
        self,
        samples: list[DetectionSample],
        classes: list[str],
        image_id_to_int: dict[str, int],
    ) -> COCO:
        annotations = []
        ann_id = 1
        for sample in samples:
            for target in sample.targets:
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id_to_int[sample.image_id],
                        "category_id": target.label_id + 1,
                        "bbox": list(target.bbox_xywh),
                        "area": area_xywh(target.bbox_xywh),
                        "iscrowd": int(target.iscrowd),
                    }
                )
                ann_id += 1

        coco = COCO()
        coco.dataset = {
            "info": {},
            "licenses": [],
            "images": [
                {"id": image_id_to_int[s.image_id], "width": s.width, "height": s.height}
                for s in samples
            ],
            "annotations": annotations,
            "categories": [{"id": idx + 1, "name": name} for idx, name in enumerate(classes)],
        }
        with contextlib.redirect_stdout(io.StringIO()):
            coco.createIndex()
        return coco

    def _build_coco_predictions(
        self,
        predictions: list[PredictionRecord],
        classes: list[str],
        image_id_to_int: dict[str, int],
    ) -> list[dict]:
        class_to_id = {name: idx + 1 for idx, name in enumerate(classes)}
        rows = []
        for record in predictions:
            if record.image_id not in image_id_to_int:
                continue
            for pred in record.predictions:
                category_id = class_to_id.get(pred.label)
                if category_id is None:
                    continue
                rows.append(
                    {
                        "image_id": image_id_to_int[record.image_id],
                        "category_id": category_id,
                        "bbox": list(pred.bbox.xywh()),
                        "score": pred.score,
                    }
                )
        return rows

    def _stats_to_metrics(self, stats) -> dict[str, float]:
        names = [
            "map_50_95",
            "map_50",
            "map_75",
            "ap_small",
            "ap_medium",
            "ap_large",
            "ar_1",
            "ar_10",
            "ar_100",
            "ar_small",
            "ar_medium",
            "ar_large",
        ]
        return {name: float(value) for name, value in zip(names, stats) if value >= 0}

    def _zero_metrics(self) -> dict[str, float]:
        return {
            "map_50_95": 0.0,
            "map_50": 0.0,
            "map_75": 0.0,
            "ar_1": 0.0,
            "ar_10": 0.0,
            "ar_100": 0.0,
        }

    def _size_metrics(self, metrics: dict[str, float]) -> dict[str, dict[str, float]]:
        groups: dict[str, dict[str, float]] = {}
        mapping = {
            "small": ("ap_small", "ar_small"),
            "medium": ("ap_medium", "ar_medium"),
            "large": ("ap_large", "ar_large"),
        }
        for name, (ap_key, ar_key) in mapping.items():
            group = {}
            if ap_key in metrics:
                group["ap"] = metrics[ap_key]
            if ar_key in metrics:
                group["ar_100"] = metrics[ar_key]
            if group:
                groups[name] = group
        return groups
