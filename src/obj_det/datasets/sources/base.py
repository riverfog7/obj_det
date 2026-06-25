from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator, Sequence

from obj_det.datasets.models import BBox, ImageRecord, ObjectAnnotation
from obj_det.datasets.models.source_config import SourceDatasetConfig, SourceSplitConfig


class BaseSourceDataset(ABC):
    """
    Base interface for importing arbitrary raw object-detection datasets.

    This is not a torch Dataset.
    This is not a Hugging Face Dataset.
    This is only a raw-source -> canonical ImageRecord adapter.
    """

    def __init__(self, cfg: SourceDatasetConfig):
        self.cfg = cfg
        self.key = cfg.key

    # ------------------------------------------------------------------
    # Required implementation
    # ------------------------------------------------------------------

    @abstractmethod
    def _iter_records(self, split: str) -> Iterator[ImageRecord]:
        """
        Yield canonical ImageRecord objects for one split.

        Subclasses parse whatever raw format they need:
            folders
            JSON
            TXT
            XML
            CSV
            mixed files
            generated synthetic data
            Hugging Face dataset, if it works

        The only required output is ImageRecord.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def splits(self) -> list[str]:
        return list(self.cfg.splits.keys())

    def iter_records(
        self,
        split: str,
        *,
        validate_geometry: bool = True,
    ) -> Iterator[ImageRecord]:
        """
        Public record iterator.

        This wraps the adapter implementation with split checking
        and optional geometry validation.
        """
        self.require_split(split)

        for record in self._iter_records(split):
            self.validate_record_identity(record, split)

            if validate_geometry:
                record.assert_valid_geometry()

            yield record

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def require_split(self, split: str) -> None:
        if split not in self.cfg.splits:
            raise KeyError(
                f"Unknown split '{split}' for dataset '{self.key}'. "
                f"Available splits: {self.splits}"
            )

    def split_cfg(self, split: str) -> SourceSplitConfig:
        self.require_split(split)
        return self.cfg.splits[split]

    def resolve_path(self, path: Path) -> Path:
        """
        Resolve a path relative to dataset root unless it is already absolute.
        """
        if path.is_absolute():
            return path
        return self.cfg.root / path

    def path(self, split: str, key: str) -> Path:
        """
        Get a named split path.

        Example:
            self.path(split, "images")
            self.path(split, "annotations")
            self.path(split, "labels")
        """
        split_cfg = self.split_cfg(split)

        if key not in split_cfg.paths:
            raise KeyError(
                f"Missing path key '{key}' for dataset='{self.key}', split='{split}'. "
                f"Available path keys: {list(split_cfg.paths.keys())}"
            )

        return self.resolve_path(split_cfg.paths[key])

    def verify_paths(
        self,
        split: str | None = None,
        *,
        keys: Sequence[str] | None = None,
    ) -> None:
        """
        Optional filesystem validation.

        The adapter may call this with only the keys it actually needs.
        """
        splits = [split] if split is not None else self.splits

        for split_name in splits:
            split_cfg = self.split_cfg(split_name)

            if keys is None:
                path_items = list(split_cfg.paths.items())
            else:
                missing_keys = [key for key in keys if key not in split_cfg.paths]
                if missing_keys:
                    raise KeyError(
                        f"Missing path keys {missing_keys} for dataset='{self.key}', "
                        f"split='{split_name}'. Available path keys: "
                        f"{list(split_cfg.paths.keys())}"
                    )

                path_items = [(key, split_cfg.paths[key]) for key in keys]

            for key, raw_path in path_items:
                resolved = self.resolve_path(raw_path)
                if not resolved.exists():
                    raise FileNotFoundError(
                        f"Missing source path for dataset='{self.key}', "
                        f"split='{split_name}', key='{key}': {resolved}"
                    )

    # ------------------------------------------------------------------
    # Bbox helpers
    # ------------------------------------------------------------------

    def make_bbox_xywh(
        self,
        xywh: Sequence[float],
        *,
        image_width: int,
        image_height: int,
    ) -> BBox | None:
        """
        Create canonical BBox using configured bbox policy.

        xywh format:
            x, y, width, height
        absolute pixels.
        """
        try:
            bbox = BBox.from_xywh(list(xywh))
        except ValueError:
            if self.cfg.bbox_policy == "drop":
                return None
            raise

        if bbox.within_image(image_width, image_height):
            return bbox

        if self.cfg.bbox_policy == "strict":
            raise ValueError(
                f"Bbox {bbox.xywh()} exceeds image bounds {image_width}x{image_height}"
            )

        if self.cfg.bbox_policy == "drop":
            return None

        if self.cfg.bbox_policy == "clip":
            return bbox.clipped(image_width, image_height)

        raise ValueError(f"Unknown bbox_policy: {self.cfg.bbox_policy}")

    # ------------------------------------------------------------------
    # Object / record helpers
    # ------------------------------------------------------------------

    def make_object(
        self,
        *,
        bbox_xywh: Sequence[float],
        image_width: int,
        image_height: int,
        native_label: str,
        native_label_id: int | str | None = None,
        ignore: bool = False,
        iscrowd: bool = False,
        meta: dict[str, Any] | None = None,
    ) -> ObjectAnnotation | None:
        native_label = native_label.strip()

        if native_label in self.cfg.ignore_labels:
            return None

        bbox = self.make_bbox_xywh(
            bbox_xywh,
            image_width=image_width,
            image_height=image_height,
        )

        if bbox is None:
            return None

        return ObjectAnnotation(
            bbox=bbox,
            native_label=native_label,
            native_label_id=native_label_id,
            meta_label=self.cfg.class_map.get(native_label),
            ignore=ignore,
            iscrowd=iscrowd,
            meta=meta or {},
        )

    def make_record(
        self,
        *,
        split: str,
        source_id: str | int,
        image_path: Path,
        width: int,
        height: int,
        objects: list[ObjectAnnotation],
        condition: str | None = None,
        domain: str | None = None,
        is_synthetic: bool | None = None,
        meta: dict[str, Any] | None = None,
    ) -> ImageRecord:
        """
        Convenience constructor for adapters.

        Adapters can override condition/domain/is_synthetic per image if needed.
        Otherwise split/default config values are used.
        """
        split_cfg = self.split_cfg(split)

        record_meta = dict(self.cfg.meta)
        record_meta.update(split_cfg.meta)
        if meta:
            record_meta.update(meta)

        if is_synthetic is None:
            is_synthetic = (
                split_cfg.is_synthetic
                if split_cfg.is_synthetic is not None
                else self.cfg.default_is_synthetic
            )

        return ImageRecord(
            image_id=f"{self.key}:{split}:{source_id}",
            dataset=self.key,
            split=split,
            image_path=self.resolve_path(image_path),
            width=width,
            height=height,
            objects=objects,
            condition=condition or split_cfg.condition or self.cfg.default_condition,
            domain=domain or split_cfg.domain or self.cfg.default_domain,
            is_synthetic=is_synthetic,
            meta=record_meta,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_record_identity(self, record: ImageRecord, split: str) -> None:
        if record.dataset != self.key:
            raise ValueError(
                f"Record dataset mismatch: expected '{self.key}', got '{record.dataset}'"
            )

        if record.split != split:
            raise ValueError(f"Record split mismatch: expected '{split}', got '{record.split}'")
