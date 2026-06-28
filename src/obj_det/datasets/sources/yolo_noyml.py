from __future__ import annotations

from .yolo import YoloSourceDataset


class YoloNoYamlSourceDataset(YoloSourceDataset):
    def _load_class_names(self) -> dict[int, str]:
        if self._class_names is not None:
            return self._class_names

        if self.cfg.class_names is None:
            raise ValueError(
                f"source_format='yolo_noyml' requires class_names for dataset={self.key!r}"
            )

        self._class_names = dict(self.cfg.class_names)
        return self._class_names
