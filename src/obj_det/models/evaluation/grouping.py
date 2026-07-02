from __future__ import annotations

from obj_det.models.data.sample import DetectionSample


def image_ids_by_field(samples: list[DetectionSample], field: str) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = {}
    for sample in samples:
        value = str(getattr(sample, field) or "unknown")
        groups.setdefault(value, set()).add(sample.image_id)
    return groups
