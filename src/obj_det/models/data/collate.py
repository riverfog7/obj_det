from __future__ import annotations

from obj_det.models.data.sample import DetectionBatch, DetectionSample


def detection_collate(samples: list[DetectionSample]) -> DetectionBatch:
    return DetectionBatch(
        images=[sample.image for sample in samples],
        targets=[sample.targets for sample in samples],
        samples=samples,
    )
