from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import shutil
import struct
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from datasets import Dataset, DatasetDict, load_from_disk

from obj_det.datasets.converters.hf import HF_FEATURES


logger = logging.getLogger(__name__)


SOURCE_DATASETS = (
    "acdc",
    "carpk",
    "dawn",
    "exdark",
    "hazydet",
    "hazydet_clear",
    "hazydet_real",
    "visdrone",
    "voc2007",
    "xwod",
)

OMITTED_DATASETS = {
    "bdd100k": "The current Detection 2020 source package is unavailable.",
    "cityscapes": "The official source requires an approved Cityscapes account.",
    "udacity": (
        "The obtainable raw mirror has no upstream train/val/test assignment and "
        "does not match the configured split-preserving COCO export."
    ),
}

FINAL_CLASSES = (
    "person",
    "bicycle",
    "motorcycle",
    "car",
    "bus",
    "truck",
)

LABEL_ALIASES = {
    "person": "person",
    "people": "person",
    "pedestrian": "person",
    "rider": "person",
    "bicycle": "bicycle",
    "bike": "bicycle",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "motor": "motorcycle",
    "tricycle": "motorcycle",
    "awning-tricycle": "motorcycle",
    "car": "car",
    "van": "car",
    "bus": "bus",
    "truck": "truck",
}

OUTPUT_SPLITS = ("train", "val", "test")
SOURCE_SPLITS = OUTPUT_SPLITS
SPLIT_PRIORITY = {"train": 0, "val": 1, "test": 2}
SOURCE_RANK = {name: index for index, name in enumerate(SOURCE_DATASETS)}
SPLIT_RANK = {name: index for index, name in enumerate(SOURCE_SPLITS)}
HAZY_PAIR_DATASETS = {"hazydet", "hazydet_clear"}
GEOMETRY_EPSILON = 1e-3


@dataclass(frozen=True)
class FilteredObjects:
    objects: list[dict[str, Any]]
    total: int
    ignored: int
    excluded: int


@dataclass(frozen=True)
class RowRef:
    ref_id: int
    dataset: str
    split: str
    index: int
    order: int
    image_id: str
    width: int
    height: int
    content_hash: str
    source_image_id: str | None
    object_count: int
    class_counts: tuple[tuple[str, int], ...]
    annotation_hash: str

    @property
    def class_count(self) -> int:
        return len(self.class_counts)


@dataclass(frozen=True)
class MergedRow:
    representative_id: int
    donor_id: int
    component_ids: tuple[int, ...]
    lineage_id: str


class UnionFind:
    def __init__(self) -> None:
        self.parent: list[int] = []
        self.rank: list[int] = []

    def add(self) -> int:
        item = len(self.parent)
        self.parent.append(item)
        self.rank.append(0)
        return item

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def canonical_label(dataset: str, native_label: Any) -> str | None:
    label = str(native_label).strip().casefold()
    if dataset == "carpk" and label == "0":
        return "car"
    return LABEL_ALIASES.get(label)


def parse_json(value: Any, *, context: str) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {context}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object in {context}")
    return parsed


def _float32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", value))[0]


def _previous_float32(value: float) -> float:
    bits = struct.unpack("!I", struct.pack("!f", value))[0]
    if bits == 0:
        return 0.0
    return struct.unpack("!f", struct.pack("!I", bits - 1))[0]


def _clip_float32_extent(
    start: float,
    extent: float,
    limit: int,
    *,
    axis: str,
    context: str,
    bbox: list[float],
) -> float:
    if start + extent <= limit:
        return extent
    if start + extent > limit + GEOMETRY_EPSILON:
        raise ValueError(f"BBox exceeds {axis} in {context}: {bbox!r} vs {limit}")
    clipped = _float32(float(limit) - start)
    while start + clipped > limit:
        clipped = _previous_float32(clipped)
    if clipped <= 0:
        raise ValueError(f"BBox has no area after clipping in {context}: {bbox!r}")
    return clipped


def validate_bbox(bbox: Any, *, width: int, height: int, context: str) -> list[float]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"Expected xywh bbox in {context}, got {bbox!r}")
    raw_values = [float(value) for value in bbox]
    if not all(math.isfinite(value) for value in raw_values):
        raise ValueError(f"Non-finite bbox in {context}: {raw_values!r}")
    try:
        values = [_float32(value) for value in raw_values]
    except OverflowError as exc:
        raise ValueError(
            f"BBox exceeds float32 range in {context}: {raw_values!r}"
        ) from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite float32 bbox in {context}: {values!r}")
    x, y, box_width, box_height = values
    if x < 0 or y < 0 or box_width <= 0 or box_height <= 0:
        raise ValueError(f"Invalid bbox in {context}: {values!r}")
    values[2] = _clip_float32_extent(
        x,
        box_width,
        width,
        axis="width",
        context=context,
        bbox=values,
    )
    values[3] = _clip_float32_extent(
        y,
        box_height,
        height,
        axis="height",
        context=context,
        bbox=values,
    )
    return values


def filter_objects(
    row: dict[str, Any],
    dataset: str,
    *,
    annotation_source: dict[str, str] | None = None,
) -> FilteredObjects:
    width = int(row["width"])
    height = int(row["height"])
    retained: list[dict[str, Any]] = []
    ignored = 0
    excluded = 0
    source_objects = row.get("objects") or []

    for object_index, obj in enumerate(source_objects):
        if bool(obj.get("ignore", False)):
            ignored += 1
            continue
        meta_label = canonical_label(dataset, obj.get("native_label", ""))
        if meta_label is None:
            excluded += 1
            continue
        context = f"{dataset}/{row['split']}/{row['image_id']} object {object_index}"
        object_meta = parse_json(obj.get("meta_json"), context=f"{context} meta_json")
        if annotation_source is not None:
            object_meta["merge_annotation_source"] = annotation_source
        retained.append(
            {
                "bbox": validate_bbox(
                    obj.get("bbox"),
                    width=width,
                    height=height,
                    context=context,
                ),
                "native_label": str(obj.get("native_label", "")),
                "native_label_id": (
                    None
                    if obj.get("native_label_id") is None
                    else str(obj["native_label_id"])
                ),
                "meta_label": meta_label,
                "ignore": False,
                "iscrowd": bool(obj.get("iscrowd", False)),
                "meta_json": json.dumps(
                    object_meta, sort_keys=True, separators=(",", ":")
                ),
            }
        )

    return FilteredObjects(
        objects=retained,
        total=len(source_objects),
        ignored=ignored,
        excluded=excluded,
    )


def image_bytes(row: dict[str, Any]) -> bytes:
    image = row.get("image")
    if not isinstance(image, dict):
        raise ValueError(f"Expected embedded image mapping for {row.get('image_id')}")
    data = image.get("bytes")
    if data is not None:
        return bytes(data)
    path_value = image.get("path") or row.get("image_path")
    if not path_value:
        raise ValueError(f"Missing image bytes and path for {row.get('image_id')}")
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"Image fallback path does not exist: {path}")
    return path.read_bytes()


def content_hash(row: dict[str, Any]) -> str:
    return hashlib.blake2b(image_bytes(row), digest_size=32).hexdigest()


def annotation_hash(objects: list[dict[str, Any]]) -> str:
    signature = [
        {
            "bbox": obj["bbox"],
            "meta_label": obj["meta_label"],
            "iscrowd": obj["iscrowd"],
        }
        for obj in objects
    ]
    encoded = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


def load_sources(input_root: Path) -> dict[str, DatasetDict]:
    missing = [name for name in SOURCE_DATASETS if not (input_root / name).is_dir()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing required converted datasets under {input_root}: {joined}"
        )

    sources: dict[str, DatasetDict] = {}
    for name in SOURCE_DATASETS:
        loaded = load_from_disk(str(input_root / name))
        if not isinstance(loaded, DatasetDict):
            raise TypeError(f"Expected DatasetDict at {input_root / name}")
        unexpected = sorted(set(loaded) - set(SOURCE_SPLITS))
        if unexpected:
            raise ValueError(f"Unsupported splits for {name}: {unexpected}")
        for split, dataset in loaded.items():
            if dataset.features != HF_FEATURES:
                raise ValueError(
                    f"Dataset {name}/{split} does not use the canonical HF schema"
                )
        sources[name] = loaded
    return sources


def _stable_ref_key(ref: RowRef) -> tuple[int, int, int, str]:
    return SOURCE_RANK[ref.dataset], SPLIT_RANK[ref.split], ref.index, ref.image_id


def scan_sources(
    sources: dict[str, DatasetDict],
) -> tuple[list[RowRef], UnionFind, dict[str, Any]]:
    refs: list[RowRef] = []
    unions = UnionFind()
    hash_first: dict[str, int] = {}
    hazy_first: dict[str, int] = {}
    totals: Counter[str] = Counter()
    input_classes: Counter[str] = Counter()
    per_dataset: dict[str, Counter[str]] = {name: Counter() for name in SOURCE_DATASETS}
    source_metadata: dict[str, dict[str, set[str]]] = {
        name: defaultdict(set) for name in SOURCE_DATASETS
    }

    for dataset_name in SOURCE_DATASETS:
        dataset_dict = sources[dataset_name]
        for split in SOURCE_SPLITS:
            if split not in dataset_dict:
                continue
            dataset = dataset_dict[split]
            logger.info("Scanning %s/%s (%d rows)", dataset_name, split, len(dataset))
            for index, row in enumerate(dataset):
                if str(row["dataset"]) != dataset_name:
                    raise ValueError(
                        f"Dataset field mismatch in {dataset_name}/{split}/{index}: "
                        f"{row['dataset']!r}"
                    )
                if str(row["split"]) != split:
                    raise ValueError(
                        f"Split field mismatch in {dataset_name}/{split}/{index}: "
                        f"{row['split']!r}"
                    )

                filtered = filter_objects(row, dataset_name)
                object_classes = Counter(obj["meta_label"] for obj in filtered.objects)
                row_hash = content_hash(row)
                row_meta = parse_json(
                    row.get("meta_json"),
                    context=f"{dataset_name}/{split}/{row['image_id']} meta_json",
                )
                source_image_id = row_meta.get("source_image_id")
                ref_id = unions.add()
                ref = RowRef(
                    ref_id=ref_id,
                    dataset=dataset_name,
                    split=split,
                    index=index,
                    order=len(refs),
                    image_id=str(row["image_id"]),
                    width=int(row["width"]),
                    height=int(row["height"]),
                    content_hash=row_hash,
                    source_image_id=(
                        None if source_image_id is None else str(source_image_id)
                    ),
                    object_count=len(filtered.objects),
                    class_counts=tuple(sorted(object_classes.items())),
                    annotation_hash=annotation_hash(filtered.objects),
                )
                refs.append(ref)

                previous = hash_first.setdefault(row_hash, ref_id)
                unions.union(previous, ref_id)
                if dataset_name in HAZY_PAIR_DATASETS:
                    if source_image_id is None:
                        raise ValueError(
                            f"Missing source_image_id for {dataset_name}/{split}/{ref.image_id}"
                        )
                    hazy_key = str(source_image_id)
                    previous_hazy = hazy_first.setdefault(hazy_key, ref_id)
                    unions.union(previous_hazy, ref_id)

                totals.update(
                    rows=1,
                    objects=filtered.total,
                    ignored_objects_dropped=filtered.ignored,
                    excluded_class_objects_dropped=filtered.excluded,
                    retained_class_objects=len(filtered.objects),
                )
                if not filtered.objects:
                    totals["rows_without_final_objects_before_enrichment"] += 1
                input_classes.update(object_classes)

                dataset_stats = per_dataset[dataset_name]
                dataset_stats.update(
                    rows=1,
                    objects=filtered.total,
                    ignored_objects_dropped=filtered.ignored,
                    excluded_class_objects_dropped=filtered.excluded,
                    retained_class_objects=len(filtered.objects),
                )
                dataset_stats[f"split:{split}"] += 1
                if not filtered.objects:
                    dataset_stats["rows_without_final_objects_before_enrichment"] += 1

                for key in ("homepage", "license", "source_annotation_format"):
                    value = row_meta.get(key)
                    if value not in (None, ""):
                        source_metadata[dataset_name][key].add(str(value))

    return (
        refs,
        unions,
        {
            "totals": totals,
            "classes": input_classes,
            "per_dataset": per_dataset,
            "source_metadata": source_metadata,
        },
    )


def _lineage_id(members: list[RowRef]) -> str:
    identities = [
        f"{ref.dataset}|{ref.split}|{ref.image_id}|{ref.content_hash}"
        for ref in sorted(members, key=_stable_ref_key)
    ]
    return hashlib.blake2b("\n".join(identities).encode(), digest_size=16).hexdigest()


def plan_merge(
    refs: list[RowRef],
    unions: UnionFind,
) -> tuple[list[MergedRow], dict[str, int]]:
    components: dict[int, list[RowRef]] = defaultdict(list)
    hash_groups: dict[str, list[RowRef]] = defaultdict(list)
    for ref in refs:
        components[unions.find(ref.ref_id)].append(ref)
        hash_groups[ref.content_hash].append(ref)

    stats: Counter[str] = Counter()
    stats["lineage_components"] = len(components)
    stats["multi_record_components"] = sum(
        len(group) > 1 for group in components.values()
    )
    duplicate_groups = [group for group in hash_groups.values() if len(group) > 1]
    stats["exact_duplicate_groups"] = len(duplicate_groups)
    stats["cross_dataset_exact_duplicate_groups"] = sum(
        len({ref.dataset for ref in group}) > 1 for group in duplicate_groups
    )
    stats["cross_split_exact_duplicate_groups"] = sum(
        len({ref.split for ref in group}) > 1 for group in duplicate_groups
    )

    output_rows: list[MergedRow] = []
    ordered_components = sorted(
        components.values(),
        key=lambda group: min(ref.order for ref in group),
    )
    for members in ordered_components:
        members.sort(key=_stable_ref_key)
        dimensions = {(ref.width, ref.height) for ref in members}
        if len(dimensions) != 1:
            identities = [
                f"{ref.dataset}/{ref.split}/{ref.image_id}:{ref.width}x{ref.height}"
                for ref in members
            ]
            raise ValueError(
                "Cannot transfer annotations across mismatched dimensions: "
                + ", ".join(identities)
            )

        donor = min(
            members,
            key=lambda ref: (
                -ref.class_count,
                -ref.object_count,
                _stable_ref_key(ref),
            ),
        )
        if donor.object_count == 0:
            stats["components_without_final_annotations"] += 1
            stats["rows_dropped_without_final_annotations"] += len(members)
            continue

        protected_split = max(
            (ref.split for ref in members),
            key=lambda split: SPLIT_PRIORITY[split],
        )
        protected = [ref for ref in members if ref.split == protected_split]
        conflict_count = len(members) - len(protected)
        if conflict_count:
            stats["split_conflict_components"] += 1
            stats["rows_dropped_for_split_conflict"] += conflict_count

        by_hash: dict[str, list[RowRef]] = defaultdict(list)
        for ref in protected:
            by_hash[ref.content_hash].append(ref)

        lineage_id = _lineage_id(members)
        member_ids = tuple(ref.ref_id for ref in members)
        for same_image in sorted(
            by_hash.values(),
            key=lambda group: min(ref.order for ref in group),
        ):
            representative = min(same_image, key=_stable_ref_key)
            stats["rows_dropped_as_same_split_exact_duplicates"] += len(same_image) - 1
            if representative.annotation_hash != donor.annotation_hash:
                stats["annotation_enriched_rows"] += 1
            output_rows.append(
                MergedRow(
                    representative_id=representative.ref_id,
                    donor_id=donor.ref_id,
                    component_ids=member_ids,
                    lineage_id=lineage_id,
                )
            )

    output_rows.sort(key=lambda item: refs[item.representative_id].order)
    stats["output_rows"] = len(output_rows)
    return output_rows, dict(stats)


def _annotation_source(ref: RowRef) -> dict[str, str]:
    return {
        "dataset": ref.dataset,
        "split": ref.split,
        "image_id": ref.image_id,
    }


def build_output_row(
    merged: MergedRow,
    *,
    refs: list[RowRef],
    sources: dict[str, DatasetDict],
) -> dict[str, Any]:
    representative = refs[merged.representative_id]
    donor = refs[merged.donor_id]
    representative_row = sources[representative.dataset][representative.split][
        representative.index
    ]
    donor_row = sources[donor.dataset][donor.split][donor.index]
    source = _annotation_source(donor)
    filtered = filter_objects(donor_row, donor.dataset, annotation_source=source)
    if len(filtered.objects) != donor.object_count:
        raise RuntimeError(f"Annotation count changed while reading {donor.image_id}")

    members = sorted(
        (refs[ref_id] for ref_id in merged.component_ids),
        key=_stable_ref_key,
    )
    row_meta = parse_json(
        representative_row.get("meta_json"),
        context=f"{representative.dataset}/{representative.split}/{representative.image_id}",
    )
    row_meta.update(
        {
            "merge_annotation_source": source,
            "merge_content_hash": representative.content_hash,
            "merge_lineage_id": merged.lineage_id,
            "merge_original_split": representative.split,
            "merge_source_datasets": sorted(
                {ref.dataset for ref in members}, key=SOURCE_RANK.__getitem__
            ),
            "merge_source_records": [
                {
                    "dataset": ref.dataset,
                    "split": ref.split,
                    "image_id": ref.image_id,
                    "content_hash": ref.content_hash,
                }
                for ref in members
            ],
        }
    )

    return {
        "image": representative_row["image"],
        "image_id": representative_row["image_id"],
        "dataset": representative_row["dataset"],
        "split": representative.split,
        "image_path": representative_row["image_path"],
        "width": representative.width,
        "height": representative.height,
        "condition": representative_row["condition"],
        "domain": representative_row["domain"],
        "is_synthetic": representative_row["is_synthetic"],
        "objects": filtered.objects,
        "meta_json": json.dumps(row_meta, sort_keys=True, separators=(",", ":")),
    }


def _iter_output_rows(
    merged_rows: tuple[MergedRow, ...],
    refs: tuple[RowRef, ...],
    sources: dict[str, DatasetDict],
    split: str,
) -> Iterator[dict[str, Any]]:
    for position, merged in enumerate(merged_rows, start=1):
        if position % 1000 == 0:
            logger.info("Writing %s row %d/%d", split, position, len(merged_rows))
        yield build_output_row(merged, refs=refs, sources=sources)


def _split_fingerprint(rows: list[MergedRow], refs: list[RowRef]) -> str:
    parts = []
    for merged in rows:
        representative = refs[merged.representative_id]
        donor = refs[merged.donor_id]
        parts.append(
            "|".join(
                (
                    representative.content_hash,
                    representative.image_id,
                    merged.lineage_id,
                    donor.annotation_hash,
                )
            )
        )
    return hashlib.blake2b("\n".join(parts).encode(), digest_size=16).hexdigest()


def build_dataset_dict(
    merged_rows: list[MergedRow],
    *,
    refs: list[RowRef],
    sources: dict[str, DatasetDict],
    cache_dir: Path,
) -> DatasetDict:
    by_split: dict[str, list[MergedRow]] = {split: [] for split in OUTPUT_SPLITS}
    for merged in merged_rows:
        split = refs[merged.representative_id].split
        by_split[split].append(merged)

    output = DatasetDict()
    for split in OUTPUT_SPLITS:
        rows = by_split[split]
        if not rows:
            output[split] = Dataset.from_dict(
                {column: [] for column in HF_FEATURES},
                features=HF_FEATURES,
            )
            continue
        output[split] = Dataset.from_generator(
            _iter_output_rows,
            features=HF_FEATURES,
            split=split,
            cache_dir=str(cache_dir),
            fingerprint=_split_fingerprint(rows, refs),
            gen_kwargs={
                "merged_rows": tuple(rows),
                "refs": tuple(refs),
                "sources": sources,
                "split": split,
            },
        )
    return output


def build_manifest(
    refs: list[RowRef],
    merged_rows: list[MergedRow],
    scan_stats: dict[str, Any],
    merge_stats: dict[str, int],
) -> dict[str, Any]:
    output_splits: Counter[str] = Counter()
    output_datasets: Counter[str] = Counter()
    output_classes: Counter[str] = Counter()
    order_hashes: dict[str, Any] = {}
    split_rows: dict[str, list[MergedRow]] = {split: [] for split in OUTPUT_SPLITS}

    for merged in merged_rows:
        representative = refs[merged.representative_id]
        donor = refs[merged.donor_id]
        output_splits[representative.split] += 1
        output_datasets[representative.dataset] += 1
        output_classes.update(dict(donor.class_counts))
        split_rows[representative.split].append(merged)

    for split, rows in split_rows.items():
        order_hashes[split] = _split_fingerprint(rows, refs)

    source_stats: dict[str, Any] = {}
    for name in SOURCE_DATASETS:
        counters: Counter[str] = scan_stats["per_dataset"][name]
        source_stats[name] = {
            "rows": counters["rows"],
            "splits": {
                split: counters[f"split:{split}"]
                for split in SOURCE_SPLITS
                if counters[f"split:{split}"]
            },
            "objects": counters["objects"],
            "retained_class_objects": counters["retained_class_objects"],
            "ignored_objects_dropped": counters["ignored_objects_dropped"],
            "excluded_class_objects_dropped": counters[
                "excluded_class_objects_dropped"
            ],
            "rows_without_final_objects_before_enrichment": counters[
                "rows_without_final_objects_before_enrichment"
            ],
            "metadata": {
                key: sorted(values)
                for key, values in scan_stats["source_metadata"][name].items()
            },
        }

    totals: Counter[str] = scan_stats["totals"]
    return {
        "schema_version": 1,
        "included_datasets": list(SOURCE_DATASETS),
        "omitted_datasets": OMITTED_DATASETS,
        "class_selection": {
            "minimum_independent_source_families": 5,
            "hazy_variants_count_as_one_family": True,
            "final_classes": list(FINAL_CLASSES),
            "aliases": LABEL_ALIASES,
            "dataset_specific_aliases": {"carpk": {"0": "car"}},
        },
        "split_policy": {
            "preserved_splits": list(OUTPUT_SPLITS),
            "conflict_priority": ["test", "val", "train"],
            "rows_are_never_reassigned": True,
        },
        "input": {
            "rows": totals["rows"],
            "objects": totals["objects"],
            "retained_class_objects": totals["retained_class_objects"],
            "retained_class_counts": dict(
                sorted(scan_stats["classes"].items(), key=lambda item: item[0])
            ),
            "ignored_objects_dropped": totals["ignored_objects_dropped"],
            "excluded_class_objects_dropped": totals["excluded_class_objects_dropped"],
            "rows_without_final_objects_before_enrichment": totals[
                "rows_without_final_objects_before_enrichment"
            ],
            "datasets": source_stats,
        },
        "deduplication": merge_stats,
        "output": {
            "rows": len(merged_rows),
            "objects": sum(output_classes.values()),
            "splits": {split: output_splits[split] for split in OUTPUT_SPLITS},
            "representative_datasets": {
                name: output_datasets[name] for name in SOURCE_DATASETS
            },
            "class_counts": {name: output_classes[name] for name in FINAL_CLASSES},
            "record_order_hashes": order_hashes,
        },
    }


def validate_saved_dataset(path: Path, manifest: dict[str, Any]) -> None:
    loaded = load_from_disk(str(path))
    if not isinstance(loaded, DatasetDict):
        raise TypeError(f"Merged output is not a DatasetDict: {path}")
    if tuple(loaded.keys()) != OUTPUT_SPLITS:
        raise ValueError(f"Unexpected output splits: {list(loaded.keys())}")

    content_splits: dict[str, str] = {}
    lineage_splits: dict[str, str] = {}
    class_counts: Counter[str] = Counter()
    seen_image_ids: set[str] = set()
    split_counts: Counter[str] = Counter()

    for split in OUTPUT_SPLITS:
        dataset = loaded[split]
        if dataset.features != HF_FEATURES:
            raise ValueError(
                f"Merged split {split} does not use the canonical HF schema"
            )
        columns = dataset.select_columns(
            ["image_id", "split", "width", "height", "objects", "meta_json"]
        )
        for row in columns:
            image_id = str(row["image_id"])
            if image_id in seen_image_ids:
                raise ValueError(f"Duplicate merged image_id: {image_id}")
            seen_image_ids.add(image_id)
            if row["split"] != split:
                raise ValueError(f"Merged row moved splits: {image_id}")
            objects = row.get("objects") or []
            if not objects:
                raise ValueError(f"Merged row has no final objects: {image_id}")
            for object_index, obj in enumerate(objects):
                if obj["meta_label"] not in FINAL_CLASSES or obj["ignore"]:
                    raise ValueError(f"Invalid merged object in {image_id}: {obj}")
                validate_bbox(
                    obj["bbox"],
                    width=int(row["width"]),
                    height=int(row["height"]),
                    context=f"merged/{split}/{image_id} object {object_index}",
                )
                class_counts[obj["meta_label"]] += 1

            meta = parse_json(row["meta_json"], context=f"merged/{split}/{image_id}")
            image_hash = str(meta["merge_content_hash"])
            lineage_id = str(meta["merge_lineage_id"])
            previous_split = content_splits.setdefault(image_hash, split)
            if previous_split != split:
                raise ValueError(f"Image content spans {previous_split} and {split}")
            previous_split = lineage_splits.setdefault(lineage_id, split)
            if previous_split != split:
                raise ValueError(f"Lineage spans {previous_split} and {split}")
            split_counts[split] += 1

    expected = manifest["output"]
    if dict(split_counts) != {
        split: count for split, count in expected["splits"].items() if count
    }:
        raise ValueError("Saved split counts do not match the manifest")
    if {name: class_counts[name] for name in FINAL_CLASSES} != expected["class_counts"]:
        raise ValueError("Saved class counts do not match the manifest")


def publish_generated_dataset_dict(dataset_dict: DatasetDict, output: Path) -> None:
    """Publish generated Arrow shards without rewriting embedded image bytes."""
    output.mkdir(parents=True)
    (output / "dataset_dict.json").write_text(
        json.dumps({"splits": list(dataset_dict)}, sort_keys=True),
        encoding="utf-8",
    )

    for split, dataset in dataset_dict.items():
        split_path = output / split
        cache_files = [Path(item["filename"]) for item in dataset.cache_files]
        if not cache_files:
            dataset.save_to_disk(split_path, num_shards=1)
            continue

        split_path.mkdir()
        data_files = []
        for index, source in enumerate(cache_files):
            filename = f"data-{index:05d}-of-{len(cache_files):05d}.arrow"
            destination = split_path / filename
            try:
                destination.hardlink_to(source)
            except OSError:
                shutil.copy2(source, destination)
            data_files.append({"filename": filename})

        state = {
            "_data_files": data_files,
            "_fingerprint": dataset._fingerprint,
            "_format_columns": None,
            "_format_kwargs": {},
            "_format_type": None,
            "_output_all_columns": False,
            "_split": split,
        }
        (split_path / "state.json").write_text(
            json.dumps(state, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        dataset.info.write_to_directory(split_path, pretty_print=True)


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def merge_datasets(
    input_root: Path,
    output: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    input_root = input_root.resolve()
    output = output.resolve()
    source_paths = {(input_root / name).resolve() for name in SOURCE_DATASETS}
    if any(
        output == source
        or output.is_relative_to(source)
        or source.is_relative_to(output)
        for source in source_paths
    ):
        raise ValueError(
            "Output cannot contain, replace, or be inside a source dataset"
        )
    if output.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {output}; pass --force to replace it"
        )

    sources = load_sources(input_root)
    refs, unions, scan_stats = scan_sources(sources)
    merged_rows, merge_stats = plan_merge(refs, unions)
    manifest = build_manifest(refs, merged_rows, scan_stats, merge_stats)

    output.parent.mkdir(parents=True, exist_ok=True)
    build_path = output.with_name(f".{output.name}.build")
    _remove_path(build_path)
    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{output.name}.cache-", dir=output.parent
        ) as cache_dir:
            merged = build_dataset_dict(
                merged_rows,
                refs=refs,
                sources=sources,
                cache_dir=Path(cache_dir),
            )
            publish_generated_dataset_dict(merged, build_path)
        validate_saved_dataset(build_path, manifest)
        (build_path / "merge_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output.exists():
            _remove_path(output)
        build_path.replace(output)
    except Exception:
        _remove_path(build_path)
        raise

    logger.info(
        "Merged %d rows into %s with splits=%s",
        manifest["output"]["rows"],
        output,
        manifest["output"]["splits"],
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge the obtainable converted detection datasets into traffic6."
    )
    parser.add_argument("--input-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output", type=Path, default=Path("datasets/merged_traffic6"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    manifest = merge_datasets(args.input_root, args.output, force=args.force)
    print(json.dumps(manifest["output"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
