from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset, DatasetDict, load_from_disk
from PIL import Image

from obj_det.datasets.converters.hf import HF_FEATURES
from obj_det.models.data.row_parser import HFDetectionRowParser
from scripts.merge_datasets import (
    FINAL_CLASSES,
    MergedRow,
    OMITTED_DATASETS,
    RowRef,
    SOURCE_DATASETS,
    UnionFind,
    build_output_row,
    filter_objects,
    load_sources,
    merge_datasets,
    plan_merge,
    validate_bbox,
)


_DEFAULT_META_LABEL = object()


def _image_bytes(color: tuple[int, int, int], size: tuple[int, int] = (8, 8)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _object(
    label: str,
    *,
    bbox: list[float] | None = None,
    ignore: bool = False,
    meta_label: str | None | object = _DEFAULT_META_LABEL,
) -> dict[str, object]:
    if meta_label is _DEFAULT_META_LABEL:
        meta_label = label.lower()
    return {
        "bbox": bbox or [1.0, 1.0, 3.0, 3.0],
        "native_label": label,
        "native_label_id": label,
        "meta_label": meta_label,
        "ignore": ignore,
        "iscrowd": False,
        "meta_json": json.dumps({"source": "test"}),
    }


def _row(
    dataset: str,
    split: str,
    image_id: str,
    labels: list[str],
    *,
    color: tuple[int, int, int] = (1, 2, 3),
    data: bytes | None = None,
    source_image_id: int | None = None,
    size: tuple[int, int] = (8, 8),
    meta_labels: list[str | None] | None = None,
) -> dict[str, object]:
    width, height = size
    meta = {
        "homepage": f"https://example.test/{dataset}",
        "license": "test-only",
        "source_annotation_format": "fixture",
    }
    if source_image_id is not None:
        meta["source_image_id"] = source_image_id
    return {
        "image": {
            "bytes": data if data is not None else _image_bytes(color, size),
            "path": f"{image_id}.png",
        },
        "image_id": image_id,
        "dataset": dataset,
        "split": split,
        "image_path": f"{image_id}.png",
        "width": width,
        "height": height,
        "condition": "clear",
        "domain": "road",
        "is_synthetic": False,
        "objects": [
            _object(
                label,
                meta_label=(
                    _DEFAULT_META_LABEL if meta_labels is None else meta_labels[index]
                ),
            )
            for index, label in enumerate(labels)
        ],
        "meta_json": json.dumps(meta),
    }


def _dataset_dict(rows: list[dict[str, object]]) -> DatasetDict:
    by_split: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_split.setdefault(str(row["split"]), []).append(row)
    return DatasetDict(
        {
            split: Dataset.from_list(split_rows, features=HF_FEATURES)
            for split, split_rows in by_split.items()
        }
    )


def _ref(
    ref_id: int,
    dataset: str,
    split: str,
    *,
    image_hash: str = "same",
    width: int = 8,
    height: int = 8,
    object_count: int = 1,
    class_counts: tuple[tuple[str, int], ...] = (("car", 1),),
    annotation_hash: str = "annotations",
    image_id: str | None = None,
) -> RowRef:
    return RowRef(
        ref_id=ref_id,
        dataset=dataset,
        split=split,
        index=0,
        order=ref_id,
        image_id=image_id or f"{dataset}:{split}:{ref_id}",
        width=width,
        height=height,
        content_hash=image_hash,
        source_image_id=None,
        object_count=object_count,
        class_counts=class_counts,
        annotation_hash=annotation_hash,
    )


class MergeDatasetsTest(unittest.TestCase):
    def test_merge_catalog_and_final_classes_match_repo_configs(self):
        dataset_keys = {
            path.stem for path in Path("configs/datasets").glob("*.yaml")
        }
        self.assertEqual(set(SOURCE_DATASETS) | set(OMITTED_DATASETS), dataset_keys)
        self.assertFalse(set(SOURCE_DATASETS) & set(OMITTED_DATASETS))
        self.assertEqual(OMITTED_DATASETS, {})

        class_space = yaml.safe_load(
            Path("configs/class_spaces/traffic6.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(list(FINAL_CLASSES), class_space["classes"])

    def test_filter_objects_uses_converted_meta_labels_without_remapping(self):
        row = _row("udacity", "train", "sample", [])
        row["objects"] = [
            _object("biker", meta_label=None),
            _object("not-a-car", meta_label="car"),
            _object("car", meta_label="boat"),
            _object("bus", meta_label="bus", ignore=True),
        ]

        filtered = filter_objects(row, "udacity")

        self.assertEqual(filtered.total, 4)
        self.assertEqual(filtered.ignored, 1)
        self.assertEqual(filtered.excluded, 2)
        self.assertEqual(len(filtered.objects), 1)
        self.assertEqual(filtered.objects[0]["native_label"], "not-a-car")
        self.assertEqual(filtered.objects[0]["meta_label"], "car")

    def test_filter_objects_removes_ignored_and_excluded_classes(self):
        row = _row("exdark", "train", "sample", [])
        row["objects"] = [
            _object("Car"),
            _object("dog"),
            _object("bus", ignore=True),
        ]

        filtered = filter_objects(row, "exdark")

        self.assertEqual(filtered.total, 3)
        self.assertEqual(filtered.ignored, 1)
        self.assertEqual(filtered.excluded, 1)
        self.assertEqual([obj["meta_label"] for obj in filtered.objects], ["car"])
        self.assertFalse(filtered.objects[0]["ignore"])

    def test_bbox_clips_only_float32_boundary_rounding(self):
        clipped = validate_bbox(
            [3109.10009765625, 1101.824951171875, 730.9000244140625, 485.0],
            width=3840,
            height=2160,
            context="fixture",
        )
        self.assertEqual(clipped[0] + clipped[2], 3840.0)

        clipped = validate_bbox(
            [0.005, 38.86000061035156, 1268.57, 681.1400146484375],
            width=1280,
            height=720,
            context="fixture",
        )
        self.assertLessEqual(clipped[1] + clipped[3], 720.0)

        with self.assertRaisesRegex(ValueError, "exceeds width"):
            validate_bbox(
                [3109.1, 1101.8, 731.0, 485.0],
                width=3840,
                height=2160,
                context="fixture",
            )

    def test_split_conflicts_keep_only_the_most_protected_original_split(self):
        refs = [
            _ref(0, "acdc", "train"),
            _ref(1, "carpk", "val"),
            _ref(2, "exdark", "test"),
        ]
        unions = UnionFind()
        for _ in refs:
            unions.add()
        for ref_id in range(1, len(refs)):
            unions.union(0, ref_id)

        rows, stats = plan_merge(refs, unions)

        self.assertEqual(len(rows), 1)
        self.assertEqual(refs[rows[0].representative_id].split, "test")
        self.assertEqual(stats["rows_dropped_for_split_conflict"], 2)
        self.assertEqual(stats["split_conflict_components"], 1)

    def test_same_split_duplicates_use_the_richer_annotation_donor(self):
        refs = [
            _ref(0, "acdc", "train", annotation_hash="car-only"),
            _ref(
                1,
                "carpk",
                "train",
                object_count=2,
                class_counts=(("car", 1), ("person", 1)),
                annotation_hash="richer",
            ),
        ]
        unions = UnionFind()
        unions.add()
        unions.add()
        unions.union(0, 1)

        rows, stats = plan_merge(refs, unions)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].representative_id, 0)
        self.assertEqual(rows[0].donor_id, 1)
        self.assertEqual(stats["rows_dropped_as_same_split_exact_duplicates"], 1)
        self.assertEqual(stats["annotation_enriched_rows"], 1)

    def test_mismatched_lineage_dimensions_fail_closed(self):
        refs = [
            _ref(0, "hazydet", "train", image_hash="hazy", width=8),
            _ref(1, "hazydet_clear", "train", image_hash="clear", width=9),
        ]
        unions = UnionFind()
        unions.add()
        unions.add()
        unions.union(0, 1)

        with self.assertRaisesRegex(ValueError, "mismatched dimensions"):
            plan_merge(refs, unions)

    def test_build_output_row_transfers_richer_annotations_and_provenance(self):
        representative_row = _row(
            "hazydet", "train", "hazydet:train:7", ["car"], source_image_id=7
        )
        donor_row = _row("visdrone", "train", "visdrone:train:7", ["person", "car"])
        sources = {
            "hazydet": _dataset_dict([representative_row]),
            "visdrone": _dataset_dict([donor_row]),
        }
        refs = [
            _ref(0, "hazydet", "train", image_hash="hazy", annotation_hash="car"),
            _ref(
                1,
                "visdrone",
                "train",
                image_hash="clear",
                object_count=2,
                class_counts=(("car", 1), ("person", 1)),
                annotation_hash="richer",
                image_id="visdrone:train:7",
            ),
        ]
        merged = MergedRow(
            representative_id=0,
            donor_id=1,
            component_ids=(0, 1),
            lineage_id="lineage",
        )

        output = build_output_row(merged, refs=refs, sources=sources)

        self.assertEqual(
            [obj["meta_label"] for obj in output["objects"]], ["person", "car"]
        )
        row_meta = json.loads(output["meta_json"])
        self.assertEqual(row_meta["merge_original_split"], "train")
        self.assertEqual(row_meta["merge_annotation_source"]["dataset"], "visdrone")
        object_meta = json.loads(output["objects"][0]["meta_json"])
        self.assertEqual(
            object_meta["merge_annotation_source"]["image_id"],
            "visdrone:train:7",
        )

    def test_missing_required_source_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "Missing required"):
                load_sources(Path(tmp))

    def test_output_path_cannot_overlap_source_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_root = Path(tmp) / "datasets"
            unsafe_outputs = (
                Path(tmp),
                input_root,
                input_root / "acdc",
                input_root / "acdc" / "nested",
            )
            for output in unsafe_outputs:
                with self.subTest(output=output):
                    with self.assertRaisesRegex(ValueError, "source dataset"):
                        merge_datasets(input_root, output, force=True)

    def test_real_interface_merges_all_sources_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "datasets"
            output = root / "merged"
            input_root.mkdir()
            shared = _image_bytes((255, 0, 0))
            rows: dict[str, list[dict[str, Any]]] = {
                "acdc": [_row("acdc", "train", "acdc:train:1", ["car"], data=shared)],
                "bdd100k": [
                    _row(
                        "bdd100k",
                        "train",
                        "bdd100k:train:1",
                        ["person"],
                        color=(11, 0, 0),
                    )
                ],
                "carpk": [
                    _row(
                        "carpk",
                        "test",
                        "carpk:test:1",
                        ["0"],
                        data=shared,
                        meta_labels=["car"],
                    )
                ],
                "dawn": [
                    _row("dawn", "val", "dawn:val:1", ["bus"], color=(2, 0, 0))
                ],
                "exdark": [
                    _row("exdark", "train", "exdark:train:1", [], color=(3, 0, 0))
                ],
                "hazydet": [
                    _row(
                        "hazydet",
                        "train",
                        "hazydet:train:9",
                        ["car"],
                        color=(4, 0, 0),
                        source_image_id=9,
                    )
                ],
                "hazydet_clear": [
                    _row(
                        "hazydet_clear",
                        "train",
                        "hazydet_clear:train:9",
                        ["car"],
                        color=(5, 0, 0),
                        source_image_id=9,
                    )
                ],
                "hazydet_real": [
                    _row(
                        "hazydet_real",
                        "test",
                        "hazydet_real:test:1",
                        ["truck"],
                        color=(6, 0, 0),
                    )
                ],
                "udacity": [
                    _row(
                        "udacity",
                        "val",
                        "udacity:val:1",
                        ["car"],
                        color=(10, 0, 0),
                    )
                ],
                "visdrone": [
                    _row(
                        "visdrone",
                        "train",
                        "visdrone:train:1",
                        ["person"],
                        color=(7, 0, 0),
                    )
                ],
                "voc2007": [
                    _row(
                        "voc2007",
                        "val",
                        "voc2007:val:1",
                        ["bicycle"],
                        color=(8, 0, 0),
                    )
                ],
                "xwod": [
                    _row(
                        "xwod",
                        "train",
                        "xwod:train:1",
                        ["motorcycle"],
                        color=(9, 0, 0),
                    )
                ],
            }
            rows["exdark"][0]["objects"] = [
                _object("Dog", meta_label=None),
                _object("Bus", meta_label="bus", ignore=True),
            ]
            for name in SOURCE_DATASETS:
                _dataset_dict(rows[name]).save_to_disk(input_root / name)

            first_manifest = merge_datasets(input_root, output)
            merged = load_from_disk(str(output))

            self.assertEqual(list(merged), ["train", "val", "test"])
            self.assertEqual(first_manifest["schema_version"], 2)
            self.assertEqual(first_manifest["included_datasets"], list(SOURCE_DATASETS))
            self.assertEqual(first_manifest["omitted_datasets"], {})
            self.assertEqual(
                first_manifest["class_selection"]["mapping_source"],
                "converted_meta_label",
            )
            self.assertEqual(first_manifest["output"]["rows"], 10)
            self.assertEqual(
                first_manifest["output"]["splits"],
                {
                    "train": 5,
                    "val": 3,
                    "test": 2,
                },
            )
            exdark_stats = first_manifest["input"]["datasets"]["exdark"]
            self.assertEqual(
                exdark_stats["observed_mappings"],
                [
                    {"native_label": "Bus", "meta_label": "bus", "objects": 1},
                    {"native_label": "Dog", "meta_label": None, "objects": 1},
                ],
            )
            self.assertEqual(exdark_stats["excluded_native_label_counts"], {"Dog": 1})
            self.assertEqual(exdark_stats["ignored_native_label_counts"], {"Bus": 1})
            self.assertEqual(
                exdark_stats["retained_class_counts"],
                {name: 0 for name in FINAL_CLASSES},
            )
            all_rows = [row for split in merged.values() for row in split]
            self.assertNotIn("acdc:train:1", {row["image_id"] for row in all_rows})
            self.assertTrue(all(row["objects"] for row in all_rows))
            self.assertTrue(
                all(
                    obj["meta_label"] in FINAL_CLASSES
                    for row in all_rows
                    for obj in row["objects"]
                )
            )
            parser = HFDetectionRowParser(
                classes=list(FINAL_CLASSES), label_mode="meta"
            )
            for row in all_rows:
                parsed = parser.parse_targets_only(row)
                self.assertEqual(
                    [target.label_id for target in parsed.targets],
                    [list(FINAL_CLASSES).index(target.label) for target in parsed.targets],
                )
            hazy_rows = [
                row
                for row in all_rows
                if row["dataset"] in {"hazydet", "hazydet_clear"}
            ]
            self.assertEqual(len(hazy_rows), 2)
            self.assertEqual(
                len(
                    {
                        json.loads(row["meta_json"])["merge_lineage_id"]
                        for row in hazy_rows
                    }
                ),
                1,
            )
            self.assertTrue((output / "merge_manifest.json").is_file())

            with self.assertRaises(FileExistsError):
                merge_datasets(input_root, output)
            second_manifest = merge_datasets(input_root, output, force=True)
            self.assertEqual(first_manifest, second_manifest)


if __name__ == "__main__":
    unittest.main()
