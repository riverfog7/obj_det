#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any


CAMERA_CHANNELS = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
)


def infer_condition(description: str) -> str:
    description = description.lower()
    if "rain" in description:
        return "rain"
    if "night" in description:
        return "night"
    return "clear"


@lru_cache(maxsize=1)
def _projection_dependencies():
    import numpy as np
    from pyquaternion import Quaternion
    from shapely.geometry import MultiPoint, box as image_box
    from nuscenes.utils.geometry_utils import view_points

    return np, Quaternion, MultiPoint, image_box, view_points


def project_box(
    nusc,
    annotation_token: str,
    sample_data: dict[str, Any],
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    np, Quaternion, MultiPoint, image_box, view_points = _projection_dependencies()
    calibrated_sensor = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])

    box = nusc.get_box(annotation_token)
    box.translate(-np.asarray(ego_pose["translation"]))
    box.rotate(Quaternion(ego_pose["rotation"]).inverse)
    box.translate(-np.asarray(calibrated_sensor["translation"]))
    box.rotate(Quaternion(calibrated_sensor["rotation"]).inverse)

    corners = box.corners()
    corners = corners[:, corners[2, :] > 0]
    if corners.shape[1] == 0:
        return None

    projected = view_points(
        corners,
        np.asarray(calibrated_sensor["camera_intrinsic"]),
        normalize=True,
    ).T[:, :2]
    polygon = MultiPoint(projected.tolist()).convex_hull
    intersection = polygon.intersection(image_box(0, 0, width, height))
    if intersection.is_empty:
        return None

    x1, y1, x2, y2 = map(float, intersection.bounds)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def export_split(
    nusc,
    *,
    root: Path,
    split: str,
    scene_names: set[str],
) -> dict[str, Any]:
    categories = [
        {
            "id": index,
            "name": category["name"],
            "supercategory": category["name"].split(".", 1)[0],
            "meta": {
                "token": category["token"],
                "description": category.get("description"),
            },
        }
        for index, category in enumerate(nusc.category, start=1)
    ]
    category_ids = {category["name"]: category["id"] for category in categories}
    scenes = {scene["token"]: scene for scene in nusc.scene}
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []

    for sample_index, sample in enumerate(nusc.sample, start=1):
        scene = scenes[sample["scene_token"]]
        if scene["name"] not in scene_names:
            continue
        log = nusc.get("log", scene["log_token"])
        condition = infer_condition(scene.get("description", ""))

        for channel in CAMERA_CHANNELS:
            sample_data_token = sample["data"][channel]
            sample_data = nusc.get("sample_data", sample_data_token)
            if not sample_data["is_key_frame"]:
                continue

            filename = sample_data["filename"]
            image_path = root / filename
            if not image_path.exists():
                raise FileNotFoundError(f"Missing nuScenes camera image: {image_path}")

            image_id = len(images) + 1
            width = int(sample_data["width"])
            height = int(sample_data["height"])
            images.append(
                {
                    "id": image_id,
                    "file_name": filename,
                    "width": width,
                    "height": height,
                    "condition": condition,
                    "domain": "road",
                    "is_synthetic": False,
                    "meta": {
                        "sample_token": sample["token"],
                        "sample_data_token": sample_data_token,
                        "scene_token": scene["token"],
                        "scene_name": scene["name"],
                        "scene_description": scene.get("description"),
                        "channel": channel,
                        "location": log.get("location"),
                        "timestamp": sample_data.get("timestamp"),
                    },
                }
            )

            for annotation_token in sample["anns"]:
                coords = project_box(
                    nusc,
                    annotation_token,
                    sample_data,
                    width=width,
                    height=height,
                )
                if coords is None:
                    continue

                source = nusc.get("sample_annotation", annotation_token)
                x1, y1, x2, y2 = coords
                box_width = x2 - x1
                box_height = y2 - y1
                attribute_names = [
                    nusc.get("attribute", token)["name"]
                    for token in source.get("attribute_tokens", [])
                ]
                annotations.append(
                    {
                        "id": len(annotations) + 1,
                        "image_id": image_id,
                        "category_id": category_ids[source["category_name"]],
                        "bbox": [x1, y1, box_width, box_height],
                        "area": box_width * box_height,
                        "iscrowd": 0,
                        "meta": {
                            "sample_annotation_token": annotation_token,
                            "instance_token": source["instance_token"],
                            "visibility_token": source.get("visibility_token"),
                            "attribute_names": attribute_names,
                            "num_lidar_pts": source.get("num_lidar_pts"),
                            "num_radar_pts": source.get("num_radar_pts"),
                            "channel": channel,
                        },
                    }
                )

        if sample_index % 500 == 0:
            print(
                f"{split}: scanned {sample_index} samples, "
                f"kept {len(images)} images and {len(annotations)} boxes",
                flush=True,
            )

    return {
        "info": {
            "description": "nuScenes 3D boxes projected onto keyframe camera images",
            "version": nusc.version,
            "split": split,
            "camera_channels": list(CAMERA_CHANNELS),
        },
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project nuScenes train/val 3D annotations into COCO camera boxes."
    )
    parser.add_argument("root", type=Path, help="Extracted nuScenes dataset root.")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes

    root = args.root.resolve()
    output_dir = args.output_dir or root / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version=args.version, dataroot=str(root), verbose=True)
    split_scenes = create_splits_scenes()
    split_keys = (
        {"train": "mini_train", "val": "mini_val"}
        if args.version == "v1.0-mini"
        else {"train": "train", "val": "val"}
    )

    for output_split, scene_key in split_keys.items():
        payload = export_split(
            nusc,
            root=root,
            split=output_split,
            scene_names=set(split_scenes[scene_key]),
        )
        output_path = output_dir / f"{output_split}.json"
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, separators=(",", ":"))
        print(
            f"Wrote {len(payload['images'])} images and "
            f"{len(payload['annotations'])} boxes to {output_path}"
        )


if __name__ == "__main__":
    main()
