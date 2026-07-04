# obj-det

The current workflow is:

```text
raw dataset files in source_datasets/
    -> source adapter
    -> canonical ImageRecord objects
    -> Hugging Face DatasetDict saved locally or pushed to Hub
```

Supported source formats:

- `coco`
- `exdark`
- `visdrone_det`
- `yolo`
- `yolo_noyml`

## Setup

```bash
uv sync
```

Run the CLI with:

```bash
uv run obj-det --help
```

## Download raw datasets

There is one helper script:

```bash
bash scripts/download.sh
```

It downloads and extracts HazyDet, VisDrone, XWOD, DAWN, and ExDark into:

```text
source_datasets/
```

Warning: the script starts by deleting `source_datasets/`, so do not put anything there that you want to keep.

The script uses Hugging Face, Google Drive, and Kaggle downloads, so your local auth/setup for those tools may be required.

If Google Drive downloads do not work, please comment out the script and download manually.

## Dataset YAML configs

Dataset configs live in:

```text
configs/datasets/
```

Existing configs:

```text
configs/datasets/hazydet.yaml
configs/datasets/hazydet_clear.yaml
configs/datasets/visdrone.yaml
configs/datasets/xwod.yaml
configs/datasets/dawn.yaml
configs/datasets/exdark.yaml
```

A config describes where the raw dataset is and how to interpret it.

Important fields:

```yaml
key: xwod                     # dataset key used in output records
root: source_datasets/xwod    # raw dataset root
source_format: yolo           # coco | exdark | visdrone_det | yolo | yolo_noyml

bbox_policy: clip             # strict | clip | drop

default_condition: mixed      # default image condition
default_domain: road          # default image domain
default_is_synthetic: false

splits:
  train:
    paths:
      images: train/images
      labels: train/labels
    condition: mixed
    domain: road
    is_synthetic: false

class_map:
  bike: bicycle               # native label -> harmonized meta label

ignore_labels: []             # labels to remove during import
```

Split path keys depend on the source format:

- COCO configs use `images` and `annotations`.
- VisDrone configs use `images` and `annotations`.
- YOLO configs use `images` and `labels`; class names come from `root/data.yaml`.
- `yolo_noyml` configs use `images` and `labels`; class names come from `class_names`.
- ExDark configs use `images`, `annotations`, and `imageclasslist`.

DAWN currently has one source split:

```text
full
```

ExDark uses the official `train`, `val`, and `test` split ids from
`imageclasslist.txt`. Upstream marks ExDark as non-commercial research use.

Optional split field:

```yaml
output_split: real_tr
```

This renames the split in the saved `DatasetDict`. It is useful when a source split name would otherwise be merged or inferred strangely by Hugging Face tooling.

## Convert a dataset

Convert all splits and save locally:

```bash
uv run obj-det datasets convert configs/datasets/xwod.yaml datasets/xwod
```

Convert selected splits only:

```bash
uv run obj-det datasets convert configs/datasets/xwod.yaml datasets/xwod \
  --split train \
  --split val
```

Limit shard size:

```bash
uv run obj-det datasets convert configs/datasets/xwod.yaml datasets/xwod \
  --max-shard-size 500MB
```

Push to the Hugging Face Hub after saving locally:

```bash
uv run obj-det datasets convert configs/datasets/xwod.yaml datasets/xwod \
  --hub-id USER/DATASET
```

Useful Hub options:

```bash
--private
--token YOUR_TOKEN        # or use HF_TOKEN env var
--config-name default
--max-shard-size 500MB
```

## Load converted datasets

Local conversion uses `DatasetDict.save_to_disk`, so load it with `load_from_disk`:

```python
from datasets import load_from_disk

ds = load_from_disk("datasets/xwod")
print(ds)
```

Do not use `load_dataset("datasets/xwod")` for local saved outputs. That is for dataset scripts/files and can infer/merge splits differently.

## Output columns

Converted datasets currently include:

```text
image
image_id
dataset
split
image_path
width
height
condition
domain
is_synthetic
objects
meta_json
```

Each object contains:

```text
bbox                 # xywh absolute pixels
native_label
native_label_id
meta_label
ignore
iscrowd
meta_json
```

## Model training API

The model layer consumes only converted Hugging Face datasets:

```text
HF DatasetDict
    -> row parser
    -> shared preprocessing / augmentation
    -> model adapter
    -> canonical predictions
    -> shared evaluator
    -> optional tuning runner
```

It does not parse raw dataset folders, COCO JSON, YOLO label files, VisDrone TXT files, or other source formats.
Raw dataset parsing belongs only in `obj_det.datasets`.

Minimal Python usage:

```python
from datasets import load_from_disk

from obj_det.models.adapters.factory import model_adapter_from_config
from obj_det.models.schemas import DataLoaderConfig, EvalConfig, ModelConfig, TrainConfig, TransformConfig

hf_ds = load_from_disk("datasets/hazydet")

model_cfg = ModelConfig(
    key="fasterrcnn_r50",
    backend="torchvision",
    model_name_or_path="fasterrcnn_resnet50_fpn",
)
adapter = model_adapter_from_config(model_cfg)
transform = TransformConfig(
    image_size=640,
    horizontal_flip_p=0.5,
    color_jitter_strength=0.1,
)

train_cfg = TrainConfig(
    run_key="fasterrcnn_hazydet_seed0",
    classes=["person", "bicycle", "motorcycle", "car", "bus", "truck"],
    label_mode="meta",
    output_dir="runs/fasterrcnn/hazydet/seed0",
    transform=transform,
    loader=DataLoaderConfig(num_workers=4, persistent_workers=True, prefetch_factor=2),
    max_epochs=50,
    batch_size=16,
)

eval_cfg = EvalConfig(
    classes=train_cfg.classes,
    label_mode=train_cfg.label_mode,
    transform=transform,
)

artifact = adapter.train(hf_ds["train"], hf_ds["validation"], train_cfg)
result = adapter.evaluate(hf_ds["test"], artifact, eval_cfg)
print(result.primary_metric, result.primary_metric_value)
```

Current backend status:

- `torchvision`: Faster R-CNN adapter trained through Hugging Face `Trainer`; no custom PyTorch loop.
- `hf_trainer`: Transformers `Trainer` adapter with COCO-style image-processor targets and canonical predictions.
- `ultralytics`: HF-backed Ultralytics `DetectionTrainer` adapter; no YOLO folder export is used.

Optional backend smoke tests are available with:

```bash
OBJ_DET_RUN_BACKEND_SMOKE=1 uv run python -m unittest tests.models.test_backend_smoke
```

## Model CLI

Model runs can also be driven by YAML configs under `configs/experiments/`.
The model CLI only loads converted Hugging Face datasets from disk.

```bash
uv run obj-det models train configs/experiments/yolo11n_hazydet_controlled.yaml

uv run obj-det models evaluate configs/experiments/yolo11n_hazydet_controlled.yaml \
  --artifact runs/yolo11n/hazydet/controlled/artifact.json \
  --split test

uv run obj-det models optimize configs/experiments/yolo11n_hazydet_controlled.yaml

uv run obj-det models final configs/experiments/yolo11n_hazydet_controlled.yaml \
  --best-trial runs/hpo/yolo11n_hazydet_controlled/best_trial.json
```

Use `model_file` and `search_space_file` when you want reusable configs:

```text
configs/models/
configs/search_spaces/
configs/experiments/
```

Training configs can set `train.loader.num_workers` for parallel loading and
`train.loader.predecode_images: true` to decode the HF image bytes into RAM before training.
