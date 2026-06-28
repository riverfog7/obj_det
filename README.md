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

It downloads and extracts HazyDet, VisDrone, XWOD, and DAWN into:

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
```

A config describes where the raw dataset is and how to interpret it.

Important fields:

```yaml
key: xwod                     # dataset key used in output records
root: source_datasets/xwod    # raw dataset root
source_format: yolo           # coco | visdrone_det | yolo | yolo_noyml

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

DAWN currently has one source split:

```text
full
```

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
