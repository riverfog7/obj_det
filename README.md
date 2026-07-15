# obj-det

The current workflow is:

```text
raw dataset files in source_datasets/
    -> source adapter
    -> canonical ImageRecord objects
    -> Hugging Face DatasetDict saved locally or pushed to Hub
```

Supported source formats:

- `bdd100k`
- `cityscapes`
- `coco`
- `exdark`
- `pascal_voc`
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

Downloads are explicit and dataset-scoped:

```bash
bash scripts/download.sh voc2007
bash scripts/download.sh carpk udacity
bash scripts/download.sh all
```

Raw files are stored under:

```text
source_datasets/
```

An existing dataset directory is never replaced implicitly. To reacquire one
dataset, use `--force`; only that dataset's directory is removed:

```bash
bash scripts/download.sh --force voc2007
```

Available download keys are `acdc`, `bdd100k`, `carpk`, `cityscapes`, `dawn`,
`exdark`, `hazydet`, `udacity`, `visdrone`, `voc2007`, and `xwod`. The helper
uses the official or upstream distribution path for each dataset. Hugging Face,
Google Drive, Kaggle, Cityscapes, and Roboflow commands may require their normal
local authentication.

BDD100K is downloaded through its portal. Point the helper at the two staged
official archives:

```bash
export BDD100K_IMAGES_ARCHIVE=/downloads/bdd100k_images_100k.zip
export BDD100K_LABELS_ARCHIVE=/downloads/bdd100k_det_20_labels_trainval.zip
bash scripts/download.sh bdd100k
```

Cityscapes uses `csDownload` and prompts for the account associated with the
dataset license. ACDC, CARPK, Udacity, and Pascal VOC use their public upstream
package endpoints.

Every configured model input is one ordinary RGB image and every target is a
list of 2D objects. nuScenes is intentionally not supported because its native
task requires synchronized multi-camera/LiDAR data, calibration, and 3D box
projection. Cityscapes was recorded with stereo hardware, but this integration
uses only `leftImg8bit`; ACDC likewise uses only its released RGB detection
images and COCO boxes.

## Dataset YAML configs

Dataset configs live in:

```text
configs/datasets/
```

The repository contains 13 raw-source configs and one additional ref for the
locally merged dataset:

| Dataset key | Source format | Converted splits | Controlled plan |
| --- | --- | --- | --- |
| `acdc` | COCO | train, val | no: public test labels unavailable |
| `bdd100k` | BDD100K Scalabel JSON | train, val | no: public test labels unavailable |
| `carpk` | COCO | train, val, test | yes |
| `cityscapes` | Cityscapes polygons | train, val | no: public test labels unavailable |
| `dawn` | YOLO without YAML | train, val, test | yes: project-defined split |
| `exdark` | ExDark | train, val, test | yes |
| `hazydet` | YOLO | train, val, test | yes |
| `hazydet_clear` | YOLO | train, val, test | yes |
| `hazydet_real` | YOLO | train, test | no: validation split unavailable |
| `udacity` | COCO | train, val, test | yes |
| `visdrone` | VisDrone detection TXT | train, val, test | yes |
| `voc2007` | Pascal VOC XML | train, val, test | yes |
| `xwod` | YOLO | train, val, test | yes |
| `merged_traffic6` | merged HF DatasetDict | train, val, test | yes |

A config describes where the raw dataset is and how to interpret it.

Important fields:

```yaml
key: xwod                     # dataset key used in output records
root: source_datasets/xwod    # raw dataset root
source_format: yolo           # see the supported source-format list above

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
- BDD100K configs use `images` and one Scalabel `annotations` JSON file.
- Cityscapes configs use `images` and polygon `annotations` directories.
- Pascal VOC configs use `images`, `annotations`, and `image_set`.
- VisDrone configs use `images` and `annotations`.
- YOLO configs use `images` and `labels`; class names come from `root/data.yaml`.
- `yolo_noyml` configs use `images` and `labels`; class names come from `class_names`.
- ExDark configs use `images`, `annotations`, and `imageclasslist`.

The DAWN archive is an unsplit image pool. `scripts/download.sh dawn` creates a
deterministic 80/10/10 train/validation/test split. Byte-identical images stay
in the same split to avoid direct duplicate leakage. This is a project-defined
split, not an official DAWN benchmark split.

ExDark uses the official `train`, `val`, and `test` split ids from
`imageclasslist.txt`. Upstream marks ExDark as non-commercial research use.

Optional split field:

```yaml
output_split: real_tr
```

This renames the split in the saved `DatasetDict`. It is useful when a source split name would otherwise be merged or inferred strangely by Hugging Face tooling.

## Convert a dataset

The selective helper converts one or more configured datasets and refuses to
replace existing output unless `--force` is explicit:

```bash
bash scripts/convert.sh voc2007
bash scripts/convert.sh carpk udacity
bash scripts/convert.sh --force voc2007
bash scripts/convert.sh all
```

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

## Merge converted datasets

Merge the obtainable converted datasets into one local six-class dataset:

```bash
uv run python scripts/merge_datasets.py
```

Use `--force` to replace an existing merged output, or override the paths with
`--input-root` and `--output`. The default output is
`datasets/merged_traffic6`.

The merge includes ACDC, CARPK, DAWN, ExDark, HazyDet, HazyDet-clear,
HazyDet-real, VisDrone, VOC2007, and XWOD. It omits BDD100K because the current
Detection 2020 package is unavailable, Cityscapes because its official source
is account-gated, and Udacity because the obtainable source does not carry an
authoritative train/validation/test assignment.

The final classes are `person`, `bicycle`, `motorcycle`, `car`, `bus`, and
`truck`. The fixed merge policy harmonizes pedestrian/people/rider as person,
bike as bicycle, motorbike/motor/tricycle/awning-tricycle as motorcycle, and
van as car. CARPK's numeric `0` label maps to car only within CARPK. All other
classes, ignored objects, and images left without a retained object are
removed.

Surviving rows keep their original `train`, `val`, or `test` split. Exact
duplicates and aligned HazyDet clear/hazy variants are treated as one lineage.
When a lineage spans splits, only rows already in the most protected split are
retained, using `test > val > train`; rows are never reassigned. Richer aligned
annotations are shared across retained variants when image dimensions match.

Run the complete controlled model plan on the merged dataset with:

```bash
uv run obj-det models plan run --all configs/plans/merged_traffic6_controlled.yaml
```

The combined source licenses include non-commercial and dataset-specific
terms, so the script deliberately creates a local artifact and has no upload
option. Inspect `datasets/merged_traffic6/merge_manifest.json` for source,
filtering, deduplication, split, class, provenance, and reproducibility counts.

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
from obj_det.models.schemas import (
    AugmentationConfig,
    DataLoaderConfig,
    EvalConfig,
    ModelConfig,
    PreprocessConfig,
    TrainConfig,
)

hf_ds = load_from_disk("datasets/hazydet")

model_cfg = ModelConfig(
    key="fasterrcnn_r50",
    backend="torchvision",
    model_name_or_path="fasterrcnn_resnet50_fpn",
)
adapter = model_adapter_from_config(model_cfg)
preprocess = PreprocessConfig(image_size=640)
augmentation = AugmentationConfig(policy="basic", horizontal_flip_p=0.5, color_jitter_strength=0.1)

train_cfg = TrainConfig(
    run_key="fasterrcnn_hazydet_seed0",
    classes=["person", "bicycle", "motorcycle", "car", "bus", "truck"],
    label_mode="meta",
    output_dir="runs/fasterrcnn/hazydet/seed0",
    preprocess=preprocess,
    augmentation=augmentation,
    loader=DataLoaderConfig(num_workers=16, persistent_workers=True, prefetch_factor=4),
    max_epochs=50,
    batch_size=16,
)

eval_cfg = EvalConfig(
    classes=train_cfg.classes,
    label_mode=train_cfg.label_mode,
    preprocess=preprocess,
)

artifact = adapter.train(hf_ds["train"], hf_ds["validation"], train_cfg)
result = adapter.evaluate(hf_ds["test"], artifact, eval_cfg)
print(result.primary_metric, result.primary_metric_value)
```

Current backend status:

- `torchvision`: Faster R-CNN, RetinaNet, FCOS, and Mask R-CNN box-only through Hugging Face `Trainer`.
- `hf_trainer`: Transformers `Trainer` adapter with COCO-style image-processor targets and canonical predictions.
- `ultralytics`: HF-backed Ultralytics `DetectionTrainer` adapter; no YOLO folder export is used.

Model configs live in `configs/models/`. The current controlled matrix includes:

- Ultralytics: YOLO26 n/s/m, YOLO11 n/s/m, YOLOv8 n/s/m.
- HF Trainer: RT-DETR, D-FINE nano/small, RF-DETR nano/small/medium/base, DETR, Conditional DETR, Deformable DETR, YOLOS tiny/small.
- TorchVision: Faster R-CNN, RetinaNet, FCOS, and Mask R-CNN box-only.

Controlled experiment plans live in `configs/plans/`. Plans are enabled for
CARPK, ExDark, HazyDet, HazyDet-clear, Udacity, VisDrone, Pascal VOC 2007, and
XWOD. Each plan composes one HF dataset reference, the shared `traffic6` class
space, the controlled recipe, and the 25-model detection group. Together they
expand to 200 validated `ExperimentConfig` objects. Dataset refs without a
public annotated train/validation/test split remain fail-closed and have no
controlled plan. `configs/experiments/` remains supported for direct/debug
single-run configs, but plans are the preferred scalable source of truth.

The controlled recipe fixes AdamW, a one-epoch warmup followed by a 50-epoch
cosine schedule, checkpointing, and early stopping for every backend. HPO uses
`configs/search_spaces/global_learning_rate.yaml`: eight independent TPE trials
per model tune only canonical `learning_rate` for 10 epochs while retaining the
50-epoch scheduler horizon. Backend defaults contain runtime sizing only.

```python
from obj_det.models.plan import (
    ExperimentPlanRunner,
    load_and_expand_experiment_plan,
    load_experiment_plan,
    write_resolved_experiments,
)
from obj_det.models.runner import ExperimentRunner

exps = load_and_expand_experiment_plan("configs/plans/hazydet_controlled.yaml")
for exp in exps:
    ExperimentRunner(exp).optimize()

plan = load_experiment_plan("configs/plans/hazydet_controlled.yaml")
write_resolved_experiments(plan, "runs/resolved_configs/hazydet_controlled")
ExperimentPlanRunner(plan).optimize_all(model_keys=["yolo26s", "rtdetr_r50vd"])
```

The generated resolved YAMLs are artifacts for inspection/reproduction, not the
main source of truth.

Optional backend smoke tests are available with:

```bash
OBJ_DET_RUN_BACKEND_SMOKE=1 uv run python -m unittest tests.models.test_backend_smoke
```

## Model CLI

Plan configs are the preferred CLI entrypoint for model sweeps. You can run each
stage manually:

```bash
uv run obj-det models plan list configs/plans/hazydet_controlled.yaml
uv run obj-det models plan resolve configs/plans/hazydet_controlled.yaml --model yolo26m
uv run obj-det models plan optimize configs/plans/hazydet_controlled.yaml --model yolo26m
uv run obj-det models plan final configs/plans/hazydet_controlled.yaml --model yolo26m
```

Or run the full plan pipeline for explicit models:

```bash
uv run obj-det models plan run configs/plans/hazydet_controlled.yaml --model yolo26m
```

`plan run` requires `--model` or explicit `--all` so a whole benchmark is not
launched accidentally.

Direct YAML configs under `configs/experiments/` are still supported for one-off
debugging; this repo keeps only `yolo26m_hazydet_controlled.yaml` as that debug
fixture.

```bash
uv run obj-det models train configs/experiments/yolo26m_hazydet_controlled.yaml

uv run obj-det models evaluate configs/experiments/yolo26m_hazydet_controlled.yaml \
  --artifact runs/yolo26m/hazydet/controlled/artifact.json \
  --split test

uv run obj-det models optimize configs/experiments/yolo26m_hazydet_controlled.yaml

uv run obj-det models final configs/experiments/yolo26m_hazydet_controlled.yaml \
  --best-trial runs/hpo/yolo26m_hazydet_controlled/best_trial.json
```

Reusable model-training config directories:

```text
configs/dataset_refs/    # HF dataset path and split names
configs/class_spaces/    # labels and native/meta label mode
configs/recipes/         # protocol defaults, preprocess, augmentation, HPO/final defaults
configs/model_groups/    # reusable model lists
configs/plans/           # preferred scalable experiment entrypoints
configs/models/          # backend/model identity
configs/search_spaces/   # shared Optuna learning-rate search space
configs/experiments/     # optional direct/debug resolved configs
```

Training configs can set `train.logging_steps` to control scalar training-log
cadence for all backends. It means every N training steps/batches; default
experiment configs use `logging_steps: 100`. They can also set
`train.loader.num_workers`, `prefetch_factor`, `decode_backend: pil|opencv`, and
`profile_every_n` for data-path tuning. `predecode_images: true` is diagnostic
and memory-heavy; use it to prove decode is the bottleneck, not as the default
large-dataset training mode.

Model runs support scalar-only logging for all CLI flows:

```yaml
logging:
  backends: [local, wandb]  # allowed: none, local, wandb
  local:
    path: null             # default: <run-output>/logs/events.jsonl
  wandb:
    project: obj-det
    entity: null
    group: null
    name: null
    mode: online           # online, offline, disabled
    tags: []
```

Logged values include backend train scalars and evaluator metrics including
per-class, per-condition, per-domain, and per-size results. `models optimize`
creates one logging run per HPO trial, grouped by study name. `models final`
creates one logging run per seed, grouped by final run name. Images, prediction
previews, tables, and plots are not logged.
