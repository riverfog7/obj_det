#!/bin/bash

DATASET_SAVE_PATH="source_datasets"

HAZYDET_SAVE_PATH="$DATASET_SAVE_PATH/hazydet"
HAZYDET_DATASET_ID="xdedeerha/HazyDet"
HAZYDET_TRAIN_ZIP="train.zip"
HAZYDET_TEST_ZIP="test.zip"
HAZYDET_VAL_ZIP="val.zip"
HAZYDET_REAL_WORLD_ZIP="real_world.zip"

rm -rf "$DATASET_SAVE_PATH

echo "Downloading hazydet dataset..."
mkdir -p "$HAZYDET_SAVE_PATH"
uv tool run hf download --repo-type dataset "$HAZYDET_DATASET_ID" "$HAZYDET_TRAIN_ZIP" --local-dir "$HAZYDET_SAVE_PATH"
uv tool run hf download --repo-type dataset "$HAZYDET_DATASET_ID" "$HAZYDET_VAL_ZIP" --local-dir "$HAZYDET_SAVE_PATH"
uv tool run hf download --repo-type dataset "$HAZYDET_DATASET_ID" "$HAZYDET_TEST_ZIP" --local-dir "$HAZYDET_SAVE_PATH"
uv tool run hf download --repo-type dataset "$HAZYDET_DATASET_ID" "$HAZYDET_REAL_WORLD_ZIP" --local-dir "$HAZYDET_SAVE_PATH"

echo "Extracting hazydet dataset..."
pushd "$HAZYDET_SAVE_PATH"
unzip -q "$HAZYDET_TRAIN_ZIP"
unzip -q "$HAZYDET_VAL_ZIP"
unzip -q "$HAZYDET_TEST_ZIP"
unzip -q "$HAZYDET_REAL_WORLD_ZIP"

rm "$HAZYDET_TRAIN_ZIP"
rm "$HAZYDET_VAL_ZIP"
rm "$HAZYDET_TEST_ZIP"
rm "$HAZYDET_REAL_WORLD_ZIP"
popd
