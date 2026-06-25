#!/bin/bash

DATASET_SAVE_PATH="source_datasets"

HAZYDET_SAVE_PATH="$DATASET_SAVE_PATH/hazydet"
HAZYDET_DATASET_ID="xdedeerha/HazyDet"
HAZYDET_TRAIN_ZIP="train.zip"
HAZYDET_TEST_ZIP="test.zip"
HAZYDET_VAL_ZIP="val.zip"
HAZYDET_REAL_WORLD_ZIP="real_world.zip"

VISDRONE_SAVE_PATH="$DATASET_SAVE_PATH/visdrone"
VISDRONE_TRAIN_URL="https://drive.google.com/file/d/1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn/view?usp=sharing"
VISDRONE_TEST_URL="https://drive.google.com/open?id=1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V"
VISDRONE_VAL_URL="https://drive.google.com/file/d/1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59/view?usp=sharing"
VISDRONE_TRAIN_ZIP="VisDrone2019-DET-train.zip"
VISDRONE_VAL_ZIP="VisDrone2019-DET-val.zip"
VISDRONE_TEST_ZIP="VisDrone2019-DET-test-dev.zip"

XWOD_SAVE_PATH="$DATASET_SAVE_PATH/xwod"
XWOD_DATASET_ID="kuantinglai/exwod"

rm -rf "$DATASET_SAVE_PATH"

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

echo "Downloading visdrone dataset..."
mkdir -p "$VISDRONE_SAVE_PATH"
pushd "$VISDRONE_SAVE_PATH"
uv tool run gdown "$VISDRONE_TEST_URL"
uv tool run gdown "$VISDRONE_TRAIN_URL"
uv tool run gdown "$VISDRONE_VAL_URL"

echo "Extracting visdrone dataset..."
unzip -q "$VISDRONE_TRAIN_ZIP"
unzip -q "$VISDRONE_VAL_ZIP"
unzip -q "$VISDRONE_TEST_ZIP" -d "VisDrone2019-DET-test-dev"
rm "$VISDRONE_TEST_ZIP"
rm "$VISDRONE_TRAIN_ZIP"
rm "$VISDRONE_VAL_ZIP"
popd

echo "Download and extracting xwod dataset..."
uv tool run kaggle datasets download "$XWOD_DATASET_ID" --unzip -p "$XWOD_SAVE_PATH"
mv -r "$XWOD_SAVE_PATH/dataset/*" "$XWOD_SAVE_PATH"
rm -rf "$XWOD_SAVE_PATH/dataset"
