#!/usr/bin/env bash

set -euo pipefail

DATASET_SAVE_PATH="${SOURCE_DATASET_ROOT:-source_datasets}"
AVAILABLE_DATASETS=(hazydet visdrone xwod dawn exdark voc2007 cityscapes)
FORCE=false

usage() {
    cat <<'EOF'
Usage: scripts/download.sh [--force] <dataset> [<dataset> ...]
       scripts/download.sh [--force] all

Datasets: hazydet, visdrone, xwod, dawn, exdark, voc2007, cityscapes

Existing dataset directories are preserved unless --force is supplied.
EOF
}

dataset_path() {
    printf '%s/%s\n' "$DATASET_SAVE_PATH" "$1"
}

prepare_dataset_path() {
    local path="$1"

    if [[ -e "$path" ]]; then
        if [[ "$FORCE" != true ]]; then
            echo "Refusing to replace existing dataset directory: $path" >&2
            echo "Use --force to replace only this dataset." >&2
            return 1
        fi
        rm -rf -- "$path"
    fi
    mkdir -p "$path"
}

download_hazydet() {
    local path
    path="$(dataset_path hazydet)"
    local dataset_id="xdedeerha/HazyDet"
    local archives=(train.zip val.zip test.zip real_world.zip)

    echo "Downloading HazyDet..."
    for archive in "${archives[@]}"; do
        uv tool run hf download --repo-type dataset "$dataset_id" "$archive" --local-dir "$path"
    done

    echo "Extracting HazyDet..."
    for archive in "${archives[@]}"; do
        unzip -q "$path/$archive" -d "$path"
        rm "$path/$archive"
    done
}

download_visdrone() {
    local path
    path="$(dataset_path visdrone)"
    local train_url="https://drive.google.com/file/d/1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn/view?usp=sharing"
    local val_url="https://drive.google.com/file/d/1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59/view?usp=sharing"
    local test_url="https://drive.google.com/open?id=1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V"

    echo "Downloading VisDrone..."
    pushd "$path" >/dev/null
    uv tool run gdown "$train_url"
    uv tool run gdown "$val_url"
    uv tool run gdown "$test_url"

    echo "Extracting VisDrone..."
    unzip -q VisDrone2019-DET-train.zip
    unzip -q VisDrone2019-DET-val.zip
    unzip -q VisDrone2019-DET-test-dev.zip -d VisDrone2019-DET-test-dev
    rm VisDrone2019-DET-train.zip VisDrone2019-DET-val.zip VisDrone2019-DET-test-dev.zip
    popd >/dev/null
}

download_xwod() {
    local path
    path="$(dataset_path xwod)"

    echo "Downloading and extracting XWOD..."
    uv tool run kaggle datasets download kuantinglai/exwod --unzip -p "$path"
    mv "$path"/dataset/* "$path"
    rmdir "$path/dataset"
}

download_dawn() {
    local path
    path="$(dataset_path dawn)"

    echo "Downloading and extracting DAWN..."
    uv tool run kaggle datasets download shuvoalok/dawn-dataset --unzip -p "$path"
}

download_exdark() {
    local path
    path="$(dataset_path exdark)"
    local images_zip="exdark_images.zip"
    local annotations_zip="exdark_annotations.zip"

    echo "Downloading ExDark..."
    uv tool run gdown 1BHmPgu8EsHoFDDkMGLVoXIlCth2dW6Yx -O "$path/$images_zip"
    uv tool run gdown 1P3iO3UYn7KoBi5jiUkogJq96N6maZS1i -O "$path/$annotations_zip"
    curl -LfsS \
        https://raw.githubusercontent.com/cs-chan/Exclusively-Dark-Image-Dataset/master/Groundtruth/imageclasslist.txt \
        -o "$path/imageclasslist.txt"

    echo "Extracting ExDark..."
    unzip -q "$path/$images_zip" -d "$path"
    unzip -q "$path/$annotations_zip" -d "$path"
    rm "$path/$images_zip" "$path/$annotations_zip"
    rm -rf "$path/__MACOSX"
}

download_voc2007() {
    local path
    path="$(dataset_path voc2007)"
    local base_url="https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007"
    local archives=(VOCtrainval_06-Nov-2007.tar VOCtest_06-Nov-2007.tar)

    echo "Downloading Pascal VOC 2007..."
    for archive in "${archives[@]}"; do
        curl -LfsS "$base_url/$archive" -o "$path/$archive"
        tar -xf "$path/$archive" -C "$path"
        rm "$path/$archive"
    done
}

download_cityscapes() {
    local path
    path="$(dataset_path cityscapes)"
    local archives=(leftImg8bit_trainvaltest.zip gtFine_trainvaltest.zip)

    echo "Downloading Cityscapes (account required)..."
    uv tool run --from cityscapesscripts csDownload -d "$path" "${archives[@]}"
    for archive in "${archives[@]}"; do
        unzip -q "$path/$archive" -d "$path"
        rm "$path/$archive"
    done
}

is_available_dataset() {
    local requested="$1"
    local dataset
    for dataset in "${AVAILABLE_DATASETS[@]}"; do
        [[ "$requested" == "$dataset" ]] && return 0
    done
    return 1
}

selected=()
while (($#)); do
    case "$1" in
        --force)
            FORCE=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            selected+=("$@")
            break
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            selected+=("$1")
            ;;
    esac
    shift
done

if ((${#selected[@]} == 0)); then
    usage >&2
    exit 2
fi

if [[ " ${selected[*]} " == *" all "* ]]; then
    if ((${#selected[@]} != 1)); then
        echo "'all' cannot be combined with dataset keys." >&2
        exit 2
    fi
    selected=("${AVAILABLE_DATASETS[@]}")
fi

for dataset in "${selected[@]}"; do
    if ! is_available_dataset "$dataset"; then
        echo "Unknown dataset: $dataset" >&2
        usage >&2
        exit 2
    fi
done

for dataset in "${selected[@]}"; do
    prepare_dataset_path "$(dataset_path "$dataset")"
    "download_$dataset"
done
