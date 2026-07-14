#!/usr/bin/env bash

set -euo pipefail

CONFIG_DIR="${DATASET_CONFIG_DIR:-configs/datasets}"
OUTPUT_DIR="${DATASET_OUTPUT_ROOT:-datasets}"
FORCE=false

usage() {
    cat <<'EOF'
Usage: scripts/convert.sh [--force] <dataset> [<dataset> ...]
       scripts/convert.sh [--force] all

Dataset keys match YAML filenames in configs/datasets.
Existing output directories are preserved unless --force is supplied.
EOF
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
    shopt -s nullglob
    selected=()
    for config in "$CONFIG_DIR"/*.yaml; do
        selected+=("$(basename "${config%.yaml}")")
    done
fi

if ((${#selected[@]} == 0)); then
    echo "No dataset configs found in $CONFIG_DIR" >&2
    exit 1
fi

for dataset in "${selected[@]}"; do
    config="$CONFIG_DIR/$dataset.yaml"
    result_dir="$OUTPUT_DIR/$dataset"
    if [[ ! -f "$config" ]]; then
        echo "Unknown dataset config: $config" >&2
        exit 2
    fi
    if [[ -e "$result_dir" && "$FORCE" != true ]]; then
        echo "Refusing to replace existing dataset output: $result_dir" >&2
        echo "Use --force to replace only this dataset." >&2
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"
for dataset in "${selected[@]}"; do
    config="$CONFIG_DIR/$dataset.yaml"
    result_dir="$OUTPUT_DIR/$dataset"
    if [[ -e "$result_dir" ]]; then
        rm -rf -- "$result_dir"
    fi
    uv run obj-det datasets convert "$config" "$result_dir"
done
