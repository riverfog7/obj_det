#!/bin/bash

CONFIG_DIR="${1:-configs/datasets}"
OUTPUT_DIR="${2:-datasets}"

rm -rf "$OUTPUT_DIR"
for file in "$CONFIG_DIR"/*.yaml ; do
    basename="$(basename $file)"
    result_dir="$OUTPUT_DIR/${basename%.*}"
    uv run obj-det datasets convert "$file" "$result_dir"
done
