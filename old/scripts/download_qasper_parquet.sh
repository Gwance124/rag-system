#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 OUTPUT_DIR" >&2
  exit 2
fi

output_dir="$1"
base_url="https://huggingface.co/datasets/allenai/qasper/resolve/refs%2Fconvert%2Fparquet/qasper"
mkdir -p "$output_dir"

for split in train validation test; do
  destination="$output_dir/$split.parquet"
  temporary="$destination.tmp"
  echo "Downloading QASPER $split to $destination" >&2
  curl -fL --retry 3 --retry-delay 2 \
    "$base_url/$split/0000.parquet" \
    -o "$temporary"
  mv "$temporary" "$destination"
done

echo "QASPER Parquet files written to $output_dir"
