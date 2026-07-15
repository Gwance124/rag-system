#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${CACHE_DIR:-/mnt/nvme2/labuser/.cache/huggingface}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/results/public}"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
EMBEDDING_URL="${EMBEDDING_URL:-http://192.168.3.4:8000/v1}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-nvidia/llama-nv-embed-reasoning-3b}"
EMBEDDING_API_MODEL="${EMBEDDING_API_MODEL:-/model}"
COLLECTION_TAG="${COLLECTION_TAG:-${EMBEDDING_MODEL##*/}}"
LITSEARCH_COLLECTION="${LITSEARCH_COLLECTION:-}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BUILD_INDEXES="${BUILD_INDEXES:-1}"
REBUILD_INDEXES="${REBUILD_INDEXES:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

benchmarks=(
  "litsearch:"
  "mteb:scidocs"
  "mteb:scifact"
  "mteb:nfcorpus"
  "mteb:trec-covid"
)

mkdir -p "$RESULTS_DIR"

for spec in "${benchmarks[@]}"; do
  IFS=: read -r benchmark dataset <<< "$spec"
  name="$benchmark${dataset:+-$dataset}"
  collection="$name-$COLLECTION_TAG"
  if [[ "$name" == "litsearch" && -n "$LITSEARCH_COLLECTION" ]]; then
    collection="$LITSEARCH_COLLECTION"
  fi
  benchmark_args=(--benchmark "$benchmark" --cache-dir "$CACHE_DIR")
  if [[ -n "$dataset" ]]; then
    benchmark_args+=(--dataset "$dataset")
  fi

  needs_dense_index=0
  for mode in dense hybrid; do
    if [[ "$FORCE_RERUN" == "1" || ! -s "$RESULTS_DIR/$name-$mode.json" ]]; then
      needs_dense_index=1
    fi
  done

  if [[ "$BUILD_INDEXES" == "1" && "$needs_dense_index" == "1" ]]; then
    if [[ "$REBUILD_INDEXES" != "1" ]] && curl -fsS "$QDRANT_URL/collections/$collection" >/dev/null 2>&1; then
      echo "Reusing $collection" >&2
    else
      echo "Building $collection" >&2
      "$PYTHON_BIN" "$ROOT_DIR/scripts/build_dense_index.py" \
        "${benchmark_args[@]}" \
        --embedding-url "$EMBEDDING_URL" \
        --embedding-model "$EMBEDDING_MODEL" \
        --embedding-api-model "$EMBEDDING_API_MODEL" \
        --qdrant-url "$QDRANT_URL" \
        --collection "$collection" \
        --batch-size "$BATCH_SIZE"
    fi
  fi

  for mode in sparse dense hybrid; do
    result_file="$RESULTS_DIR/$name-$mode.json"
    if [[ "$FORCE_RERUN" != "1" && -s "$result_file" ]]; then
      echo "Skipping $name ($mode): $result_file exists" >&2
      continue
    fi
    echo "Running $name ($mode)" >&2
    "$PYTHON_BIN" "$ROOT_DIR/scripts/run_public_bench.py" \
      "${benchmark_args[@]}" \
      --mode "$mode" \
      --embedding-url "$EMBEDDING_URL" \
      --embedding-model "$EMBEDDING_MODEL" \
      --embedding-api-model "$EMBEDDING_API_MODEL" \
      --qdrant-url "$QDRANT_URL" \
      --collection "$collection" \
      > "$result_file"
  done
done

echo "Results written to $RESULTS_DIR" >&2
