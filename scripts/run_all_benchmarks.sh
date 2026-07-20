#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${CACHE_DIR:-/mnt/nvme2/labuser/.cache/huggingface}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/results/public}"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
EMBEDDING_URL="${EMBEDDING_URL:-http://192.168.3.4:8000/v1}"
EMBEDDING_API_MODEL="${EMBEDDING_API_MODEL:-}"
QUERY_PREFIX="${QUERY_PREFIX:-query: }"
PASSAGE_PREFIX="${PASSAGE_PREFIX:-passage: }"
LITSEARCH_COLLECTION="${LITSEARCH_COLLECTION:-}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BUILD_INDEXES="${BUILD_INDEXES:-1}"
REBUILD_INDEXES="${REBUILD_INDEXES:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
INCLUDE_QASPER="${INCLUDE_QASPER:-0}"
QASPER_SCOPE="${QASPER_SCOPE:-both}"
INCLUDE_SCHOLARGYM="${INCLUDE_SCHOLARGYM:-0}"
SCHOLARGYM_DIR="${SCHOLARGYM_DIR:-$CACHE_DIR/datasets/datasets--shenhao--ScholarGym}"
SCHOLARGYM_PAPER_DB="${SCHOLARGYM_PAPER_DB:-}"
SCHOLARGYM_BENCHMARK_JSONL="${SCHOLARGYM_BENCHMARK_JSONL:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
AUTO_DETECT_MODEL="${AUTO_DETECT_MODEL:-1}"

if [[ "$AUTO_DETECT_MODEL" == "1" ]]; then
  models_url="${EMBEDDING_URL%/}/models"
  echo "Detecting embedding model from $models_url" >&2
  if ! model_response="$(curl -fsS "$models_url")"; then
    echo "Could not query the embedding model endpoint: $models_url" >&2
    exit 2
  fi
  if ! detected_model="$(printf '%s' "$model_response" | "$PYTHON_BIN" -c '
import json
import sys

payload = json.load(sys.stdin)
models = payload.get("data", [])
if not models or not models[0].get("id"):
    raise SystemExit("/models response did not contain data[0].id")
print(models[0]["id"])
')"; then
    echo "Could not parse the embedding model response from $models_url" >&2
    exit 2
  fi
  EMBEDDING_MODEL="$detected_model"
  EMBEDDING_API_MODEL="${EMBEDDING_API_MODEL:-$detected_model}"
else
  EMBEDDING_MODEL="${EMBEDDING_MODEL:-nvidia/llama-nv-embed-reasoning-3b}"
  EMBEDDING_API_MODEL="${EMBEDDING_API_MODEL:-$EMBEDDING_MODEL}"
fi

MODEL_TAG="${MODEL_TAG:-${EMBEDDING_MODEL##*/}}"
COLLECTION_TAG="${COLLECTION_TAG:-$MODEL_TAG}"
FAILURES_DIR="${FAILURES_DIR:-$RESULTS_DIR/failures/$MODEL_TAG}"
FAILURE_SUMMARY="$FAILURES_DIR/summary.txt"
failure_records=()
mkdir -p "$FAILURES_DIR"
: > "$FAILURE_SUMMARY"

record_failure() {
  local benchmark="$1"
  local mode="$2"
  local stage="$3"
  local status="$4"
  local log_file="$5"
  failure_records+=("$benchmark|$mode|$stage|$status|$log_file")
  {
    printf 'FAILED %s (%s) at %s; exit code %s\n' "$benchmark" "$mode" "$stage" "$status"
    printf 'Log: %s\n' "$log_file"
    printf 'Last error output:\n'
    tail -n 20 "$log_file" 2>/dev/null || true
    printf '\n'
  } >> "$FAILURE_SUMMARY"
}

benchmarks=(
  "litsearch:"
  "mteb:scidocs"
  "mteb:scifact"
  "mteb:nfcorpus"
  "mteb:trec-covid"
)

if [[ "$INCLUDE_QASPER" == "1" ]]; then
  case "$QASPER_SCOPE" in
    global|paper)
      benchmarks+=("qasper::$QASPER_SCOPE")
      ;;
    both)
      benchmarks+=("qasper::global" "qasper::paper")
      ;;
    *)
      echo "QASPER_SCOPE must be global, paper, or both; got: $QASPER_SCOPE" >&2
      exit 2
      ;;
  esac
fi

if [[ "$INCLUDE_SCHOLARGYM" == "1" ]]; then
  benchmarks+=("scholargym:")
fi

SPARSE_RESULTS_DIR="$RESULTS_DIR/sparse"
MODEL_RESULTS_DIR="$RESULTS_DIR/$MODEL_TAG"
mkdir -p "$SPARSE_RESULTS_DIR" "$MODEL_RESULTS_DIR"

for spec in "${benchmarks[@]}"; do
  IFS=: read -r benchmark dataset qasper_scope <<< "$spec"
  name="$benchmark${dataset:+-$dataset}"
  if [[ "$benchmark" == "qasper" ]]; then
    name="$name-$qasper_scope"
  fi
  collection_name="$benchmark${dataset:+-$dataset}"
  collection="$collection_name-$COLLECTION_TAG"
  if [[ "$name" == "litsearch" && -n "$LITSEARCH_COLLECTION" ]]; then
    collection="$LITSEARCH_COLLECTION"
  fi
  benchmark_args=(--benchmark "$benchmark" --cache-dir "$CACHE_DIR")
  if [[ -n "$dataset" ]]; then
    benchmark_args+=(--dataset "$dataset")
  fi
  if [[ "$benchmark" == "qasper" ]]; then
    benchmark_args+=(--qasper-scope "$qasper_scope")
  fi
  if [[ "$benchmark" == "scholargym" ]]; then
    benchmark_args+=(--scholargym-dir "$SCHOLARGYM_DIR")
    if [[ -n "$SCHOLARGYM_PAPER_DB" ]]; then
      benchmark_args+=(--scholargym-paper-db "$SCHOLARGYM_PAPER_DB")
    fi
    if [[ -n "$SCHOLARGYM_BENCHMARK_JSONL" ]]; then
      benchmark_args+=(--scholargym-benchmark "$SCHOLARGYM_BENCHMARK_JSONL")
    fi
  fi

  needs_dense_index=0
  if [[ "$FORCE_RERUN" == "1" || ! -s "$MODEL_RESULTS_DIR/$name-dense.json" || ! -s "$MODEL_RESULTS_DIR/$name-hybrid.json" ]]; then
    needs_dense_index=1
  fi

  index_failed=0
  index_log="$FAILURES_DIR/$name-index.log"
  if [[ "$BUILD_INDEXES" == "1" && "$needs_dense_index" == "1" ]]; then
    if [[ "$REBUILD_INDEXES" != "1" ]] && curl -fsS "$QDRANT_URL/collections/$collection" >/dev/null 2>&1; then
      echo "Reusing $collection" >&2
    else
      echo "Building $collection" >&2
      if "$PYTHON_BIN" "$ROOT_DIR/scripts/build_dense_index.py" \
          "${benchmark_args[@]}" \
          --embedding-url "$EMBEDDING_URL" \
          --embedding-model "$EMBEDDING_MODEL" \
          --embedding-api-model "$EMBEDDING_API_MODEL" \
          --query-prefix "$QUERY_PREFIX" \
          --passage-prefix "$PASSAGE_PREFIX" \
          --qdrant-url "$QDRANT_URL" \
          --collection "$collection" \
          --batch-size "$BATCH_SIZE" > >(tee "$index_log" >&2) 2>&1; then
        :
      else
        status=$?
        index_failed=1
        record_failure "$name" "dense/hybrid" "build index" "$status" "$index_log"
      fi
    fi
  fi

  for mode in sparse dense hybrid; do
    if [[ "$mode" == "sparse" ]]; then
      result_file="$SPARSE_RESULTS_DIR/$name-$mode.json"
    else
      result_file="$MODEL_RESULTS_DIR/$name-$mode.json"
    fi
    if [[ "$FORCE_RERUN" != "1" && -s "$result_file" ]]; then
      echo "Skipping $name ($mode): $result_file exists" >&2
      continue
    fi
    if [[ "$mode" != "sparse" && "$index_failed" == "1" ]]; then
      continue
    fi
    echo "Running $name ($mode)" >&2
    result_log="$FAILURES_DIR/$name-$mode.log"
    temporary_result="$result_file.tmp.$$"
    if "$PYTHON_BIN" "$ROOT_DIR/scripts/run_public_bench.py" \
        "${benchmark_args[@]}" \
        --mode "$mode" \
        --embedding-url "$EMBEDDING_URL" \
        --embedding-model "$EMBEDDING_MODEL" \
        --embedding-api-model "$EMBEDDING_API_MODEL" \
        --query-prefix "$QUERY_PREFIX" \
        --passage-prefix "$PASSAGE_PREFIX" \
        --qdrant-url "$QDRANT_URL" \
        --collection "$collection" \
        >"$temporary_result" 2> >(tee "$result_log" >&2); then
      mv "$temporary_result" "$result_file"
    else
      status=$?
      rm -f "$temporary_result"
      record_failure "$name" "$mode" "run benchmark" "$status" "$result_log"
    fi
  done
done

echo "Sparse results written to $SPARSE_RESULTS_DIR" >&2
echo "Model results written to $MODEL_RESULTS_DIR" >&2

if ((${#failure_records[@]} > 0)); then
  echo >&2
  echo "Benchmark sweep completed with ${#failure_records[@]} failure(s)." >&2
  echo "Failure summary: $FAILURE_SUMMARY" >&2
  for record in "${failure_records[@]}"; do
    IFS='|' read -r failed_benchmark failed_mode failed_stage failed_status failed_log <<< "$record"
    echo "FAILED $failed_benchmark [$failed_mode] at $failed_stage (exit $failed_status)" >&2
    tail -n 5 "$failed_log" 2>/dev/null | sed 's/^/  /' >&2 || true
  done
  exit 1
fi

echo "Benchmark sweep completed without failures." >&2
