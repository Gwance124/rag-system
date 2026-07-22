#!/usr/bin/env bash
set -euo pipefail

MODEL="${VLLM_MODEL:-${EMBEDDING_MODEL:-nvidia/llama-nv-embed-reasoning-3b}}"
SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-/model}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
TASK="${VLLM_TASK:-embed}"

args=(serve "$MODEL" --host "$HOST" --port "$PORT" --served-model-name "$SERVED_MODEL_NAME")
if [[ -n "$TASK" ]]; then
  args+=(--task "$TASK")
fi
if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
  read -r -a extra_args <<< "$VLLM_EXTRA_ARGS"
  args+=("${extra_args[@]}")
fi

exec vllm "${args[@]}"
