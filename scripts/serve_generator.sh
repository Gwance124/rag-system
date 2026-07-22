#!/usr/bin/env bash
set -euo pipefail

: "${RAG_MODEL_PATH:?Set RAG_MODEL_PATH to the transferred Qwen3.6-27B directory}"

if [[ ! -d "$RAG_MODEL_PATH" ]]; then
  echo "Model directory does not exist: $RAG_MODEL_PATH" >&2
  exit 2
fi
for required_file in config.json model.safetensors.index.json tokenizer_config.json; do
  if [[ ! -f "$RAG_MODEL_PATH/$required_file" ]]; then
    echo "Model directory is missing $required_file: $RAG_MODEL_PATH" >&2
    exit 2
  fi
done

PYTHON_BIN="${PYTHON_BIN:-python}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-qwen3.6-27b}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"
VLLM_TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-qwen3_coder}"
VLLM_REASONING_CONFIG="${VLLM_REASONING_CONFIG:-{\"reasoning_start_str\":\"<think>\",\"reasoning_end_str\":\"</think>\"}}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_ATTENTION_BACKEND

"$PYTHON_BIN" -c '
from importlib.metadata import version

import flashinfer
import torch
import vllm

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
name = torch.cuda.get_device_name(0)
memory_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
if "A100" not in name or memory_gib < 75:
    raise SystemExit(f"expected the dedicated A100 80 GB, found {name} ({memory_gib:.1f} GiB)")
flashinfer_version = version("flashinfer-python")
print(
    f"vLLM={vllm.__version__} FlashInfer={flashinfer_version} "
    f"GPU={name} ({memory_gib:.1f} GiB)"
)
'

args=(
  serve "$RAG_MODEL_PATH"
  --host "$VLLM_HOST"
  --port "$VLLM_PORT"
  --served-model-name "$VLLM_SERVED_MODEL_NAME"
  --tensor-parallel-size 1
  --dtype bfloat16
  --max-model-len "$VLLM_MAX_MODEL_LEN"
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --max-num-seqs "$VLLM_MAX_NUM_SEQS"
  --reasoning-parser qwen3
  --reasoning-config "$VLLM_REASONING_CONFIG"
  --enable-auto-tool-choice
  --tool-call-parser "$VLLM_TOOL_CALL_PARSER"
  --generation-config vllm
  --no-enable-prefix-caching
)

# vLLM's CLI surface changes between releases. FlashInfer is selected through
# VLLM_ATTENTION_BACKEND; optional serve flags are added only when supported by
# the exact installed build and their absence is recorded rather than hidden.
VLLM_SERVE_HELP="$(vllm serve --help=all 2>&1 || vllm serve --help 2>&1)"
VLLM_VERSION="$(vllm --version 2>&1 || true)"

if [[ "$VLLM_TOOL_CALL_PARSER" == "qwen3_coder" && "$VLLM_SERVE_HELP" != *"qwen3_coder"* ]]; then
  echo "Installed $VLLM_VERSION does not provide the qwen3_coder tool-call parser." >&2
  echo "Upgrade the generator vLLM environment; do not substitute hermes for a Qwen3.6 benchmark run." >&2
  exit 2
fi

if [[ "$VLLM_SERVE_HELP" != *"--reasoning-config"* ]]; then
  echo "Installed $VLLM_VERSION does not provide --reasoning-config." >&2
  echo "This generator cannot enforce --thinking-token-budget safely." >&2
  exit 2
fi

supports_vllm_flag() {
  [[ "$VLLM_SERVE_HELP" == *"$1"* ]]
}

if supports_vllm_flag "--language-model-only"; then
  args+=(--language-model-only)
elif supports_vllm_flag "--limit-mm-per-prompt"; then
  args+=(--limit-mm-per-prompt '{"image":0,"video":0}')
  echo "vLLM lacks --language-model-only; disabling image/video with --limit-mm-per-prompt" >&2
else
  echo "Warning: this vLLM cannot explicitly disable the multimodal encoder" >&2
fi

if supports_vllm_flag "--enable-per-request-metrics"; then
  args+=(--enable-per-request-metrics)
else
  echo "Warning: per-request vLLM timing metrics are unavailable in this build" >&2
fi

if supports_vllm_flag "--enable-server-load-tracking"; then
  args+=(--enable-server-load-tracking)
else
  echo "Warning: vLLM server-load tracking is unavailable in this build" >&2
fi

if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
  read -r -a extra_args <<< "$VLLM_EXTRA_ARGS"
  args+=("${extra_args[@]}")
fi

echo "Launching pinned local model from $RAG_MODEL_PATH" >&2
echo "vLLM arguments: ${args[*]}" >&2
exec vllm "${args[@]}"
