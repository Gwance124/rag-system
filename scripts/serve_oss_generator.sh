#!/usr/bin/env bash
set -euo pipefail

: "${RAG_MODEL_PATH:?Set RAG_MODEL_PATH to the transferred GPT-OSS model directory}"

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
VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-openai/gpt-oss-20b}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-0}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"$PYTHON_BIN" -c '
import torch
import vllm

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
name = torch.cuda.get_device_name(0)
memory_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
if "A100" not in name or memory_gib < 75:
    raise SystemExit(f"expected the dedicated A100 80 GB, found {name} ({memory_gib:.1f} GiB)")
print(f"vLLM={vllm.__version__} GPU={name} ({memory_gib:.1f} GiB)")
'

VLLM_SERVE_HELP="$(vllm serve --help=all 2>&1 || vllm serve --help 2>&1)"
VLLM_VERSION="$(vllm --version 2>&1 || true)"
if [[ "$VLLM_SERVE_HELP" != *"openai"* ]]; then
  echo "Installed $VLLM_VERSION does not provide the OpenAI OSS tool parser." >&2
  exit 2
fi

args=(
  serve "$RAG_MODEL_PATH"
  --host "$VLLM_HOST"
  --port "$VLLM_PORT"
  --served-model-name "$VLLM_SERVED_MODEL_NAME"
  --tensor-parallel-size 1
  --dtype auto
  --max-model-len "$VLLM_MAX_MODEL_LEN"
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --max-num-seqs "$VLLM_MAX_NUM_SEQS"
  --enable-auto-tool-choice
  --tool-call-parser openai
  --generation-config vllm
)

case "$VLLM_ENABLE_PREFIX_CACHING" in
  1|true|TRUE|yes|YES)
    args+=(--enable-prefix-caching)
    ;;
  0|false|FALSE|no|NO)
    args+=(--no-enable-prefix-caching)
    ;;
  *)
    echo "VLLM_ENABLE_PREFIX_CACHING must be 1 or 0" >&2
    exit 2
    ;;
esac

supports_vllm_flag() {
  [[ "$VLLM_SERVE_HELP" == *"$1"* ]]
}

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

echo "Launching GPT-OSS from $RAG_MODEL_PATH" >&2
echo "vLLM arguments: ${args[*]}" >&2
exec vllm "${args[@]}"
