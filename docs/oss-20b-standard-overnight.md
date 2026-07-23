# GPT-OSS-20B Standard overnight run

This path adapts `BrowseComp-Plus/search_agent/oss_client.py` at commit
`046949032b0328319cc9a02663a759ec601d9402` to the existing two-host setup:

- `openai/gpt-oss-20b` replaces `openai/gpt-oss-120b`;
- the persistent generator remains on the SXM4 A100;
- the persistent Standard top-5/512 search service remains on p7 and calls the
  Qwen3-Embedding-8B encoder on the PCIe A100;
- queries run sequentially to preserve the measurement scaffold;
- only the frozen development split is exposed while the setup is still being
  validated.

The prompt, `local_knowledge_base_retrieval` tool definition, indented JSON
tool output, reasoning settings, 10,000-token per-turn output cap, and
100-iteration limit follow the upstream OSS runner.

The upstream client requests `truncation: auto`, but older vLLM GPT-OSS
Harmony/Responses implementations can still return HTTP 400 once the rendered
transcript reaches the native 131,072-token context. The runner records that
query as `status: incomplete` with
`termination_reason: context_length_exceeded`; the batch then continues to the
next query. This is a failed benchmark row, not a reason to abort the remaining
development split or to advertise a larger synthetic context window.

## Preflight

On p7, set paths without placing decrypted benchmark text in the shell history:

```bash
cd /mnt/nvme2/mlee/rag-system

# Locate the preparation output. Use the directory containing split.json.
find /mnt/nvme2/mlee/rag-system -type f \
  -path '*/datasets/browsecomp-plus/split.json' -print

export RAG_ARTIFACT_ROOT=/actual/artifact/root
export RAG_PREPARED_DIR="$RAG_ARTIFACT_ROOT/datasets/browsecomp-plus"
export RAG_RUN_DIR=/mnt/nvme2/mlee/rag-system/results/runs/gpt-oss-20b/high/standard/development
export RAG_GENERATOR_URL=http://solab-g3:8000/v1
export RAG_SEARCH_URL=http://127.0.0.1:8012

test -f "$RAG_PREPARED_DIR/split.json"
test -f "$RAG_PREPARED_DIR/queries.decrypted.jsonl"
python -c 'import json, os; p=os.environ["RAG_PREPARED_DIR"] + "/split.json"; print("development queries:", len(json.load(open(p))["development_query_ids"]))'
curl -fsS "$RAG_SEARCH_URL/health"
curl -fsS "$RAG_GENERATOR_URL/models"
```

The search response must report `top_k: 5` and `snippet_max_tokens: 512`; the
model response must list `openai/gpt-oss-20b`, and the split check must print
`development queries: 100`.

## Generator

If the generator is not already running at the full model context, start it in
a persistent session on g3:

```bash
tmux new -s oss20b-generator
cd /mnt/nvme2/mlee/rag-system
export RAG_MODEL_PATH=/path/to/openai--gpt-oss-20b
export CUDA_VISIBLE_DEVICES=<SXM4_A100_GPU_INDEX>
export VLLM_MAX_MODEL_LEN=131072
export VLLM_MAX_NUM_SEQS=1
export VLLM_ENABLE_PREFIX_CACHING=0
scripts/serve_oss_generator.sh
```

Detach with `Ctrl-b d`. The query encoder and p7 search service must also stay
in persistent sessions.

## Overnight development batch

Start a separate tmux session on p7:

```bash
tmux new -s oss20b-development
cd /mnt/nvme2/mlee/rag-system
export RAG_ARTIFACT_ROOT=/actual/artifact/root
export RAG_PREPARED_DIR="$RAG_ARTIFACT_ROOT/datasets/browsecomp-plus"
export RAG_RUN_DIR=/mnt/nvme2/mlee/rag-system/results/runs/gpt-oss-20b/high/standard/development
export RAG_GENERATOR_URL=http://solab-g3:8000/v1
export RAG_SEARCH_URL=http://127.0.0.1:8012
mkdir -p "$RAG_RUN_DIR"
set -o pipefail
PYTHONUNBUFFERED=1 python scripts/run_oss_standard_batch.py \
  --prepared-dir "$RAG_PREPARED_DIR" \
  --search-url "$RAG_SEARCH_URL" \
  --generator-url "$RAG_GENERATOR_URL" \
  --model openai/gpt-oss-20b \
  --output-dir "$RAG_RUN_DIR" \
  --reasoning-effort high \
  --max-output-tokens 10000 \
  --max-iterations 100 \
  --max-search-calls 100 \
  --generator-timeout-seconds 2400 \
  2>&1 | tee -a "$RAG_RUN_DIR/overnight.log"
```

Detach with `Ctrl-b d`. Monitor without disturbing the run:

```bash
tail -F "$RAG_RUN_DIR/overnight.log"
tail -n 20 "$RAG_RUN_DIR/batch.progress.jsonl"
```

The batch writes one private `run_<query_id>.json` and progress log per query,
plus `batch_summary.json`. It skips valid existing run artifacts. If the
process or host stops during a query, run the exact same command again; the
orphaned per-query progress log is replaced and that query is restarted.

The batch stops on the first new infrastructure/error row so a failed service
cannot poison the rest of the split. A native context exhaustion is instead
retained as an `incomplete` row and the next query runs. After correcting an
infrastructure failure, rerun that query with the single-query command and
`--force`, then launch the same batch command again. Completed refusals remain
completed rows and are reported through
`diagnostics.final_answer_validation`; they are not silently discarded.
