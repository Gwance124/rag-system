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

`--context-budget-tokens` (default 128,000) guards a related but distinct
pathology observed on query 1035 (2026-07-23): once a query's running prompt
gets close to 131,072, `_drop_oldest_turn` can free just enough room for one
more turn without ever tripping the hard vLLM context-length exception again,
leaving only single-digit-to-low-hundreds of output tokens per turn — not
enough to complete a Harmony message. The model then emits `reasoning`-only
output every turn, tool_call_count 0, response_status `incomplete`, and the
`reasoning_only_retry` recovery path (which exists for a genuinely transient
truncated turn) retries forever with no way out short of `max_iterations`
(50+ wasted turns observed, no progress, no scorable answer). Once the
running prompt crosses this threshold, the workflow stops declaring the
search tool and appends an instruction to answer immediately with whatever
evidence has been gathered; if the model still tries to search anyway (it
can, regardless of what tools are declared this turn — see
`docs/oss-20b-pinned-generator-parity.md`), the call is rejected rather than
executed. See `context_budget_final_answer_forced` and
`context_budget_search_rejected` in the progress log, and
`diagnostics.context_budget_triggered` in the run record.

## Preflight

On p7, set paths without placing decrypted benchmark text in the shell history:

```bash
cd /mnt/nvme2/mlee/rag-system

# Locate the preparation output. Use the directory containing split.json.
find /mnt/nvme2/mlee/rag-system -type f \
  -path '*/datasets/browsecomp-plus/split.json' -print

export RAG_ARTIFACT_ROOT=/mnt/nvme2/mlee/rag-system/results
export RAG_PREPARED_DIR="$RAG_ARTIFACT_ROOT/datasets/browsecomp-plus"
export RAG_RUN_DIR=/mnt/nvme2/mlee/rag-system/results/runs/gpt-oss-20b/high/standard/development
export RAG_GENERATOR_URL=http://192.168.3.4:8000/v1  # solab-g3 lab IP; hostname is unreliable
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
export RAG_MODEL_PATH=/mnt/nvme3n1/labuser/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee
export CUDA_VISIBLE_DEVICES=0
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
export RAG_ARTIFACT_ROOT=/mnt/nvme2/mlee/rag-system/results
export RAG_PREPARED_DIR="$RAG_ARTIFACT_ROOT/datasets/browsecomp-plus"
export RAG_RUN_DIR=/mnt/nvme2/mlee/rag-system/results/runs/gpt-oss-20b/high/standard/development
export RAG_GENERATOR_URL=http://192.168.3.4:8000/v1  # solab-g3 lab IP; hostname is unreliable
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
  --context-budget-tokens 128000 \
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
