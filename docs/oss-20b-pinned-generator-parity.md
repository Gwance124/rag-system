# GPT-OSS-20B parity rerun on the pinned upstream vLLM

Date: 2026-07-23

## Why this run exists

The bare-metal g3 vLLM build fails on GPT-OSS Harmony tool calls: HTTP 500s,
`invalid_tool_call` rows, unrecognized `mcp_call` recipients, and
`unexpected tokens remaining in message header
to=functions.local_knowledge_base_retrieval` (the vllm-project/vllm#23567
class). One crashed trajectory still reached 0.83 evidence / 0.75 gold
cumulative recall in 8 searches, so the model and the two-host Standard
scaffold are healthy; the Harmony parser in the installed server is not. The
overnight dev batch that "completed" with 0.16 evidence recall and no valid
answer format is a symptom of the same silently degraded tool loop.

The upstream leaderboard runs did not solve this in code. They pinned serving
images: `BrowseComp-Plus/docs/oss.md` uses `vllm/vllm-openai:v0.10.1` (BM25
section) and `vllm/vllm-openai:gptoss` (Qwen3-Embedding section), dated
2025-08-09. Parity therefore means serving from one of those pinned builds.

**A parity run is not a measurement run.** This server keeps upstream
defaults (including prefix caching on). Never use rows from it for latency or
KV-cache reporting; the instrumented measurement configuration comes after
parity passes.

Reference numbers (upstream evaluation summary for `openai/gpt-oss-20b`):
Accuracy 32.17%, Recall 43.0%, mean 12.6 searches/query over all 830 queries.
On the 100-query development split, sampling noise widens the recall gate to
a band: **0.38–0.48 mean trajectory evidence recall**.

## Step 1 — Serve gpt-oss-20b from the pinned build (g3, SXM4 A100)

Stop the current bare-metal generator tmux session first. Leave the PCIe
A100 query-encoder service and the p7 search service untouched.

Preferred (docker, exact upstream image):

```bash
tmux new -s oss20b-pinned
docker --version   # confirm docker + NVIDIA runtime exist before pulling
export RAG_MODEL_PATH=/path/to/openai--gpt-oss-20b
export SXM4_GPU_INDEX=<index from nvidia-smi>
docker run --rm --name oss20b-pinned \
  --gpus "\"device=${SXM4_GPU_INDEX}\"" \
  -p 8000:8000 --ipc=host \
  -v "$RAG_MODEL_PATH":/models/gpt-oss-20b:ro \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  vllm/vllm-openai:v0.10.1 \
  --model /models/gpt-oss-20b \
  --served-model-name openai/gpt-oss-20b
```

`--served-model-name` is the only intentional deviation from the upstream
command; it keeps the existing p7 preflight (`/v1/models` must list
`openai/gpt-oss-20b`) working while serving from the local offline snapshot.
Add no other flags: upstream ran with image defaults.

No docker on g3 → dedicated venv with the same pinned release:

```bash
export UV_CACHE_DIR=/mnt/nvme3n1/labuser/.cache/uv
export TMPDIR=/mnt/nvme3n1/labuser/.cache/tmp
uv venv --python 3.12 /mnt/nvme3n1/mlee/venvs/vllm0101
source /mnt/nvme3n1/mlee/venvs/vllm0101/bin/activate
uv pip install vllm==0.10.1
export CUDA_VISIBLE_DEVICES=$SXM4_GPU_INDEX
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 vllm serve "$RAG_MODEL_PATH" \
  --served-model-name openai/gpt-oss-20b
```

Do not reuse `scripts/serve_oss_generator.sh` for this parity server: it adds
Chat-Completions tool-parser and telemetry flags that upstream never set.

Record what is actually serving:

```bash
curl -fsS http://solab-g3:8000/version
curl -fsS http://solab-g3:8000/v1/models
```

Save both outputs next to the run directory.

## Step 2 — One-query smoke through the existing scaffold (p7)

```bash
cd /mnt/nvme2/mlee/rag-system
export RAG_PREPARED_DIR=<dir containing split.json>
export RAG_GENERATOR_URL=http://solab-g3:8000/v1
export RAG_SEARCH_URL=http://127.0.0.1:8012
python scripts/run_oss_standard_agent.py \
  --prepared-dir "$RAG_PREPARED_DIR" \
  --query-id <one frozen development ID> \
  --search-url "$RAG_SEARCH_URL" \
  --generator-url "$RAG_GENERATOR_URL" \
  --model openai/gpt-oss-20b \
  --output-dir /mnt/nvme2/mlee/rag-system/results/runs/gpt-oss-20b/high/standard/pinned-v0101-smoke
```

Smoke gate, all required:

- zero `generation_retrying` / `generation_failed` events with Harmony header
  or HTTP 500 messages;
- zero `invalid_tool_call` / `mcp_call_rejected` events (occasional
  normalized `mcp_search_aliases` are acceptable);
- several real search calls (expect roughly 5–15, not 0–2);
- `status=completed` with `final_format_valid=true`.

Repeat for 2–3 more development IDs before committing to the batch.

## Step 3 — Development-split recall batch (p7)

Same command as `docs/oss-20b-standard-overnight.md`, with a fresh output
directory so the broken-server rows are never mixed in:

```bash
export RAG_RUN_DIR=/mnt/nvme2/mlee/rag-system/results/runs/gpt-oss-20b/high/standard/development-pinned-v0101
```

After the batch finishes:

```bash
python scripts/summarize_agent_runs.py \
  --run-dir "$RAG_RUN_DIR" \
  --prepared-dir "$RAG_PREPARED_DIR" \
  --output "$RAG_RUN_DIR/recall_summary.json"
```

Parity gate:

- `evidence_recall_mean` in **0.38–0.48**;
- `search_calls_mean` roughly 8–18 (leaderboard mean 12.6);
- `status_counts` dominated by `completed`; no `error` rows from Harmony
  parsing;
- most rows `final_answer_format_valid=true`.

Accuracy via the Qwen3-32B judge is deferred; recall parity unblocks the
systems work.

## Single-pass baseline (no tool calling; runs even if parity stalls)

`scripts/run_single_pass.py` packs the top-k documents of the frozen
Qwen3-Embedding-8B `top1000.trec` into one prompt using the same 512-token
snippet contract and issues one no-tool Responses request per query. It needs
the corpus snapshot and 0.6B tokenizer (the same inputs as the p7 search
service) but no search service and no tool calling, so Harmony tool bugs
cannot affect it.

Smoke one development query first, then the split at k=5:

```bash
cd /mnt/nvme2/mlee/rag-system
export RAG_RANKING_TREC=/mnt/nvme2/mlee/rag-system/results/retrieval/qwen3-embedding-8b/top1000.trec
export RAG_CORPUS_REPO=<corpus snapshot dir given to serve_standard_search.py>
export RAG_SNIPPET_TOKENIZER=<0.6B tokenizer dir given to serve_standard_search.py>

python scripts/run_single_pass.py \
  --prepared-dir "$RAG_PREPARED_DIR" \
  --ranking-trec "$RAG_RANKING_TREC" \
  --corpus-repo "$RAG_CORPUS_REPO" \
  --tokenizer-path "$RAG_SNIPPET_TOKENIZER" \
  --generator-url "$RAG_GENERATOR_URL" \
  --model openai/gpt-oss-20b \
  --top-k 5 \
  --query-id <one frozen development ID> \
  --output-dir /mnt/nvme2/mlee/rag-system/results/runs/gpt-oss-20b/high/single-pass/k5-smoke

python scripts/run_single_pass.py \
  --prepared-dir "$RAG_PREPARED_DIR" \
  --ranking-trec "$RAG_RANKING_TREC" \
  --corpus-repo "$RAG_CORPUS_REPO" \
  --tokenizer-path "$RAG_SNIPPET_TOKENIZER" \
  --generator-url "$RAG_GENERATOR_URL" \
  --model openai/gpt-oss-20b \
  --top-k 5 \
  --output-dir /mnt/nvme2/mlee/rag-system/results/runs/gpt-oss-20b/high/single-pass/development-k5
```

The batch is resumable, writes one private `run_<query_id>.json` per query,
and finishes by writing `recall_summary.json` (context evidence/gold recall
of the shown documents plus answer-format validity). For the context-budget
sweep, rerun with `--top-k 10` and `--top-k 20` into `development-k10` and
`development-k20`; every k reuses the same frozen ranking, so smaller-k
contexts are prefixes of larger-k contexts.

Expected anchor: the retrieval reproduction measured Evidence Recall@5 at
0.1494 over all 830 queries, so the k=5 dev context recall should land near
0.15; agentic search's ~0.43 trajectory recall against this fixed-context
floor is exactly the breadth-vs-cost comparison the study needs.

These runs are quality/recall rows. Latency and KV-cache rows require the
instrumented measurement server configuration (prefix caching off,
exclusive GPU), not this parity server.

## Escalation ladder (timebox: one day)

1. `v0.10.1` still throws Harmony header errors → same command with image
   `vllm/vllm-openai:gptoss`.
2. Server healthy but recall clearly below the band → isolate client
   divergence: run upstream `search_agent/oss_client.py` unmodified on g3
   (FAISS shards + Qwen3-Embedding-8B on the PCIe A100; requires copying the
   official index shards from the p7 HF cache to g3).
3. Timebox exhausted → proceed to the single-pass baseline (next section);
   it does not depend on tool calling at all.
