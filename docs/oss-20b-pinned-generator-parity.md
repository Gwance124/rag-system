# GPT-OSS-20B parity rerun on the pinned upstream vLLM

Date: 2026-07-23

> **Status update (2026-07-23, later):** the g3 server turned out to be
> vLLM **0.19.1**, not a broken older build, and after the client fixes
> (tolerant `user_query` parsing, error feedback instead of hard abort,
> bounded retries) queries 703 and 215 completed with real searches on it.
> Therefore run Step 3 (dev-100 batch) directly on the current 0.19.1
> server and record `curl :8000/version` with the run. Steps 1–2 below
> (pinned image) are now the **escalation fallback**, used only if recall
> misses the 0.44–0.54 band or Harmony failures survive 3 retries.
>
> **Correction (2026-07-23, evening):** the `unexpected tokens remaining in
> message header` 500s are **not** generic temperature-related noise —
> they are [openai/harmony#80](https://github.com/openai/harmony/issues/80):
> gpt-oss frequently omits the `<|message|>` token before a refusal's
> analysis-channel text, and the strict Harmony parser hard-errors on it.
> This is deterministic per-query (a query whose honest answer trends
> toward refusal can hit it on every retry, not just occasionally), so
> "bounded retries absorb them" cannot be assumed. The upstream library fix
> shipped in `openai-harmony>=0.0.6` but is **opt-in** (`strict=False`
> constructor argument) and vLLM 0.19.1 does not pass it — confirmed our
> server's installed `openai-harmony==0.0.8` still reproduces the exact
> failure with vLLM's default (non-strict-disabled) parser construction.
> Apply `patches/vllm-0.19.1-harmony-strict-false.patch` to the installed
> vLLM before running Step 3; see the preflight below.
>
> **Second correction (2026-07-23, same evening):** a separate, unrelated
> vLLM bug — [vllm-project/vllm#32587](https://github.com/vllm-project/vllm/issues/32587),
> open and unowned upstream — intermittently leaks the raw
> `<|channel|>commentary` special token onto the end of the tool name gpt-oss
> reports (e.g. `local_knowledge_base_retrievalcommentary` instead of
> `local_knowledge_base_retrieval`), on an otherwise well-formed call.
> Observed live 4+ times in one query (703). `skip_special_tokens` does not
> help (the leak happens during generation, not post-processing), so this is
> handled client-side: `_strip_leaked_channel_suffix` in
> `oss_standard_agent.py` strips a trailing `commentary`/`analysis`/`final`
> before alias-checking, recovering the call on the same turn instead of
> rejecting it and costing a wasted generation round-trip. No server-side
> action needed for this one — it's already in the workflow code.

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

Reference numbers (BrowseComp-Plus paper, Table 4, `gpt-oss-20B-high` +
`Qwen3-Embed-8B` row — matches this runbook's `--reasoning-effort high`):
Accuracy 34.58%, Recall 49.29%, mean 23.87 searches/query, Calibration Error
27.81%, over all 830 queries. (An earlier draft of this doc cited
32.17%/43.0%/12.6 searches; that number does not correspond to any row in
Table 4 and was likely transcribed from a different table/version — do not
use it.)
On the 100-query development split, sampling noise widens the recall gate to
a band: **0.44–0.54 mean trajectory evidence recall** (same ±5-point half
width as before, recentered on 0.4929).

## Step 1 — Serve gpt-oss-20b from the pinned build (g3, SXM4 A100)

Stop the current bare-metal generator tmux session first. Leave the PCIe
A100 query-encoder service and the p7 search service untouched.

Preferred (docker, exact upstream image):

```bash
tmux new -s oss20b-pinned
docker --version   # confirm docker + NVIDIA runtime exist before pulling
export RAG_MODEL_PATH=/mnt/nvme3n1/labuser/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee
export SXM4_GPU_INDEX=0
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
export RAG_PREPARED_DIR=/mnt/nvme2/mlee/rag-system/results/datasets/browsecomp-plus
export RAG_GENERATOR_URL=http://192.168.3.4:8000/v1  # solab-g3 lab IP; hostname is unreliable
export RAG_SEARCH_URL=http://127.0.0.1:8012
python scripts/run_oss_standard_agent.py \
  --prepared-dir "$RAG_PREPARED_DIR" \
  --query-id 703 \
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

## Step 2.5 — Patch the Harmony refusal-parsing bug (g3, before starting the generator)

Apply `patches/vllm-0.19.1-harmony-strict-false.patch` to the installed
vLLM before serving. It changes one function
(`get_streamable_parser_for_assistant` in
`vllm/entrypoints/openai/parser/harmony_utils.py`) to construct the Harmony
`StreamableParser` with `strict=False`, working around
[openai/harmony#80](https://github.com/openai/harmony/issues/80). This is a
pure-Python file inside the installed package — no vLLM rebuild, no
recompiled kernels, just an edit and a server restart.

```bash
HARMONY_UTILS_PATH=$(python -c \
  "import vllm.entrypoints.openai.parser.harmony_utils as m; print(m.__file__)")
echo "$HARMONY_UTILS_PATH"
patch -p1 --directory "$(dirname "$(dirname "$(dirname "$(dirname "$(dirname "$HARMONY_UTILS_PATH")")")")")" \
  < /mnt/nvme2/mlee/rag-system/patches/vllm-0.19.1-harmony-strict-false.patch
grep -n "strict=False" "$HARMONY_UTILS_PATH"
```

If `patch` reports "previously applied" or the `grep` above already finds
`strict=False`, the patch is already in place — do not reapply. Reapply
after any vLLM reinstall/upgrade on g3, since it lives inside
`site-packages` and is not tracked by pip. Record whether the patch was
present in the run's provenance notes alongside `curl :8000/version`.

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

- `evidence_recall_mean` in **0.44–0.54**;
- `search_calls_mean` roughly 19–29 (leaderboard mean 23.87);
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
export RAG_CORPUS_REPO=/mnt/nvme2/labuser/.cache/huggingface/datasets/datasets--Tevatron--browsecomp-plus-corpus/snapshots/b27b02bc3e45511b8b82a13e6f90ce761df726f6
read -r TOKENIZER_REV < "/mnt/nvme2/labuser/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/refs/main"
export RAG_SNIPPET_TOKENIZER="/mnt/nvme2/labuser/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/$TOKENIZER_REV"

python scripts/run_single_pass.py \
  --prepared-dir "$RAG_PREPARED_DIR" \
  --ranking-trec "$RAG_RANKING_TREC" \
  --corpus-repo "$RAG_CORPUS_REPO" \
  --tokenizer-path "$RAG_SNIPPET_TOKENIZER" \
  --generator-url "$RAG_GENERATOR_URL" \
  --model openai/gpt-oss-20b \
  --top-k 5 \
  --query-id 703 \
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
