# rag-system

This repository is being rebuilt to study quality and serving cost across three
controlled BrowseComp-Plus workflows:

1. single-pass RAG;
2. iterative single-agent research; and
3. parallel multi-agent research.

The active four-week design is in
[`docs/plans/2026-07-21-rag-systems-four-week-plan.md`](docs/plans/2026-07-21-rag-systems-four-week-plan.md).
The latest measured state, exact retrieval results, host paths, and next-step
runbook are in
[`docs/progress/2026-07-22-qwen3-embedding-baseline-handoff.md`](docs/progress/2026-07-22-qwen3-embedding-baseline-handoff.md).
Previous scientific-retrieval benchmarks and the LaTeX chunking pilot are
archived under `old/` and are not imported by the active package.

## Deployment topology

```text
Windows work laptop (internet)
  |-- datasets / corpus / index --------------------> solab-p7
  |                                                    preparation / runs / metrics
  |
  `-- Qwen models ----------------------------------> solab-g3
                                                       PCIe A100: Qwen3-Embedding-8B
                                                       SXM4 A100: Qwen3.6-27B + vLLM

solab-p7 ---------------- search/generation HTTP ----------------> solab-g3
```

The current primary generator decision is `Qwen/Qwen3.6-27B`, served from a
local directory with vLLM and FlashInfer. The primary retriever is the local
`Qwen/Qwen3-Embedding-8B` checkpoint with the official precomputed
BrowseComp-Plus index. Hugging Face access is never required from either
server.

## Initial setup

Install active development and evaluation dependencies:

```bash
python -m pip install -e ".[dev,eval]"
```

On the internet-connected laptop, install the staging extra and create a
transfer bundle:

```bash
python -m pip install -e ".[staging]"
python scripts/stage_offline_assets.py --output-root offline-assets
python scripts/verify_offline_assets.py --root offline-assets
```

After copying datasets/tokenizer to `solab-p7`, prepare the local benchmark:

```bash
python scripts/prepare_browsecomp_plus.py \
  --queries-repo "$RAG_OFFLINE_ROOT/datasets/Tevatron--browsecomp-plus" \
  --corpus-repo "$RAG_OFFLINE_ROOT/datasets/Tevatron--browsecomp-plus-corpus" \
  --output-dir "$RAG_ARTIFACT_ROOT/datasets/browsecomp-plus"
```

After copying the model to `solab-g3`, launch the generator:

```bash
RAG_MODEL_PATH="$RAG_OFFLINE_ROOT/models/Qwen--Qwen3.6-27B" \
  ./scripts/serve_generator.sh
```

See [`docs/offline-deployment.md`](docs/offline-deployment.md) for exact staging,
copy, verification, host, and vLLM instructions.

## Dynamic Standard search smoke

The first live search uses a split service boundary: g3 encodes a query with
Qwen3-Embedding-8B on the PCIe A100, while p7 retains the official FAISS index
and canonical corpus. The query vector is the only retrieval artifact sent
back to p7.

On g3, from an environment containing Tevatron, Transformers, and PyTorch:

```bash
export CUDA_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/path/to/rag-system/src

python /path/to/rag-system/scripts/serve_query_encoder.py \
  --model-path /path/to/Qwen3-Embedding-8B-snapshot \
  --host 0.0.0.0 \
  --port 8011 \
  --max-length 512 \
  --attention-backend flash_attention_2
```

On p7, select an ID from the frozen development split and run:

```bash
python -m pip install -e ".[search]"

python scripts/smoke_standard_search.py \
  --prepared-dir "$RAG_ARTIFACT_ROOT/datasets/browsecomp-plus" \
  --query-id "$DEV_QUERY_ID" \
  --corpus-repo "$RAG_BROWSECOMP_CORPUS_DIR" \
  --index-path "$RAG_QWEN3_8B_INDEX_PATTERN" \
  --tokenizer-path "$RAG_QWEN3_06B_TOKENIZER_DIR" \
  --encoder-url "http://solab-g3:8011" \
  --datasets-cache "$HF_DATASETS_CACHE"
```

The smoke command refuses held-out IDs and prints only document IDs, scores,
and snippet token counts. It succeeds only when exactly five unique documents
are returned and every snippet is at most 512 tokens under the local upstream
tokenizer.

## Tests

```bash
pytest
```

Tests are offline by default. Checks requiring transferred benchmark files or a
running vLLM server are marked `integration`.
