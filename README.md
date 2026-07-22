# rag-system

This repository is being rebuilt to study quality and serving cost across three
controlled BrowseComp-Plus workflows:

1. single-pass RAG;
2. iterative single-agent research; and
3. parallel multi-agent research.

The active four-week design is in
[`docs/plans/2026-07-21-rag-systems-four-week-plan.md`](docs/plans/2026-07-21-rag-systems-four-week-plan.md).
Previous scientific-retrieval benchmarks and the LaTeX chunking pilot are
archived under `old/` and are not imported by the active package.

## Deployment topology

```text
work laptop (internet)
  |-- BrowseComp-Plus datasets + Qwen tokenizer --> scp --> solab-p7
  |                                                     data prep / retrieval / benchmark runner
  |
  `-- Qwen3.5-27B model --------------------------> scp --> solab-g3
                                                        vLLM + FlashInfer on 1x A100 80 GB

solab-p7 ---------------- OpenAI-compatible HTTP ----------------> solab-g3
```

The initial generator is pinned to `Qwen/Qwen3.5-27B`. The model is served from
a local directory with vLLM's language-only mode and the FlashInfer attention
backend. Hugging Face access is never required from either server.

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
RAG_MODEL_PATH="$RAG_OFFLINE_ROOT/models/Qwen--Qwen3.5-27B" \
  ./scripts/serve_generator.sh
```

See [`docs/offline-deployment.md`](docs/offline-deployment.md) for exact staging,
copy, verification, host, and vLLM instructions.

## Tests

```bash
pytest
```

Tests are offline by default. Checks requiring transferred benchmark files or a
running vLLM server are marked `integration`.
