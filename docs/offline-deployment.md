# Offline asset staging and two-host deployment

This project never asks `solab-p7` or `solab-g3` to resolve a Hugging Face
repository ID. The internet-connected work laptop downloads immutable snapshots,
records a file manifest, and transfers ordinary directories. Runtime code loads
local Parquet and model/tokenizer files only.

## Pinned assets

| Component | Repository | Revision | Destination |
| --- | --- | --- | --- |
| Queries, answers, labels | `Tevatron/browsecomp-plus` | `144cff8e35b5eaef7e526346aa60774a9deb941f` | `solab-p7` |
| Canonical corpus | `Tevatron/browsecomp-plus-corpus` | `b27b02bc3e45511b8b82a13e6f90ce761df726f6` | `solab-p7` |
| Qwen tokenizer/config | `Qwen/Qwen3.5-27B` | `fc05daec18b0a78c049392ed2e771dde82bdf654` | `solab-p7` |
| Qwen3.5-27B model | `Qwen/Qwen3.5-27B` | `fc05daec18b0a78c049392ed2e771dde82bdf654` | `solab-g3` |

The model snapshot is roughly 56 GB. The query dataset is large because it
contains encrypted copies of evidence, gold, and negative document text. The
preparer decrypts only the question, reference answer, and labeled IDs; all
retrieval/context text comes from the separate canonical corpus.

## 1. Stage on the work laptop

From this repository:

```bash
python -m pip install -e ".[staging]"
python scripts/stage_offline_assets.py \
  --output-root "$HOME/rag-offline-assets"
python scripts/verify_offline_assets.py \
  --root "$HOME/rag-offline-assets"
```

The staging command is resumable and downloads exact revisions. By default it
computes SHA-256 for every transferred file, including the model. Use
`--skip-checksums` only if the hashing time is unacceptable; file sizes will
still be recorded, but the weaker manifest must be noted in the run manifest.

Components can be staged separately:

```bash
python scripts/stage_offline_assets.py \
  --output-root "$HOME/rag-offline-assets" \
  --component datasets --component tokenizer

python scripts/stage_offline_assets.py \
  --output-root "$HOME/rag-offline-assets" \
  --component model
```

Use `HF_TOKEN` or `hf auth login` if authentication becomes necessary. Do not
put a token directly in a command saved to shell history.

The resulting layout is:

```text
rag-offline-assets/
  OFFLINE_ASSET_MANIFEST.json
  datasets/
    Tevatron--browsecomp-plus/
    Tevatron--browsecomp-plus-corpus/
  tokenizers/
    Qwen--Qwen3.5-27B/
  models/
    Qwen--Qwen3.5-27B/
```

## 2. Copy to the two servers

Choose storage roots appropriate to each host. These examples deliberately do
not assume a personal username or mount path.

Using `scp`:

```bash
scp -r \
  "$HOME/rag-offline-assets/datasets" \
  "$HOME/rag-offline-assets/tokenizers" \
  "$HOME/rag-offline-assets/OFFLINE_ASSET_MANIFEST.json" \
  "<user>@solab-p7:<p7-offline-root>/"

scp -r \
  "$HOME/rag-offline-assets/models" \
  "$HOME/rag-offline-assets/OFFLINE_ASSET_MANIFEST.json" \
  "<user>@solab-g3:<g3-offline-root>/"
```

For the 56 GB model, `rsync --partial --progress -a` is safer than `scp` because
it can resume an interrupted copy. It does not change the bundle layout or
verification process.

Copy this Git repository to both hosts separately. Do not copy decrypted
queries or experiment results to `solab-g3`; it needs only the model and server
code.

## 3. Verify and prepare on `solab-p7`

```bash
export RAG_OFFLINE_ROOT="<p7-offline-root>"
export RAG_ARTIFACT_ROOT="<p7-artifact-root>"
export RAG_BROWSECOMP_QUERIES_DIR="$RAG_OFFLINE_ROOT/datasets/Tevatron--browsecomp-plus"
export RAG_BROWSECOMP_CORPUS_DIR="$RAG_OFFLINE_ROOT/datasets/Tevatron--browsecomp-plus-corpus"
export RAG_TOKENIZER_PATH="$RAG_OFFLINE_ROOT/tokenizers/Qwen--Qwen3.5-27B"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python scripts/verify_offline_assets.py \
  --root "$RAG_OFFLINE_ROOT" \
  --component browsecomp_queries \
  --component browsecomp_corpus \
  --component qwen3_5_27b_tokenizer

python scripts/prepare_browsecomp_plus.py \
  --queries-repo "$RAG_BROWSECOMP_QUERIES_DIR" \
  --corpus-repo "$RAG_BROWSECOMP_CORPUS_DIR" \
  --output-dir "$RAG_ARTIFACT_ROOT/datasets/browsecomp-plus" \
  --cache-dir "$RAG_ARTIFACT_ROOT/cache/datasets"
```

The preparation command uses the local Parquet builder with offline environment
flags. It validates the expected 830 queries and 100,195 canonical documents,
checks every evidence/gold ID, writes the deterministic 100/730 split, and
creates private (`0600`) decrypted query artifacts. It prints counts and hashes,
never questions or answers.

The output directory must stay outside Git and should be readable only by the
benchmark user.

## 4. Verify and serve on `solab-g3`

Qwen3.5 support requires a recent vLLM nightly/main build. FlashInfer is a
separate installation; vLLM wheels do not bundle it. Match the vLLM, PyTorch,
FlashInfer, CUDA, and NVIDIA driver versions on `solab-g3`, and retain the
installation commands plus `pip freeze` in the eventual run manifest.

FlashInfer may JIT or obtain kernels on first use. For a completely offline
runtime, install the matching `flashinfer-cubin` and, when available for the
host CUDA version, `flashinfer-jit-cache` packages before disconnecting. Verify
the installation with:

```bash
flashinfer show-config
python -c 'import torch, vllm, flashinfer; print(torch.__version__, torch.version.cuda, vllm.__version__, flashinfer.__version__)'
```

Then verify the transferred model and launch:

```bash
export RAG_OFFLINE_ROOT="<g3-offline-root>"
export RAG_MODEL_PATH="$RAG_OFFLINE_ROOT/models/Qwen--Qwen3.5-27B"

python scripts/verify_offline_assets.py \
  --root "$RAG_OFFLINE_ROOT" \
  --component qwen3_5_27b_model

./scripts/serve_generator.sh
```

The initial single-A100 configuration is deliberately conservative:

- local BF16 Qwen3.5-27B weights;
- tensor parallel size 1;
- text-only `--language-model-only` mode;
- FlashInfer selected explicitly rather than silently auto-selected;
- 65,536 total model tokens, leaving room around the 32K retrieved-context arm;
- maximum four live sequences initially;
- prefix caching explicitly disabled;
- per-request metrics and server load tracking enabled.

Qwen3.5 uses a hybrid Gated DeltaNet/full-attention cache, so the ordinary
all-transformer KV-bytes formula is not the complete capacity model. Treat
successful 32K startup/request tests and vLLM's reported cache configuration as
hard gates before running the benchmark. If 65,536 does not fit, diagnose model
weights, vision loading, vLLM version, backend, and cache allocation; do not
silently reduce the 32K experiment arm.

The server script exports `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, checks
the local model files, CUDA, vLLM, and FlashInfer, then passes the local model
directory to `vllm serve`.

## 5. Connect `solab-p7` to `solab-g3`

From `solab-p7`:

```bash
export RAG_GENERATOR_BASE_URL="http://solab-g3:8000/v1"

curl -fsS "http://solab-g3:8000/health"
curl -fsS "http://solab-g3:8000/v1/models"
curl -fsS "http://solab-g3:8000/metrics" | head
```

If `solab-g3` is not resolvable from `solab-p7`, use its stable lab IP in
`RAG_GENERATOR_BASE_URL`. Restrict port 8000 to the lab network or an SSH tunnel;
the initial server has no API key.

Save these before the first measurement:

```bash
curl -fsS "http://solab-g3:8000/metrics" > "$RAG_ARTIFACT_ROOT/vllm-metric-catalog.prom"
python -m pip freeze > "$RAG_ARTIFACT_ROOT/p7-pip-freeze.txt"
```

Capture the corresponding package list, launch log, `nvidia-smi -q`, and
FlashInfer configuration on `solab-g3`. These are inputs to the run manifest,
not ad hoc notes.

## Official references

- [Qwen3.5-27B model card](https://huggingface.co/Qwen/Qwen3.5-27B)
- [BrowseComp-Plus query dataset](https://huggingface.co/datasets/Tevatron/browsecomp-plus)
- [BrowseComp-Plus corpus dataset](https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus)
- [vLLM Qwen3.5 recipe](https://github.com/vllm-project/recipes/blob/main/Qwen/Qwen3.5.md)
- [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models/)
- [FlashInfer installation](https://docs.flashinfer.ai/installation.html)
