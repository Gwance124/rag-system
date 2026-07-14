# rag-system

The repository now has two small projects:

- `latex-parser/` contains the original LaTeX parsing/chunking pilot, with its
  own `src/`, `scripts/`, `tests/`, and `pyproject.toml`.
- `src/retrieval/` and `scripts/run_public_bench.py` contain the retrieval and
  public-benchmark implementation.

Run all tests from the repository root with:

```bash
pytest
```

To run the LaTeX pilot scripts as a standalone project:

```bash
cd latex-parser
pip install -e ".[dev]"
```

The public runner defaults to the LitSearch sparse baseline. Install the eval
extra before using the Hugging Face datasets:

```bash
pip install -e ".[eval]"
python scripts/run_public_bench.py --benchmark litsearch
```

LitSearch and the BEIR scientific sanity checks are loaded from these cached
Hugging Face datasets:

- `princeton-nlp/LitSearch`
- `mteb/scifact`
- `mteb/trec-covid`

Warm the cache once on a machine with internet, copy that Hugging Face cache
to `solab-p7`, then run:

```bash
cd /mnt/nvme2/mlee/rag-system

python - <<'PY'
from datasets import load_dataset

cache = "/mnt/nvme2/labuser/.cache/huggingface/datasets"
for config in ("query", "corpus_clean"):
    load_dataset("princeton-nlp/LitSearch", config, split="full", cache_dir=cache)

for dataset in ("scifact", "trec-covid"):
    for config in ("corpus", "queries", "default"):
        load_dataset(f"mteb/{dataset}", config, split="test", cache_dir=cache)
PY

python scripts/run_public_bench.py \
  --benchmark litsearch --cache-dir /mnt/nvme2/labuser/.cache/huggingface

python scripts/run_public_bench.py \
  --benchmark beir --dataset scifact --cache-dir /mnt/nvme2/labuser/.cache/huggingface

python scripts/run_public_bench.py \
  --benchmark beir --dataset trec-covid --cache-dir /mnt/nvme2/labuser/.cache/huggingface
```

`--cache-dir` is the Hugging Face root, not the `hub/` directory itself. The
runner uses `/mnt/nvme2/labuser/.cache/huggingface/hub` for Hub files and
`/mnt/nvme2/labuser/.cache/huggingface/datasets` for dataset files.

For dense or hybrid runs, build each Qdrant collection first:

```bash
python scripts/build_dense_index.py \
  --benchmark litsearch --cache-dir /mnt/nvme2/labuser/.cache/huggingface \
  --qdrant-url http://localhost:6333 --collection litsearch

python scripts/run_public_bench.py \
  --benchmark litsearch --cache-dir /mnt/nvme2/labuser/.cache/huggingface \
  --mode dense --qdrant-url http://localhost:6333 --collection litsearch

python scripts/run_public_bench.py \
  --benchmark litsearch --cache-dir /mnt/nvme2/labuser/.cache/huggingface \
  --mode hybrid --qdrant-url http://localhost:6333 --collection litsearch
```

BRIGHT remains available through `--benchmark bright`; its configs are
`examples` and `documents` in `xlangai/BRIGHT`. All runtime HF loads are
cache-only and fail on a cache miss. Dense runs still use the configured vLLM
endpoint (`solab-g3:8000`) for embeddings.

### Windows cache staging

On the Windows laptop, use PowerShell and copy the complete cache root:

```powershell
$cache = "C:\hf-cache"
$env:HF_HOME = $cache
$env:HF_DATASETS_CACHE = "$cache\datasets"
$env:HF_HUB_CACHE = "$cache\hub"
py -m pip install -U datasets huggingface_hub

@'
import os
from datasets import load_dataset

cache = r"C:\hf-cache"
dataset_cache = os.path.join(cache, "datasets")

for config in ("query", "corpus_clean"):
    load_dataset("princeton-nlp/LitSearch", config, split="full", cache_dir=dataset_cache)

for dataset in ("scifact", "trec-covid"):
    for config in ("corpus", "queries", "default"):
        load_dataset(f"mteb/{dataset}", config, split="test", cache_dir=dataset_cache)
'@ | python -

ssh username@solab-p7 "mkdir -p /mnt/nvme2/labuser/.cache/huggingface"
scp -r "C:/hf-cache/datasets" "C:/hf-cache/hub" `
  username@solab-p7:/mnt/nvme2/labuser/.cache/huggingface/
```

The server should then have `/mnt/nvme2/labuser/.cache/huggingface/datasets`
and `/mnt/nvme2/labuser/.cache/huggingface/hub`. The embedding model cache is
separate: vLLM is already serving it from `solab-g3:8000`, so the benchmark
runner on `solab-p7` sends embedding requests there and does not need the model
copied locally.
