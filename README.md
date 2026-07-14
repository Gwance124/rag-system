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
python - <<'PY'
from datasets import load_dataset

cache = "/data/.cache/huggingface/datasets"
for config in ("query", "corpus_clean"):
    load_dataset("princeton-nlp/LitSearch", config, split="full", cache_dir=cache)

for dataset in ("scifact", "trec-covid"):
    for config in ("corpus", "queries", "default"):
        load_dataset(f"mteb/{dataset}", config, split="test", cache_dir=cache)
PY

python scripts/run_public_bench.py \
  --benchmark litsearch --cache-dir /data/.cache/huggingface

python scripts/run_public_bench.py \
  --benchmark beir --dataset scifact --cache-dir /data/.cache/huggingface

python scripts/run_public_bench.py \
  --benchmark beir --dataset trec-covid --cache-dir /data/.cache/huggingface
```

For dense or hybrid runs, build each Qdrant collection first:

```bash
python scripts/build_dense_index.py \
  --benchmark litsearch --cache-dir /data/.cache/huggingface \
  --qdrant-url http://localhost:6333 --collection litsearch

python scripts/run_public_bench.py \
  --benchmark litsearch --cache-dir /data/.cache/huggingface \
  --mode dense --qdrant-url http://localhost:6333 --collection litsearch

python scripts/run_public_bench.py \
  --benchmark litsearch --cache-dir /data/.cache/huggingface \
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

scp -r "C:/hf-cache" username@solab-p7:/data/
```

The server should then have `/data/hf-cache/datasets` and
`/data/hf-cache/hub`. The embedding model cache is separate: copy it to
`solab-g3` only if the vLLM service still needs the model staged locally.
