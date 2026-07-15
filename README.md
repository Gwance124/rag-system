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

LitSearch and the default scientific checks are loaded from these cached
Hugging Face datasets:

- `princeton-nlp/LitSearch`
- `mteb/scidocs` (the default for `--benchmark mteb`)
- `mteb/scifact`
- `mteb/nfcorpus`
- `mteb/trec-covid`

The MTEB retrieval adapter is not limited to that shortlist. `--dataset NAME` resolves
to `mteb/NAME`, and `--dataset-id ORG/NAME` accepts any cached MTEB-format
dataset directly. This covers the BEIR retrieval datasets hosted under MTEB
without adding a new loader for every dataset. `--benchmark beir` remains as
a backward-compatible alias.

Warm the cache once on a machine with internet, copy that Hugging Face cache
to `solab-p7`, then run:

```bash
cd /mnt/nvme2/mlee/rag-system

python - <<'PY'
from datasets import load_dataset

cache = "/mnt/nvme2/labuser/.cache/huggingface/datasets"
for config in ("query", "corpus_clean"):
    load_dataset("princeton-nlp/LitSearch", config, split="full", cache_dir=cache)

for dataset in ("scidocs", "scifact", "nfcorpus", "trec-covid"):
    load_dataset(f"mteb/{dataset}", "corpus", split="corpus", cache_dir=cache)
    load_dataset(f"mteb/{dataset}", "queries", split="queries", cache_dir=cache)
    load_dataset(f"mteb/{dataset}", "default", split="test", cache_dir=cache)
PY

python scripts/run_public_bench.py \
  --benchmark litsearch --cache-dir /mnt/nvme2/labuser/.cache/huggingface

python scripts/run_public_bench.py \
  --benchmark mteb --cache-dir /mnt/nvme2/labuser/.cache/huggingface

python scripts/run_public_bench.py \
  --benchmark mteb --dataset scifact --cache-dir /mnt/nvme2/labuser/.cache/huggingface

python scripts/run_public_bench.py \
  --benchmark mteb --dataset trec-covid --cache-dir /mnt/nvme2/labuser/.cache/huggingface

# Any other cached MTEB-format BEIR dataset works the same way.
python scripts/run_public_bench.py \
  --benchmark mteb --dataset nfcorpus --cache-dir /mnt/nvme2/labuser/.cache/huggingface
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
endpoint (`192.168.3.4:8000`) for embeddings.

### Dataset and index storage

All Hugging Face datasets share one cache root, but Hugging Face stores each
dataset and configuration in its own directory under `datasets/`. They do not
need hand-created directories and are not combined into one benchmark corpus.

Qdrant collections should be named by benchmark corpus and document embedding
model, for example `mteb-scidocs-llama-nv-reasoning-3b`. Each BEIR dataset gets
its own collection because its corpus and document IDs are independent. Each
embedding model also gets its own collection because vector dimensions and
spaces can differ. Dense and hybrid runs reuse the same dense collection;
BM25 is rebuilt in memory. Query rewrites, query prefixes, and rerankers reuse
the collection because they do not change stored document vectors. Changing
the passage model, passage prefix, or document text requires a new collection.

### LitSearch paper-aligned results

LitSearch results now include `recall@5` and `recall@20` plus a
`litsearch_paper_comparison` section. That section splits the 597 queries by
`query_set` (`inline-citation` or `author-written`) and `specificity` (`broad`
or `specific`), then reports the same cutoffs as the paper: broad `R@20`, and
specific `R@5` and `R@20`. It also includes the paper's BM25 reference values
and our delta from them. It also includes the paper's Table 8 `nDCG@10`
references for GTR-T5-large, Instructor-XL, E5-large-v2, and GritLM-7B.
Values are fractions, so `0.50` means 50%.

Save each run so the model and mode recorded in its `config` can be compared:

```bash
mkdir -p results

python scripts/run_public_bench.py \
  --benchmark litsearch \
  --cache-dir /mnt/nvme2/labuser/.cache/huggingface \
  --mode sparse > results/litsearch-bm25.json

python scripts/build_dense_index.py \
  --benchmark litsearch \
  --cache-dir /mnt/nvme2/labuser/.cache/huggingface \
  --embedding-model nvidia/llama-nv-embed-reasoning-3b \
  --embedding-api-model /model \
  --qdrant-url http://localhost:6333 \
  --collection litsearch-reason

python scripts/run_public_bench.py \
  --benchmark litsearch \
  --cache-dir /mnt/nvme2/labuser/.cache/huggingface \
  --mode dense \
  --embedding-model nvidia/llama-nv-embed-reasoning-3b \
  --embedding-api-model /model \
  --qdrant-url http://localhost:6333 \
  --collection litsearch-reason > results/litsearch-reason.json
```

For a general-purpose NVIDIA comparison, serve `nvidia/NV-Embed-v2` on a
separate vLLM endpoint (or restart the existing endpoint), then repeat the
index and run commands with `--embedding-model nvidia/NV-Embed-v2` and a
different collection such as `litsearch-nv-embed-v2`. Use the model's query
instruction and no passage prefix:

```bash
--embedding-model nvidia/NV-Embed-v2 \
--query-prefix $'Instruct: Given a question, retrieve passages that answer the question\nQuery: ' \
--passage-prefix ''
```

The model flag tells the OpenAI-compatible endpoint which model to use; it does
not load a model itself. Use separate collections because embedding dimensions
and vector spaces may differ. The paper's BM25 implementation uses
`rank-bm25`; this project keeps a small dependency-free implementation, so the
paper comparison is a reference check rather than an exact implementation
match.

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

for dataset in ("scidocs", "scifact", "nfcorpus", "trec-covid"):
    load_dataset(f"mteb/{dataset}", "corpus", split="corpus", cache_dir=dataset_cache)
    load_dataset(f"mteb/{dataset}", "queries", split="queries", cache_dir=dataset_cache)
    load_dataset(f"mteb/{dataset}", "default", split="test", cache_dir=dataset_cache)
'@ | python -

ssh username@solab-p7 "mkdir -p /mnt/nvme2/labuser/.cache/huggingface"
scp -r "C:/hf-cache/datasets" "C:/hf-cache/hub" `
  username@solab-p7:/mnt/nvme2/labuser/.cache/huggingface/
```

The server should then have `/mnt/nvme2/labuser/.cache/huggingface/datasets`
and `/mnt/nvme2/labuser/.cache/huggingface/hub`. The embedding model cache is
separate: vLLM is already serving it from `192.168.3.4:8000`, so the benchmark
runner on `solab-p7` sends embedding requests there and does not need the model
copied locally.

### Run the current benchmark suite

After staging the five datasets, run sparse, dense, and hybrid retrieval over
all of them with:

```bash
./scripts/run_all_benchmarks.sh
```

The script builds one Qdrant collection per dataset for the configured
embedding model, then writes JSON files under `results/public/`. To reuse
collections on later runs:

```bash
BUILD_INDEXES=0 ./scripts/run_all_benchmarks.sh
```

Existing collections are reused by default. To reuse the original LitSearch
collection created earlier in this project while building the missing MTEB
collections:

```bash
LITSEARCH_COLLECTION=litsearch-reason ./scripts/run_all_benchmarks.sh
```

Set `REBUILD_INDEXES=1` only when every selected collection should be
re-embedded.

Non-empty result files are also reused by default. Force the benchmark runs
to overwrite them without rebuilding collections with:

```bash
FORCE_RERUN=1 ./scripts/run_all_benchmarks.sh
```

The main overrides are `CACHE_DIR`, `RESULTS_DIR`, `QDRANT_URL`,
`EMBEDDING_URL`, `EMBEDDING_MODEL`, `EMBEDDING_API_MODEL`, and `BATCH_SIZE`.
This suite does not apply query alignment or reranking.

If an older staging run put `princeton-nlp___lit_search/` directly under
`hub/`, move that processed dataset cache into `datasets/` before running
offline:

```bash
mkdir -p /mnt/nvme2/labuser/.cache/huggingface/datasets
mv /mnt/nvme2/labuser/.cache/huggingface/hub/princeton-nlp___lit_search \
   /mnt/nvme2/labuser/.cache/huggingface/datasets/
```
