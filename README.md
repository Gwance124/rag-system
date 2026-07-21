# rag-system

The repository now has two small projects:

- `latex-parser/` contains the original LaTeX parsing/chunking pilot, with its
  own `src/`, `scripts/`, `tests/`, and `pyproject.toml`.
- `src/retrieval/` and `scripts/run_public_bench.py` contain the retrieval and
  public-benchmark implementation.

See [`docs/retrieval-architecture.md`](docs/retrieval-architecture.md) for the
retrieval data flow, module responsibilities, and extension points.

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

ScholarGym-static is available as a custom single-shot retrieval extension. The
authors release `scholargym_bench.jsonl` (queries and ground-truth arXiv IDs)
and `scholargym_paper_db.json` (the title/abstract corpus) in the
[ScholarGym repository](https://github.com/shenhao-stu/ScholarGym) and on
[Hugging Face](https://huggingface.co/datasets/shenhao/ScholarGym). It is not
the official agentic ScholarGym evaluation; this runner reports the static
retriever metrics only.

The ScholarGym files should be in the Hugging Face cached repository:

```text
/mnt/nvme2/labuser/.cache/huggingface/datasets/datasets--shenhao--ScholarGym/
  blobs/
  refs/
  snapshots/<commit>/
    scholargym_paper_db.json
    scholargym_bench.jsonl
```

The runner follows `refs/main` to the correct snapshot automatically.

Then run the fast sparse check with:

```bash
python scripts/run_public_bench.py \
  --benchmark scholargym \
  --cache-dir /mnt/nvme2/labuser/.cache/huggingface \
  --mode sparse --top-k 100
```

Build and run its dense or hybrid collection with the same model-specific
prefixes used for the other benchmarks:

```bash
python scripts/build_dense_index.py \
  --benchmark scholargym \
  --cache-dir /mnt/nvme2/labuser/.cache/huggingface \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --embedding-api-model /model \
  --qdrant-url http://localhost:6333 \
  --collection scholargym-qwen3-embedding-0.6b

python scripts/run_public_bench.py \
  --benchmark scholargym \
  --cache-dir /mnt/nvme2/labuser/.cache/huggingface \
  --mode dense --top-k 100 \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --embedding-api-model /model \
  --qdrant-url http://localhost:6333 \
  --collection scholargym-qwen3-embedding-0.6b
```

Use `--scholargym-query-limit 200` for a quick smoke run. The full static
benchmark reports Recall@20/50 and nDCG@10 in the normal result JSON.

The MTEB retrieval adapter is not limited to that shortlist. `--dataset NAME` resolves
to `mteb/NAME`, and `--dataset-id ORG/NAME` accepts any cached MTEB-format
dataset directly. This covers the BEIR retrieval datasets hosted under MTEB
without adding a new loader for every dataset. `--benchmark beir` remains as
a backward-compatible alias.

### LMEB v4 QASPER chunk retrieval

QASPER supports three evidence-retrieval conditions. All use the same
questions, paragraph corpus, embeddings, and gold evidence labels:

- `--qasper-scope global` searches all QASPER chunks using only the question.
  This is a custom experiment for testing whether paper retrieval is needed.
- `--qasper-scope paper` uses LMEB's candidate list to rank only chunks from
  the known target paper. This isolates the Stage 2 chunk retriever.
- `--qasper-scope two-stage` first retrieves `--qasper-paper-top-k` papers
  from raw QASPER titles+abstracts, then searches only those papers' chunks.
  This is the end-to-end funnel diagnostic.

The paper is represented by the candidate restriction; its text is not
appended to the query. Global results have incomplete labels because QASPER
annotators marked evidence only inside the original target paper.
Two-stage results keep those evidence labels unchanged: retrieved chunks are
predictions, never replacement gold labels. The result JSON reports Stage 1
paper Recall@1/5/10/20/50/100/200, end-to-end evidence metrics, and evidence metrics
conditioned on Stage 1 retrieving the target paper.

The QASPER result block reports LMEB v4's official nDCG@10 and capped
Recall@10 in addition to the runner's standard metrics. LMEB v4 also evaluates
an instruction condition using `Given a query, retrieve documents that answer
the query`; supply it through the embedding model's required query-prefix
format when reproducing that condition.

Warm all four cached configurations before copying the Hugging Face cache to
the offline benchmark machine:

```python
from datasets import load_dataset

for config in (
    "QASPER-corpus",
    "QASPER-queries",
    "QASPER-qrels",
    "QASPER-top_ranked",
):
    load_dataset("mteb/QASPER", config, split="test", cache_dir="<hf-cache>/datasets")

```

The `allenai/qasper` main branch contains a legacy Python loader whose data
URLs point to AllenAI S3. On a host where S3 is blocked, stage Hugging Face's
converted Parquet files instead (about 26 MB total):

```bash
./scripts/download_qasper_parquet.sh <hf-cache>/qasper-parquet
```

Copy that directory to the offline machine. The default loader finds
`<hf-cache>/qasper-parquet` automatically; for another location, pass
`--qasper-raw-dataset-id PATH` or set `QASPER_RAW_DATASET_ID=PATH`.

Build one chunk collection shared by all conditions and one small paper
collection for two-stage retrieval:

```bash
python scripts/build_dense_index.py \
  --benchmark qasper --cache-dir <hf-cache> \
  --qdrant-url http://localhost:6333 --collection qasper-<model>

python scripts/run_public_bench.py \
  --benchmark qasper --qasper-scope global --cache-dir <hf-cache> \
  --mode dense --qdrant-url http://localhost:6333 --collection qasper-<model>

python scripts/build_dense_index.py \
  --benchmark qasper --qasper-corpus papers --cache-dir <hf-cache> \
  --qdrant-url http://localhost:6333 --collection qasper-papers-<model>

python scripts/run_public_bench.py \
  --benchmark qasper --qasper-scope two-stage --qasper-paper-top-k 20 \
  --cache-dir <hf-cache> --mode dense --qdrant-url http://localhost:6333 \
  --collection qasper-<model> \
  --qasper-paper-collection qasper-papers-<model>

python scripts/run_public_bench.py \
  --benchmark qasper --qasper-scope paper --cache-dir <hf-cache> \
  --mode dense --qdrant-url http://localhost:6333 --collection qasper-<model>
```

Add the same embedding model, API model, and query/passage prefix flags used
when building the collection. Use `--qasper-query-limit 25` for a smoke test.

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
embedding model. Model-independent BM25 files go under
`results/public/sparse/`; dense and hybrid files go under a model directory
such as `results/public/llama-nv-embed-reasoning-3b/`. To reuse collections on
later runs:

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

Individual benchmark/index errors do not stop the sweep. The script continues
with the remaining datasets and modes, then prints the failed configurations at
the end. Full stderr logs and the last error output are saved under
`results/public/failures/<MODEL_TAG>/summary.txt`. The script exits with status
1 after completing if any failure occurred; use `|| true` if a surrounding job
must ignore that final status.

Progress for the active run is written to the model-specific log. For example:

```bash
tail -f results/public/failures/Yuan-embedding-2.0-en/scholargym-sparse.log
```

The log reports BM25 documents indexed and queries evaluated. The current run
must be started after this progress change to contain those updates.

`MODEL_TAG` overrides the model result-directory name when comparing another
configuration of the same checkpoint.

Move results created by an older version of the script with:

```bash
mkdir -p results/public/sparse \
  results/public/llama-nv-embed-reasoning-3b
mv results/public/*-sparse.json results/public/sparse/
mv results/public/*-dense.json results/public/*-hybrid.json \
  results/public/llama-nv-embed-reasoning-3b/
```

The main overrides are `CACHE_DIR`, `RESULTS_DIR`, `QDRANT_URL`,
`EMBEDDING_URL`, `EMBEDDING_MODEL`, `EMBEDDING_API_MODEL`, `QUERY_PREFIX`,
`PASSAGE_PREFIX`, `MODEL_TAG`, and `BATCH_SIZE`. Set
`INCLUDE_QASPER=1` to append both QASPER global and paper-scoped retrieval:

```bash
INCLUDE_QASPER=1 ./scripts/run_all_benchmarks.sh
```

The default `QASPER_SCOPE=both` runs the original global and paper conditions.
Use `QASPER_SCOPE=all` for those plus two-stage, or select one condition with
`QASPER_SCOPE=global`, `paper`, or `two-stage`:

```bash
INCLUDE_QASPER=1 QASPER_SCOPE=global ./scripts/run_all_benchmarks.sh

INCLUDE_QASPER=1 QASPER_SCOPE=two-stage QASPER_PAPER_TOP_K=20 \
  ./scripts/run_all_benchmarks.sh
```

Two-stage result filenames include the paper depth, for example
`qasper-two-stage-k20-dense.json` and `qasper-two-stage-k100-dense.json`.
Different K values therefore coexist and appear as separate bars in the same
plot. K changes only the candidate cutoff, so all runs reuse the same QASPER
paper and chunk collections; no K-specific Qdrant collection is built.

For a locally downloaded QASPER repository, set `QASPER_DIR` to its root:

```bash
INCLUDE_QASPER=1 QASPER_SCOPE=global \
  QASPER_DIR=/path/to/datasets--mteb--QASPER \
  ./scripts/run_all_benchmarks.sh
```

The default is `$CACHE_DIR/datasets/datasets--mteb--QASPER`. When that path is
a Hugging Face cache repository, the runner resolves `refs/main` to the actual
`snapshots/<commit-hash>` directory automatically.
Set `QASPER_RAW_DATASET_ID` if the raw `allenai/qasper` title+abstract dataset
is staged under a different local dataset ID or path. By default, the loader
checks both `$CACHE_DIR/datasets/datasets--allenai--qasper` and
`$CACHE_DIR/hub/datasets--allenai--qasper`, then resolves `refs/main` to its
`snapshots/<commit-hash>` directory before loading offline. A main-branch
snapshot containing only `qasper.py` is not sufficient; use the converted
Parquet staging command above.

All conditions reuse the same QASPER chunk collection. Two-stage additionally
builds a small `qasper-papers-<model>` title+abstract collection. Set
`INCLUDE_SCHOLARGYM=1` to append ScholarGym-static to the sweep. It looks in
`$CACHE_DIR/datasets/datasets--shenhao--ScholarGym` by default; override that with
`SCHOLARGYM_DIR`, or use the two explicit file variables
`SCHOLARGYM_PAPER_DB` and `SCHOLARGYM_BENCHMARK_JSONL` when needed.
By default, the script first queries `$EMBEDDING_URL/models` and uses the
returned `data[0].id` as `EMBEDDING_MODEL`, so the result directory and Qdrant
collection match the model currently served by vLLM. Set `AUTO_DETECT_MODEL=0`
to use an explicit `EMBEDDING_MODEL` instead.
This suite does not apply query alignment or reranking.

### Swap the vLLM model on g3

On the g3 checkout, start vLLM through the wrapper with the model you want:

```bash
VLLM_MODEL=Qwen/Qwen3-Embedding-0.6B ./scripts/serve_embedding.sh
```

The wrapper always advertises the server as `/model`, so the benchmark host
does not need a server-side model-name change. Point the benchmark run at the
same checkpoint name so result directories and Qdrant collections stay
separate:

```bash
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B \
MODEL_TAG=qwen3-embedding-0.6b \
./scripts/run_all_benchmarks.sh
```

Keep `EMBEDDING_API_MODEL=/model` (the default). Use a new collection/model
tag for every checkpoint; vector dimensions and spaces may differ.

### Checkpoint 1 report

After the sparse/dense/hybrid files for the current embedding lineup are in
`results/public/`, generate the CP1 decision table and LitSearch reference
deltas with:

```bash
python scripts/report_checkpoint1.py \
  --results-dir results/public \
  --output results/public/checkpoint1.md
```

The report compares BM25 and every available model/mode on LitSearch R@5/R@20
and SciFact nDCG@10, includes the SciDocs/NFCorpus/TREC-COVID sanity checks,
and prints Recall@100 for choosing CP2's candidate depth. The recommended
winner is provisional until the Qwen3-Embedding-0.6B sweep is present.

To add that sweep later:

```bash
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B \
MODEL_TAG=qwen3-embedding-0.6b \
./scripts/run_all_benchmarks.sh
```

### Graph model/pipeline differences

Once `results/public/` contains the benchmark JSON files, make one graph per
dataset:

```bash
python -m pip install -e '.[eval]'  # once, if matplotlib is not installed
python scripts/plot_benchmark_results.py \
  --results-dir results/public \
  --output-dir results/plots
```

This writes one file per discovered dataset, such as `litsearch.png`,
`mteb-scifact.png`, and `scholargym.png`. QASPER's three retrieval conditions are
kept separate as `qasper-global.png`, `qasper-paper.png`, and
`qasper-two-stage.png`; `qasper-two-stage-papers.png` plots the Stage 1 paper
recall curve once per model/mode (K does not change the Stage 1 ranking). Use
`--metrics recall@20,recall@50,recall@100,recall@200` to inspect deeper paper
cutoffs. Each graph compares
every discovered model/pipeline, including BM25, dense, and hybrid, using
Recall@5/20/50 and nDCG@10. Use `--metrics recall@20,recall@50` for a
ScholarGym-focused graph.

LitSearch also produces `litsearch-average-broad.png` and
`litsearch-average-specific.png` from the paper-comparable broad/specific
subsets stored in each result's `litsearch_paper_comparison` block.

If an older staging run put `princeton-nlp___lit_search/` directly under
`hub/`, move that processed dataset cache into `datasets/` before running
offline:

```bash
mkdir -p /mnt/nvme2/labuser/.cache/huggingface/datasets
mv /mnt/nvme2/labuser/.cache/huggingface/hub/princeton-nlp___lit_search \
   /mnt/nvme2/labuser/.cache/huggingface/datasets/
```
