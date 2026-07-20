# Retrieval Architecture

The retrieval package turns an offline benchmark corpus into ranked document IDs and metrics. The code has one main path:

```text
Hugging Face cache or JSONL
          ↓
benchmarks.py → Benchmark(documents, queries, qrels, exclusions, candidates)
          ↓
BM25Index and/or QdrantIndex
          ↓
HybridRetriever → reciprocal-rank fusion → top-k hits
          ↓
metrics.py → result JSON → reports and plots
```

## Core Modules

| Module | Responsibility |
| --- | --- |
| `types.py` | Shared `Document`, `SearchHit`, configuration, and result records. |
| `benchmarks.py` | Offline-only loaders that normalize JSONL, BRIGHT, LitSearch, MTEB/BEIR, QASPER, and ScholarGym into `Benchmark`. |
| `sparse.py` | In-memory BM25 indexing and search through `bm25s`. |
| `dense.py` | Embedding requests to the OpenAI-compatible vLLM endpoint and vector storage/search through Qdrant REST. |
| `fusion.py` | Weighted reciprocal-rank fusion. It also exposes optional chunk-to-paper aggregation; the benchmark CLI currently evaluates returned document IDs directly. |
| `pipeline.py` | Coordinates query rewriting, sparse search, dense search, exclusions, fusion, final truncation, and timings. |
| `metrics.py` | Computes recall, nDCG, MRR, and LitSearch paper-baseline comparisons. |

`Benchmark` is the boundary between data loading and retrieval. A loader must supply documents, query text keyed by query ID, relevant document IDs (`qrels`), and IDs excluded from each result. It may also provide per-query `candidate_ids`; sparse and dense search apply that scope before fusion. `top_n` controls candidates fetched from each index; `top_k` controls the final list after fusion and exclusions.

QASPER uses this optional scope to expose two conditions without changing the
query or corpus. Global mode leaves `candidate_ids` unset and searches every
paragraph. Paper-scoped mode loads LMEB's `top_ranked` IDs and searches only
those paragraphs. A single Qdrant collection therefore serves both modes.

## Entry Points

- `scripts/run_public_bench.py` loads one benchmark, constructs the requested `sparse`, `dense`, or `hybrid` retriever, evaluates every query, and writes JSON to stdout.
- `scripts/build_dense_index.py` embeds a corpus and upserts it into one Qdrant collection. Collections must be unique per corpus and embedding model.
- `scripts/run_all_benchmarks.sh` orchestrates index builds and benchmark runs, reuses existing artifacts, and records per-configuration failures.
- `scripts/serve_embedding.sh` launches vLLM with a stable served model name.
- `scripts/report_checkpoint1.py` and `scripts/plot_benchmark_results.py` consume saved result JSON; they do not run retrieval.

Sparse mode needs only the local corpus. Dense mode embeds the query and searches an existing Qdrant collection. Hybrid mode runs both paths and combines their ranks with equal default weights. Hugging Face loaders force offline mode, so cache misses fail instead of downloading data.

## Extending the System

To add a benchmark, implement a loader returning `Benchmark`, expose its CLI arguments in both run and index scripts, record the document-text composition in the result config, and add a small loader test with in-memory rows. Add it to `run_all_benchmarks.sh` only when it belongs in the standard sweep.
