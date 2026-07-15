# Retrieval Pipeline + Auto-Labeled Evaluation — Design Spec

## Context

Sub-project 1 (`2026-07-13-latex-chunking-pilot-design.md`) produced a working LaTeX chunking pipeline over the arxiv-latex parquet corpus. This sub-project builds the retrieval pipeline and — the central problem — an evaluation strategy that needs no hand-labeled relevance judgments, which the team (CXL memory / KV-cache / LLM serving / RAG research) cannot produce.

The core insight for evaluation: the citation graph is a relevance labeler. When paper A's related-work section cites papers B and C in a sentence, that sentence is a relevance judgment written by a domain expert. This is how LitSearch itself was constructed; we replicate the method on our corpus with the local Qwen 400B.

Sequencing principle (user decision): **validate the pipeline on public benchmarks first** to confirm the baseline is sound, then transition to the domain-specific experiment. If the pipeline underperforms on BRIGHT/LitSearch, fix that before investing in corpus and benchmark construction.

## Decisions

- **Corpus:** focused 2–10k papers via seed + citation expansion, not the full 650k / 18.7k sets.
- **Index phasing:** abstract-level index first (Phase A), full-text chunk index second (Phase B, reusing `latex-parser/src/chunking/`). These are *competing variants* scored by the same eval — NOT stages of one query path. Abstract-first routing is rejected as a recall trap (a paper whose relevant content is only in section 5 dies at the abstract gate).
- **Retrieval evaluation:** compare embedding models first, then query alignment/rewrite variants, then rerankers over a fixed candidate set. Hybrid BM25 + dense with RRF remains one pipeline variant, not a predetermined winner.
- **Scale sanity:** 2–10k papers ≈ ≤1M chunks — flat HNSW in Qdrant at ms latency. Long papers (95k tokens) are irrelevant to the index once chunked.
- **Eval labels at the paper level** (query → set of relevant arXiv IDs), so one benchmark scores Phase A, Phase B, and every variant identically (chunk hits aggregate to papers via metadata).
- **Current lab setup:** `nvidia/llama-nv-embed-reasoning-3b` is served by vLLM at `192.168.3.4:8000` (OpenAI-compatible `/v1/embeddings`) under the API model alias `/model`. The benchmark and corpus datasets are on `solab-p7`, which is the benchmark execution host. The benchmark checkout is `/mnt/nvme2/mlee/rag-system`; its Hugging Face cache root is `/mnt/nvme2/labuser/.cache/huggingface` (`hub/` for Hub artifacts and `datasets/` for dataset files), keeping both off the home disk. Hugging Face loading at benchmark time is cache-only (`local_files_only=True` plus offline mode); a cache miss must fail rather than reach the internet.
- **Constraints:** BRIGHT is already available on Hugging Face (`xlangai/BRIGHT`; the MTEB mirror is `mteb/BrightRetrieval`), as are LitSearch (`princeton-nlp/LitSearch`) and MTEB's BEIR-format tasks. SciDocs (`mteb/scidocs`) is the default BEIR task; SciFact and TREC-COVID are initial scientific checks, while the adapter accepts any cached MTEB-format BEIR dataset. Warm the cache once, copy it to `solab-p7`, and make all runtime HF loads cache-only. Semantic Scholar API is plain HTTPS; fallback is running fetch steps on the personal laptop and scp-ing the small JSON outputs. PyPI assumed reachable, but the first BM25 implementation is dependency-free.
- **Deferred:** end-to-end answer-quality eval (RAGAS-style), latency *optimization* (measurement is built in from day one), and agentic/iterative retrieval (BRIGHT-Pro style). Query alignment and reranking are part of the public-benchmark phase before corpus construction.

## Architecture

```
                 QUERY PATH (shared by all variants)
 query ─► [optional: Qwen query rewrite] ─► BM25 top-N ∥ dense top-N
       ─► RRF fusion ─► top-k documents ─► aggregate to papers (chunk indexes only)
       ─► [optional: rerank variant] ─► Qwen 400B answers with retrieved context

                 INDEXES (competing variants)
 Phase A: title+abstract per paper        Phase B: ~512-token chunks (existing chunker),
 (BM25 + nv-embed)                        chunk→paper metadata, error-budget fallback

                 EVAL TRACKS
 Track 1 (first): LitSearch + SciDocs, then SciFact/TREC-COVID and BRIGHT — embedding, query-alignment, fusion, and reranking selection
 Track 2: citation-derived domain benchmark — the reported numbers
 Track 3: live teammate queries — qualitative, covers novel connections
```

## Initial implementation

The first slice is now in `src/retrieval/`: dependency-free BM25, weighted RRF,
chunk-to-paper aggregation, shared paper-level metrics, cache-only HF loaders
for LitSearch and MTEB/BEIR, LitSearch paper-aligned R@5/R@20 subgroup
reporting, a vLLM embeddings client, and a small Qdrant REST adapter.
`scripts/run_public_bench.py` runs sparse, dense, or hybrid retrieval against
LitSearch, BRIGHT, any cached MTEB-format BEIR task (SciDocs by default), or a
generic local JSONL benchmark.

## Deliverables (in order)

### 1. Retrieval pipeline core (`src/retrieval/`)
Index-agnostic: takes any collection of `{doc_id, text, paper_id}` records.
- Dense: nv-embed-reason-3b via the existing vLLM serving; vectors into Qdrant.
- Sparse: BM25 (`bm25s`, pure Python from PyPI).
- Fusion: reciprocal rank fusion with configurable weight; optional Qwen query-rewrite step behind a flag.
- Shared metrics module: nDCG@10, Recall@10/50/100, MRR — used by both the public runner and the domain harness.
- Per-stage latency logging (rewrite, embed, search, fuse, generate) from the start.
- Every knob (rewrite on/off, N, k, RRF weight, index choice) is a named config so evals sweep variants.

### 2. Public benchmark track — the baseline gate (`scripts/run_public_bench.py`)
- Stage **LitSearch** and **SciDocs** first. Add **SciFact**, **TREC-COVID**, and **BRIGHT**; support every other cached MTEB-format BEIR task through the same generic loader rather than dataset-specific code.
- Adapter runs the same pipeline variants over these datasets with their official metrics.
- **Gate criteria:** (a) dense-only scores are in a plausible published range — confirms embedding + search plumbing is correct; (b) one frozen configuration is strong across the benchmark suite without major regressions. Sweep embedding model first, then query alignment, then reranking over fixed top-N candidates, and finally compare combined sparse/dense/hybrid variants. Only then proceed to the domain experiment; if the baseline is broken, fix the pipeline first.

### 3. Corpus builder (`src/corpus/`, `scripts/build_corpus.py`)
- Input: `seeds.txt` of ~50–150 arXiv IDs collected from teammates' reading lists and the lab's own projects.
- Expand 1–2 hops via Semantic Scholar Graph API (`/paper/{id}/references`, `/paper/{id}/citations`), keeping papers with arXiv IDs. Persist citation edges **and citation contexts** to `citation_graph.jsonl` — raw material for deliverable 4.
- Keyword net over the local arxiv-latex parquet's title+abstract ("CXL", "compute express link", "KV cache", "disaggregated memory", "prefill", "speculative decoding", "retrieval-augmented", …) to catch papers outside the citation neighborhood. Reuse filtering patterns from `latex-parser/scripts/filter_pilot.py`.
- Union → dedupe → filter local parquet by ID → `corpus.parquet`.
- **Stopping criterion:** ≥90% of seed papers' references present in the corpus; report the coverage number.

### 4. Domain benchmark builder (`src/benchmark/`, `scripts/build_benchmark.py`)
Consumes `citation_graph.jsonl` + Qwen 400B. Output: `benchmark.jsonl` — `{query_id, query, positives: [arxiv_ids], type, excluded_ids}`.

- **Type 1 — citation-context queries (~500–1000):** LitSearch's method. Citing sentence from paper A referencing B → Qwen writes a self-contained question → positives = papers cited in that sentence.
- **Type 2 — co-citation "connect ideas" with held-out citing paper:** paper A co-cites B and C in one passage → Qwen writes a question phrased as the *problem/motivation*, never naming B or C (BRIGHT-style) → positives = {B, C}; **A is excluded from the corpus for this query**. This is the temporal counterfactual: the query tests surfacing a connection that did not exist when B and C were written — the retroactive version of a teammate's novel-combination idea.
- **Automatic quality filters:** drop queries with high n-gram overlap against positives' titles/abstracts (anti-lexical-shortcut); Qwen-as-judge pass for self-containedness / answerability / non-triviality; dedupe near-identical queries.
- **Stretch (not v1):** future-work-gap queries (future-work sentence of A → positives = later papers citing A that address it, Qwen-verified from citation contexts).
- **Known limit (accepted):** connections no paper has ever made have no ground truth by definition; Track 3 covers those qualitatively.

### 5. Domain eval runs (`scripts/run_eval.py`)
- Paper-level metrics over `benchmark.jsonl`, per-query-type breakdown, variant matrix comparison table, per-stage latency stats.
- Phase A (abstract index) numbers are the first domain result.

### 6. Phase B — chunk-level index
- Run the existing `latex-parser/` chunking pipeline over `corpus.parquet` with an **error budget**: papers that parse badly fall back to a title+abstract-only record, logged. No more chasing every LaTeX edge case.
- **Statistical guards, not more LaTeX heuristics** (pilot stats showed a pathological tail — max 5.4M cleaned tokens/paper, max 3054 tokens/chunk — caused by non-prose data leaking through, not by fixable parse rules):
  - *Block-level prose filter:* drop blocks that don't look like language (high digit/symbol ratio, low sentence-boundary density) — kills coordinate dumps, giant data tables, and blobs regardless of which LaTeX construct produced them.
  - *Paper-level cap:* cleaned paper above ~150k tokens → pathological; fall back to title+abstract-only record, log it.
  - *Chunk-level hard cap:* no chunk exceeds the embedding token budget; force-split as last resort.
  - *Tail reporting:* stats output includes p95/p99 paper tokens and counts of capped/fallback papers, so tail severity is a tracked number.
- Index chunks (dense + BM25), aggregate chunk scores to papers (max chunk score per paper).
- Score against the same `benchmark.jsonl`; the Phase A vs Phase B delta — especially on Type 2 queries — is the headline result justifying (or rejecting) full-text retrieval.

### 7. Live query set (qualitative)
- 10–20 real queries from teammates (e.g., using a hippocampus-style agentic memory *alongside* a vector DB rather than replacing it); run through best variants; teammates rate usefulness. Demo set + reality check on novel-connection queries.

## Error Handling

- S2 API: rate-limited (~1 req/s unauthenticated); builder must checkpoint progress and resume, and tolerate missing papers (not everything has an arXiv ID or S2 record) by logging and skipping.
- Chunk parsing failures: error-budget fallback to abstract-only, logged with paper ID (extends the existing `parse_failures.jsonl` pattern). Guard trips (prose filter, paper cap, chunk cap) are logged the same way — a guarded paper degrades gracefully instead of blocking the run or polluting the index.
- Benchmark generation: Qwen outputs that fail the judge filter or JSON parsing are dropped and counted; report the survival rate so filter aggressiveness is visible.

## Testing Strategy

- Unit tests (runnable on the personal laptop, no real data): RRF fusion math, chunk→paper aggregation, metrics module against tiny hand-computed qrels, benchmark-builder filters (n-gram overlap, dedupe) on synthetic fixtures.
- Public-benchmark gate (deliverable 2) doubles as the integration test for the whole retrieval stack.
- Deliverable 4 verification: manually read ~30 sampled generated queries across both types — self-contained? no title leakage? sensible positives? (Reading, not labeling — no domain expertise needed to spot broken queries.)
- Sanity check in every eval run: a random-retrieval variant must score near zero; BM25-only must land in a plausible band.
