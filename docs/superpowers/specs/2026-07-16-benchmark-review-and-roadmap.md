# Benchmark Review & Corrected Evaluation Roadmap

## Context

Follow-up to `2026-07-14-retrieval-eval-design.md`. That spec established the pipeline architecture and the public-benchmarks-first sequencing; this one reviews the proposed 3-checkpoint benchmark design (CP1 title+abstract retrieval → CP2 hierarchical full-paper retrieval → CP3 agentic research), verifies every candidate benchmark against its actual paper, and fixes the roadmap to a <1-month budget that still covers additional embedding models.

Target system: agentic RAG over ~600K arXiv papers (CXL / KV-cache / LLM inference / memory systems), Qdrant vector DB, local Qwen3 ~400B generator, embeddings served by vLLM on the lab box. Benchmark execution host: `solab-p7` (`/mnt/nvme2/mlee/rag-system`), HF loads cache-only.

## Decisions

- **No cross-encoder reranker.** The retrieval funnel tops out at hybrid RRF (BM25 + dense). Reranking-stage work is cut for time; the architecture keeps the frozen-top-N hook if it's ever revisited.
- **Embedding lineup:** `nvidia/llama-nv-embed-reasoning-3b` (incumbent), **Qwen3-Embedding-4B**, **Qwen3-Embedding-0.6B** (new), plus existing **Yuan-embedding-2.0-en** results retained in the comparison table. Qwen3-0.6B doubles as the ScholarGym-comparable and production-latency option — ScholarGym's environment is literally BM25 + Qwen3-0.6B + Qdrant over title+abstract.
- **Existing results are reused.** LitSearch/SciFact/SciDocs/TREC-COVID runs for nv-embed-3b, Qwen3-4B, and Yuan already exist on `solab-p7` from this repo's `run_all_benchmarks.sh`, so they are harness-comparable; result/index caching skips reruns automatically. Near-term work = add Qwen3-0.6B and build the decision table.
- **Model comparison is always raw-query.** Query rewriting/HyDE/decomposition are separate ablations on the winning model only (the `query_rewriter` hook in `src/retrieval/pipeline.py`); multi-round decomposition belongs to CP3.
- **Every result JSON keeps full config + ranked run files** so later stages re-score frozen runs instead of re-retrieving.

## Benchmark verdicts (all verified against their papers, 2026-07-16)

| Benchmark | Status | Verdict |
|---|---|---|
| **LitSearch** (ACL'24) | wired | **CP1 primary.** 597 NL literature-search queries, paper-level qrels, T+A corpus (~64K). Primary metric Recall@5 (official), secondary R@20; R@100 as funnel-coverage diagnostic. Cleanest model comparison (postdates most training mixes). `metrics.py` already diffs against published baselines. |
| **SciFact** (BEIR) | wired | **CP1 secondary.** Claim-shaped queries ≠ literature search, but claim verification is a real sub-task and it's tiny/cheap. nDCG@10 (official). Tiebreaker only. |
| **SciDocs / NFCorpus / TREC-COVID** (BEIR) | wired | **Sanity checks only.** SciDocs relevance is a citation proxy over title-queries; NFCorpus/TREC-COVID are off-domain. Never drive model choice. TREC-COVID (~171K docs) is first to skip when indexing time is short. Contamination note: MTEB-leaderboard models often train on BEIR-adjacent data — scores may be differentially inflated. |
| **ScholarGym** (arXiv 2601.21654) | to add | **Two uses.** (a) Official (agent policy over their fixed retriever) → **CP3**, the tractable agentic benchmark. (b) **"ScholarGym-static" (custom extension, CP1/production-scale):** extract corpus+qrels, run single-shot retrieval. Valid because gold sets come from PaSa (citation-derived + human-curated) and LitSearch — retriever-independent, no pooling bias — and 570K T+A corpus ≈ our ~600K production corpus. ~2.3 gold papers/query → Recall@20/@50 primary, nDCG@10 secondary. Caveats: label as custom extension (not comparable to their agent tables); partial query overlap with LitSearch (value is the 9× corpus, not independence); embed only BM25 + LitSearch winner (+ runner-up); **confirm the data release exists before committing.** |
| **QASPER** (NAACL'21) | to add | **CP2 primary.** 5,049 questions / 1,585 NLP papers, gold evidence paragraphs, token-F1 answers, unanswerable questions → grounded refusal comes free. Oracle-paper condition = official usage; end-to-end (global paper→chunk→answer) = our extension, reported as pipeline diagnostic (1,585-paper retrieval is easy — flag it). |
| **RPC-Bench** (arXiv 2601.14289, ACL'26) | cut | Real (15K review-rebuttal QA, LLM-judge scoring) but it's a generation-comprehension benchmark with no clean gold-passage retrieval target — cannot isolate chunk retrieval. Optional 200-question subset post-roadmap. |
| **"ResearchQA 2026 / Citation-Grounded ResearchQA"** | cut — **does not exist as described** | Real ResearchQA (arXiv 2509.00496) is open-domain survey-mined long-form QA with rubrics, not known-paper QA with grounded refusal. QASPER already covers the described capability. |
| **AutoResearchBench** (arXiv 2604.25256) | deferred | Real; Deep Research (find target paper) + Wide Research (collect qualifying set, IoU). Frontier models ~9% and it presumes open-web access. CP3 stretch goal only; ScholarGym is the realistic CP3 target. |

Coverage: discovery → LitSearch (+ScholarGym-static at scale); passage retrieval → QASPER evidence; generation+citation+refusal → QASPER answers; agentic → ScholarGym (CP3), AutoResearchBench (stretch).

## Methodology rules

- **Stage separation:** freeze ranked run files per stage; downstream stages re-score frozen candidates. CP2 candidate depth N is measured, not assumed: report paper-level Recall@N; N=100 stands if ≥ ~90–95%, else deepen to 200.
- **CP2 core claim is the triple** {T+A only, candidate-restricted chunk search, global chunk search}, grouped to paper level and scored on the same qrels. Restriction is valid only when reported alongside its ceiling (candidate-stage Recall@N) and the global baseline. This keeps faith with the 07-14 spec's rejection of abstract-gate routing: the funnel must *prove* it beats the recall trap, not assume it.
- **Aggregation:** max chunk score (already implemented, `aggregate_to_papers` in `src/retrieval/fusion.py`) vs mean of top-3 — one ablation, nothing fancier. Qdrant `search_groups` (`group_by: paper_id`) gives top-k-per-paper server-side.
- **Qdrant schema:** two collections, `papers` + `paper_chunks`; canonical ID = normalized internal ID (`arxiv:2407.12345`, version-stripped; S2 corpus ID for non-arXiv), DOI/S2 as payload. Payload indexes: `paper_id` (keyword, chunks), `year` (integer), `categories` (keyword) — nothing else. Required additions to `QdrantIndex` (`src/retrieval/dense.py`): payload-index creation (`PUT /collections/{c}/index`) and `filter` (`match: {any: [...]}` — efficient for 50–200 candidate IDs) / `search_groups` support.
- **Metrics:** Recall@5/@20 (LitSearch), nDCG@10 (BEIR), Recall@20/@50 (ScholarGym-static), Evidence Recall@k + F1 and Answer F1 (QASPER). Skip MAP (qrels too sparse). P/R/F1 only where a benchmark defines a final selected set.
- **Fairness:** each model uses its own documented query/passage prefixes; Qwen3 family runs share the same instruct-prefix convention. Pin HF dataset revisions.
- **LLM judges:** none needed inside this roadmap (all retained benchmarks score via qrels or token F1). When judges appear later (custom benchmark, RPC-Bench), never judge Qwen outputs with only a Qwen judge; second-family judge on a 10–20% subsample with reported agreement.
- **Qwen3-400B enters at Week 3 only** (QASPER answer generation). Retrieval evaluation stays generation-free.

## Roadmap (~4 weeks)

### Now / Week 1 — Finish CP1
1. Serve Qwen3-Embedding-0.6B on the lab vLLM box.
2. Run the sweep for it: `EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B MODEL_TAG=qwen3-embedding-0.6b` + the 4B run's prefix convention → `scripts/run_all_benchmarks.sh` (caching backfills anything missing for other models).
3. Build the CP1 decision table: rows {BM25, dense×4, hybrid-RRF×4}; columns LitSearch R@5/R@20 · SciFact nDCG@10 · appendix (SciDocs/NFCorpus/TREC-COVID). Include `LITSEARCH_PAPER_*` reference diffs.
4. **Deliverable: frozen winning model + mode**, plus Recall@100 per benchmark to set CP2's N.

### Week 1–2 overlap — ScholarGym-static (overnight jobs)
1. Locate the ScholarGym data release (paper claims open-source; if corpus or gold labels aren't actually released, drop this item — do not rebuild by hand this month).
2. Convert to the repo's JSONL benchmark format (`load_jsonl_benchmark` hook in `src/retrieval/benchmarks.py`).
3. Embed/index overnight for BM25 + the LitSearch winner (+ runner-up if close). Report Recall@20/@50, nDCG@10, labeled "ScholarGym-static (custom extension)"; record Qdrant build time, memory, p50/p95 latency at 570K scale as the production capacity data point.

### Week 2 — CP2 infrastructure + retrieval-only eval
1. Wire `latex-parser` chunks into a `paper_chunks` collection; add payload indexes + filter/`search_groups` to `QdrantIndex`.
2. Add QASPER loader; build its chunk index.
3. QASPER oracle-paper evidence retrieval (Evidence Recall@k / F1) — chunking + chunk retrieval in isolation.
4. LitSearch triple {T+A, restricted-chunk, global-chunk} + aggregation ablation (max vs top-3 mean).
5. **Deliverable: CP2 retrieval table; frozen funnel config (N, aggregation).**

### Week 3 — End-to-end generation; domain corpus in parallel
1. QASPER end-to-end with Qwen3-400B: paper Recall@k, evidence Recall@k, Answer F1, unanswerable-question accuracy (grounded refusal). Condition flagged as custom extension.
2. Parallel: domain corpus ingestion (arXiv + Semantic Scholar → `papers` + `paper_chunks`).
3. One query-rewrite ablation on the frozen config.

### Week 4 — Domain benchmark v0 + writeup
1. 50 domain queries (CXL/KV-cache/serving), pooled top-20 judging across all system variants + citation-derived silver qrels (per the 07-14 spec's citation-graph labeling insight). Graded 0/1/2; re-judge a random 10% later for intra-annotator consistency; document single-annotator status.
2. Writeup: CP1/CP2/ScholarGym-static tables, funnel-ceiling analysis, ground-truth caveats, contamination note.
3. If time remains: ScholarGym official-mode feasibility spike (CP3 on-ramp).

**Cut entirely:** cross-encoder reranking, RPC-Bench, ResearchQA (both variants), AutoResearchBench (CP3 stretch), learned/LLM aggregation, HyDE/multi-round decomposition.

## Verification

- Week 1: hybrid LitSearch R@5 at/above the paper's dense baselines (`LITSEARCH_PAPER_*` diff); Qwen3-0.6B dense-vs-sparse gain sanity-checked against ScholarGym's reported +4.5–12.5 pt recall range.
- Week 2: oracle-paper QASPER evidence recall vs. the QASPER paper's baselines; restricted-chunk LitSearch ≥ T+A-only (a miss is a real, reportable negative result for the funnel design).
- All runs reproduce from cache with `FORCE_RERUN=0`; every result JSON embeds full config.

## Sources

- ScholarGym: <https://arxiv.org/abs/2601.21654> · RPC-Bench: <https://arxiv.org/abs/2601.14289> · ResearchQA: <https://arxiv.org/abs/2509.00496> · AutoResearchBench: <https://arxiv.org/abs/2604.25256> · LitSearch: <https://arxiv.org/abs/2407.18940>
