# Four-week local RAG systems implementation and experiment plan

Status: implementation plan only. No runtime component in this document has
been implemented yet.

Date: 2026-07-21

## Outcome and scope

In four weeks, build and evaluate one controlled progression on the same fixed
BrowseComp-Plus corpus and one dedicated 1x A100 80 GB vLLM server:

1. single-pass RAG;
2. iterative, sequential single-agent research; and
3. parallel multi-agent research.

The primary question is whether added retrieval breadth improves evidence
coverage and answer quality enough to justify cumulative token/latency cost and,
for parallel agents, peak concurrent KV-cache pressure. This is a hypothesis,
not an expected ordering.

The scientific-retrieval and LaTeX work under `old/` remains background and
appendix material. QASPER must not be optimized or included in the new result
tables. WixQA is a pipeline sanity fallback only.

## 1. Critical assessment of the research design

### What is strong

- BrowseComp-Plus fixes both the corpus and relevance judgments, so retrieval,
  generator, and systems changes can be compared without live-web drift.
- Evidence and gold-document labels permit retrieval/context diagnostics that
  are more informative than answer accuracy alone.
- Retrieved and oracle context arms separate document-discovery failure from
  some generator failure.
- Freezing one generator, retriever, tokenizer, prompt, chunking scheme, and
  vLLM configuration makes workflow structure and context budget interpretable.
- Separating cumulative workflow tokens from concurrently live tokens directly
  targets the systems hypothesis.

### Corrections and limitations to make explicit

1. **A 32K context model is not sufficient for a 32K retrieved-context arm.**
   Retrieved text is only part of the prompt. The model gate must verify that
   32,768 rendered context tokens plus the system prompt, question, chat
   template, source headers, and maximum output fit below the configured model
   length with a safety margin. Prefer a configured total length of at least
   40K; record whether length extension is used.

2. **The benchmark has no official development split.** The 830 released
   examples are test examples. Create a stable 100-query development split by
   hashing only `seed || query_id`, save its ID list and hash, and reserve the
   other 730 queries for a protocol-frozen evaluation. Dev results are
   exploratory; held-out results are confirmatory. Do not repeatedly tune on
   the 730.

3. **The oracle arm is a document oracle, not a perfect answer oracle.** Full
   evidence pages can exceed every context budget. Define the oracle candidates
   as chunks whose parents are labeled evidence documents, rank those chunks
   using only the question and the same fixed BM25 scoring/tie-breaking, then
   use the same packer. This removes global document discovery but retains
   within-document localization and budget truncation. Call the stored strategy
   `evidence_document_oracle`; do not claim that an oracle failure is purely a
   generator failure.

4. **Deterministic answer metrics are diagnostic, not official accuracy.**
   BrowseComp-Plus's official end-to-end evaluation is judge-based, while exact
   match, containment, and token F1 can penalize correct paraphrases or aliases.
   Week 1 must not depend on an LLM judge. Add official evaluation after the
   deterministic pipeline is frozen, and report the two families separately.

5. **Stream events are not guaranteed to be individual tokens.** The chat
   client must record every raw SSE content event, its arrival time, returned
   logprob token records when available, and the number of tokenizer tokens in
   each event. Report TTFT exactly at the client. Report inter-token latency
   only when the installed server returns one token per timed event; otherwise
   name it inter-event latency and keep token timing granularity in the schema.

6. **Prometheus timing histograms are server-wide aggregates.** A before/after
   histogram delta on an exclusive, sequential server can explain a request but
   is not a request trace. Use vLLM per-request metrics if the installed version
   exposes them; otherwise keep queue/prefill/decode fields nullable with
   provenance. Never assign a server-wide histogram quantile to a specific
   request.

7. **Chunk retrieval changes the official retrieval task.** Official qrels are
   document-level. For official comparison, collapse the frozen chunk ranking
   to parent documents by first occurrence and emit a TREC run. Separately
   report the primary context-at-token-budget metrics. Do not mix these two
   evaluations.

8. **A single retriever controls scope but limits generality.** Use BM25 for the
   Week 1 system baseline because it is local, deterministic, and already has a
   tested legacy implementation. The conclusions will be about workflow changes
   under that retriever, not about the best possible BrowseComp-Plus system.
   A retriever sensitivity study is follow-on work, not a Week 1 task.

9. **Algorithmic concurrency and serving concurrency must be disentangled.**
   In Week 4, first run the multi-agent algorithm and save every generated
   request. Then replay an identical fixed request set at concurrency 1/2/4
   (and 8 only if safe). The live workflow measures quality and breadth; replay
   isolates queueing, KV pressure, and preemption from differences in prompts.

10. **The full factorial is expensive.** After the Week 1 smoke, estimate GPU
    hours as `sum(expected_calls * observed_p50_seconds) / 3600 * 1.25`. Run the
    full 2-source x 5-budget factorial on the 100-query dev set. Run only frozen,
    selected configurations on the held-out set in Week 4. If the projected
    held-out run does not fit, use a precommitted 200-query hash sample and label
    it as such; do not quietly reduce the sample after seeing results.

## 2. Repository assessment and reuse map

The repository is currently mid-migration. Thirty-nine tracked files from the
old active layout are deleted and an untracked `old/` tree contains the prior
code. The root `pyproject.toml`, `README.md`, and previous `AGENTS.md` still
point to root `src/`, `scripts/`, `tests/`, and `latex-parser/`, which no longer
exist. The files in `old/` also use `old.src...` imports, so the archive is not
a clean standalone installed package. Treat it as source material, not a
dependency.

### Reuse by porting with new tests

| Legacy file | Reusable behavior | Required change |
| --- | --- | --- |
| `old/src/retrieval/types.py` | Small immutable document/hit/config records | Replace paper-specific fields with parent document/chunk offsets and the new contracts. |
| `old/src/retrieval/sparse.py` | `bm25s` tokenization, Lucene-style BM25, deterministic hit construction | Add persistent save/load, streaming chunk ingestion, stable tie-breaking, index metadata, and no all-corpus rebuild per run. |
| `old/src/retrieval/dense.py` | Dependency-light `urllib` JSON client and Qdrant REST patterns | Remove the hard-coded lab IP/model, add typed errors/retries, include chunk payload/offsets, and keep it optional in Week 1. |
| `old/src/retrieval/fusion.py` | Weighted RRF and deterministic doc-ID tie break | Port only for later retriever sensitivity or hybrid work; it is not needed for the BM25 Week 1 baseline. |
| `old/src/retrieval/pipeline.py` | Monotonic retrieval stage timing and index-agnostic orchestration | Split retrieval from workflow execution, remove rewrite from the single-pass path, and return query-associated traces. |
| `old/src/retrieval/metrics.py` | Pure recall/nDCG/MRR functions | Add per-query context, answer, citation, and redundancy metrics; keep official and experimental metrics separate. |
| `old/scripts/build_dense_index.py` | CLI/progress reporting pattern | Keep entry points thin and move logic into the active package. Do not retain benchmark-specific branches. |
| `old/scripts/run_public_bench.py` | Query loop, progress updates, result assembly ideas | Do not copy the 500-line multi-benchmark runner. Replace it with workflow-specific commands and versioned JSONL. |
| `old/scripts/plot_benchmark_results.py` | Headless Matplotlib setup and safe output names | Rewrite loaders for long-form JSONL and confidence intervals. |
| `old/tests/test_retrieval.py` | In-memory fixtures, monkeypatched datasets, fake Qdrant HTTP, timing assertions | Split by active module and add SSE/telemetry/schema tests. |
| `old/latex-parser/src/chunking/tokenizer.py` | Minimal tokenizer protocol and offline tokenizer idea | Use the frozen generator tokenizer and expose encode/decode/span operations, not only token counts. |

### Do not reuse directly

- `old/src/retrieval/benchmarks.py` is a 654-line collection of unrelated
  benchmark adapters and QASPER-specific behavior. Write one strict
  BrowseComp-Plus loader.
- `old/latex-parser/src/chunking/chunker.py` is section/LaTeX-aware, has no
  fixed overlap, and excludes some rendered headers from its budget. It does
  not meet the fixed 512-token/64-token-overlap web-corpus contract.
- Old report schemas contain one aggregate JSON object and an order-dependent
  list of timings rather than query-keyed, resumable records.
- Old hard-coded endpoints, QASPER two-stage branches, benchmark sweep shell
  script, and scientific paper comparisons stay in `old/`.

### Missing active capabilities

There is currently no active BrowseComp-Plus loader, fixed-token web chunker,
persistent chunk index, frozen-ranking artifact, token-budget packer, chat
generation client, vLLM telemetry sampler, answer/citation evaluator, run
manifest, resumable row writer, single-pass workflow, agent workflow, or
long-form analysis code.

## 3. Target files and modules

Create the active package without importing `old/`:

```text
pyproject.toml
configs/
  single_pass.toml
  iterative.toml                 # Week 3
  multi_agent.toml               # Week 4
prompts/
  single_pass.txt
  iterative_plan.txt             # Week 3
  iterative_answer.txt           # Week 3
  coordinator.txt                # Week 4
  worker.txt                     # Week 4
src/rag_system/
  __init__.py
  contracts.py
  datasets/
    browsecomp_plus.py
  chunking/
    fixed_tokens.py
  retrieval/
    base.py
    bm25.py
    qdrant.py                    # port now only if needed; not Week 1 critical path
    fusion.py                    # later sensitivity work
    frozen.py
  context/
    packer.py
  generation/
    vllm_chat.py
  telemetry/
    prometheus.py
    vllm_metrics.py
  evaluation/
    context_metrics.py
    answer_metrics.py
    citation_metrics.py
  artifacts/
    io.py
    manifest.py
    schemas.py
  workflows/
    single_pass.py
    iterative.py                 # Week 3
    multi_agent.py               # Week 4
  analysis/
    summarize.py
    plots.py
scripts/
  prepare_browsecomp_plus.py
  build_chunk_index.py
  freeze_rankings.py
  inspect_vllm_metrics.py
  run_single_pass.py
  validate_run.py
  plot_runs.py
  run_iterative.py               # Week 3
  run_multi_agent.py             # Week 4
  replay_concurrency.py          # Week 4
tests/
  test_browsecomp_plus.py
  test_fixed_tokens.py
  test_bm25.py
  test_frozen_rankings.py
  test_context_packer.py
  test_vllm_chat.py
  test_vllm_metrics.py
  test_context_metrics.py
  test_answer_metrics.py
  test_citation_metrics.py
  test_artifacts.py
  test_single_pass.py
  integration/
    test_browsecomp_smoke.py
    test_vllm_smoke.py
```

Keep optional integrations out of the core install. Suggested extras are
`dev` (pytest), `eval` (datasets, transformers, matplotlib, numpy/pandas), and
`dense` (only what a future Qdrant/dense run needs). `bm25s` remains a core
dependency for the initial retriever.

### Public classes and functions to add

| Module | Public interface | Responsibility |
| --- | --- | --- |
| `datasets/browsecomp_plus.py` | `BrowseCompPlusLoader.load_queries()`, `BrowseCompPlusLoader.iter_corpus()`, `validate_benchmark()`, `make_hash_split()` | Pin/decrypt/load the two official datasets, normalize IDs, validate labels, and create the stable dev/held-out split. |
| `chunking/fixed_tokens.py` | `FixedTokenChunker.iter_chunks()` | Stream exact 512/64 generator-token windows with stable IDs and parent spans. |
| `retrieval/base.py` | `Retriever` protocol | Define the query-to-`RetrievalTrace` boundary. |
| `retrieval/bm25.py` | `Bm25ChunkIndex.build()`, `.load()`, `.search()` | Persist and query the fixed Week 1 BM25 chunk index. |
| `retrieval/qdrant.py` | `VllmEmbeddingClient`, `QdrantChunkIndex` | Optional port for later sensitivity work; not on the Week 1 critical path. |
| `retrieval/frozen.py` | `freeze_rankings()`, `load_frozen_ranking()`, `collapse_to_documents()` | Create/hash immutable top-N artifacts and parent-document TREC rankings. |
| `context/packer.py` | `GreedyContextPacker.pack()`, `render_source_block()` | Enforce rendered-token budgets and preserve selected IDs/source maps. |
| `generation/vllm_chat.py` | `VllmChatClient.generate()`, `iter_sse_events()` | Execute and timestamp one OpenAI-compatible streamed chat request. |
| `telemetry/prometheus.py` | `parse_prometheus_text()`, `MetricFamily` | Parse metric help/type/sample data without assuming names. |
| `telemetry/vllm_metrics.py` | `VllmMetricCatalog.discover()`, `VllmMetricSampler.start()`, `.stop()` | Map available vLLM metrics and collect raw time-series samples. |
| `evaluation/context_metrics.py` | `evaluate_context()` | Compute evidence/gold coverage, complete-set coverage, diversity, and redundancy. |
| `evaluation/answer_metrics.py` | `parse_answer()`, `normalize_answer()`, `evaluate_answer()` | Compute deterministic answer/refusal metrics without an LLM dependency. |
| `evaluation/citation_metrics.py` | `evaluate_citations()` | Resolve source labels and compute deterministic citation diagnostics. |
| `artifacts/io.py` | `JsonlArtifactWriter.append_once()`, `read_jsonl()` | Atomic, resumable writes keyed by the experiment key. |
| `artifacts/manifest.py` | `build_run_manifest()`, `hash_file()`, `hash_tree()` | Capture revisions/configuration and content-address every dependency. |
| `artifacts/schemas.py` | `validate_workflow_row()`, `validate_run()` | Enforce schema version, units, cardinality, joins, hashes, and invariants. |
| `workflows/single_pass.py` | `SinglePassWorkflow.run()` | Join a frozen ranking, packer, chat client, sampler, evaluators, and writers. |
| `workflows/iterative.py` | `IterativeResearchWorkflow.run()` | Week 3 bounded sequential state machine and aggregate accounting. |
| `workflows/multi_agent.py` | `ParallelResearchWorkflow.run()` | Week 4 coordinator/workers and concurrent request accounting. |
| `analysis/summarize.py` | `summarize_run()`, `paired_bootstrap()` | Produce query-paired estimates, confidence intervals, and comparisons. |
| `analysis/plots.py` | `plot_single_pass()`, `plot_workflows()`, `plot_concurrency()` | Generate all figures from validated long-form artifacts. |

## 4. Core contracts and data flow

### Immutable records

Define these in `contracts.py` as frozen dataclasses. All IDs are strings.

- `BenchmarkQuery(query_id, question, reference_answer,
  evidence_document_ids, gold_document_ids)`
- `CorpusDocument(document_id, text, url)`
- `Chunk(chunk_id, document_id, text, token_start, token_end, token_count,
  chunk_index)`
- `RankedChunk(query_id, rank, chunk_id, document_id, score)`
- `RetrievalTrace(query_id, ranking_id, hits, started_utc_ns, latency_ms)`
- `PackedContext(context_source, requested_tokens, realized_tokens,
  payload_tokens, selected_chunks, rendered_text, redundancy)`
- `ChatRequest(request_id, messages, model, max_output_tokens, seed)`
- `StreamEvent(offset_ms, text_delta, token_texts, token_ids)`
- `GenerationTrace(request_id, timestamps, stream_events, usage,
  finish_reason, answer, error)`
- `MetricSample(offset_ms, metric_name, labels, value)`
- `WorkflowResult(...)`, serialized using the schema below.

### Protocols

```python
class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, token_ids: Sequence[int]) -> str: ...
    def count(self, text: str) -> int: ...

class Retriever(Protocol):
    def search(self, query_id: str, question: str, top_n: int) -> RetrievalTrace: ...

class ContextPacker(Protocol):
    def pack(
        self,
        query: BenchmarkQuery,
        ranked_chunks: Sequence[RankedChunk],
        budget_tokens: int,
        context_source: str,
    ) -> PackedContext: ...

class ChatClient(Protocol):
    def generate(self, request: ChatRequest) -> GenerationTrace: ...
```

The production tokenizer and vLLM server must use the same pinned tokenizer
revision. The client applies the real chat template locally to calculate a
preflight token count, then compares it with the server-reported prompt usage.

### Preparation flow

```text
pinned obfuscated query dataset + local decrypt step
    -> strict query/evidence/gold validation
    -> hash-based 100/730 split artifact

pinned 100,195-document corpus
    -> generator-tokenizer 512-token chunks / 64-token overlap
    -> stable chunk IDs and parent token spans
    -> persistent chunk artifact
    -> persistent BM25 index + index manifest

development query
    -> one top-1000 chunk retrieval
    -> frozen ranking row + ranking hash
    -> parent-collapsed TREC ranking for official retrieval diagnostics
```

The nested `gold_docs`, `evidence_docs`, and `negative_docs` records contain
document copies. Extract labels from their IDs, but use the separately pinned
100,195-document corpus as the only retrieval/context text source. This avoids
per-query corpus construction and ensures that retrieved and oracle arms render
the same canonical document text.

Stable chunk IDs should be a readable prefix plus a digest of
`corpus_revision || document_id || token_start || token_end`. Rebuilding the
same corpus/tokenizer/chunk config must produce byte-identical chunk metadata.

### Single-pass execution flow

```text
query + frozen retrieved ranking OR evidence-document-oracle ranking
    -> greedy prefix pack (2K/4K/8K/16K/32K rendered context tokens)
    -> fixed prompt with short source labels mapped to document IDs
    -> preflight total-length check
    -> start local metrics sampling
    -> one streamed vLLM chat request
    -> stop sampling
    -> deterministic answer/context/citation evaluation
    -> append one workflow JSONL row and linked call/metrics trace rows
```

Retrieval happens only while creating the frozen ranking. Budget arms never
call the retriever. The oracle ranking is also frozen once per query.

The packer renders each candidate as a complete source block including its
label/header. `realized_context_tokens` counts the exact rendered blocks, not
only raw chunk payload. It appends candidates in rank order and stops at the
first block that would exceed the budget; it never skips to a later shorter
chunk and never truncates a chunk. Therefore smaller-budget selections are
prefixes of larger-budget selections for the same ranking.

Use `top_n=1000` for the frozen chunk ranking. This comfortably fills the 32K
arm, permits parent-collapsed official Recall@1000 diagnostics, and remains a
small artifact (830,000 hit records at full scale).

### Context and answer metric definitions

For selected parent-document set `S`, evidence set `E`, and gold set `G`:

- evidence recall = `|S intersect E| / |E|`;
- gold recall = `|S intersect G| / |G|`;
- complete evidence coverage = `1[E subset S]`;
- unique source count = `|S|`;
- selected evidence/gold counts are the corresponding intersections;
- redundant-context fraction = duplicate parent token-span coverage divided by
  selected raw payload tokens. Source headers are excluded from this ratio but
  included in realized context tokens.

Answer parsing uses a fixed output contract such as:

```text
FINAL: <short answer>
SOURCES: [S1, S4]
```

Evaluate the parsed `FINAL` text using Unicode normalization, case folding,
punctuation/article/whitespace normalization, normalized exact match, token F1,
and token-boundary answer containment. Save parse success and the raw output.
BrowseComp-Plus is answerable, so record `refusal_detected` but set
`refusal_correct` to null unless a future benchmark supplies an explicit
unanswerable label.

Map source labels back to selected document IDs. Report invalid citation count,
evidence/gold citation precision against cited IDs, recall against all labeled
IDs, and recall against the labeled IDs actually present in context. These are
document-citation diagnostics, not claim-entailment scores.

## 5. Frozen Week 1 configuration and model gate

Initial configuration:

- benchmark: BrowseComp-Plus only;
- chunking: 512 generator-token chunks, 64-token overlap;
- retriever: local BM25, `k1=1.2`, `b=0.75`, Lucene method;
- frozen candidate count: 1000 chunks;
- context budgets: 2,048; 4,096; 8,192; 16,384; 32,768 rendered tokens;
- context sources: `retrieved`, `oracle` where oracle strategy is
  `evidence_document_oracle`;
- temperature: 0;
- maximum output tokens: 4,096 for the initial Qwen3.5 thinking-mode smoke;
  freeze a lower cap only if the smoke shows it preserves complete final
  answers and reasoning traces;
- seed: fixed when supported and recorded even when ignored;
- prefix caching: disabled;
- one request at a time for Week 1 and Week 2 quality runs;
- prompt, chunk, retrieval, and generation settings fixed after the Week 1
  acceptance review.

The project generator decision is `Qwen/Qwen3.5-27B`, pinned initially at
revision `fc05daec18b0a78c049392ed2e771dde82bdf654`. It is preferred over
Qwen3.6-27B for the primary study because Qwen3.5 has the more established
deep-search evaluation profile. This is a design choice, not evidence that the
new workflow will reproduce the model card's BrowseComp results. Use
thinking mode initially so later agentic comparisons do not silently change the
model's reasoning mode; preserve reasoning content separately from the final
answer.

The fixed model still must pass these local deployment gates:

1. supported by the installed vLLM and local checkpoint revision;
2. exact tokenizer/chat-template revision available offline;
3. 32K rendered context plus overhead/output passes server preflight;
4. no OOM at the 32K arm on the A100;
5. stable parseable output and nonzero oracle diagnostic signal;
6. repeat runs at temperature 0 produce identical parsed answers;
7. BF16 weights in language-only mode leave enough hybrid cache capacity for at
   least four 32K-equivalent live sequences, or a documented smaller-context
   concurrency stress design;
8. `/metrics` exposes KV usage, running/waiting requests, token counters, and
   preemption count on the frozen vLLM version.

Qwen3.5-27B interleaves Gated DeltaNet state with full-attention KV cache, so a
single standard-transformer bytes-per-token formula is incomplete. Calculate
the full-attention portion as a sanity check:

```text
2 (K and V) x 16 attention layers x 4 KV heads x 256 head dimension x KV dtype bytes
```

Also record the Gated DeltaNet/Mamba cache configuration, vLLM's available cache
blocks/capacity, actual cache dtypes, and observed utilization. Use the estimate
for sanity checking, not as a replacement for measured utilization.

## 6. Result and manifest schemas

Use a user-supplied or gitignored `RAG_ARTIFACT_ROOT`. One run directory is:

```text
<artifact-root>/runs/<run-id>/
  manifest.json
  workflows.jsonl
  calls.jsonl
  retrievals.jsonl
  server_samples.jsonl
  metric_catalog.prom
  validation.json
  plots/
```

Prepared data is similarly versioned under `<artifact-root>/datasets/`,
`chunks/`, `indexes/`, `splits/`, and `rankings/`. Never place decrypted
questions, answers, generated answers, or benchmark document text in Git.

### `workflows.jsonl`

There is exactly one row per
`run_id x query_id x workflow x context_source x requested_context_tokens` for
Stage 1, including failures. Use nested objects but keep units in names.

```json
{
  "schema_version": "1.0",
  "run_id": "...",
  "workflow_id": "...",
  "workflow": "single_pass",
  "query_id": "...",
  "dataset": {"name": "browsecomp_plus", "revision": "...", "split": "dev"},
  "input": {
    "question": "...",
    "reference_answer": "...",
    "evidence_document_ids": ["..."],
    "gold_document_ids": ["..."]
  },
  "retrieval": {
    "ranking_id": "...",
    "ranking_sha256": "...",
    "top_n": 1000,
    "latency_ms": 0.0,
    "selected_ranks": [1],
    "selected_chunk_ids": ["..."],
    "selected_document_ids": ["..."]
  },
  "context": {
    "source": "retrieved",
    "oracle_strategy": null,
    "requested_tokens": 2048,
    "realized_tokens": 2012,
    "payload_tokens": 1988,
    "selected_chunk_count": 4,
    "selected_unique_document_count": 3,
    "redundant_fraction": 0.04
  },
  "quality": {
    "evidence_document_count": 1,
    "gold_document_count": 1,
    "evidence_recall": 0.2,
    "gold_recall": 0.5,
    "complete_evidence_coverage": false,
    "normalized_exact_match": 0,
    "token_f1": 0.5,
    "answer_containment": 1,
    "refusal_detected": false,
    "refusal_correct": null,
    "citation_evidence_precision": 1.0,
    "citation_evidence_recall": 0.2,
    "citation_selected_evidence_recall": 1.0,
    "invalid_citation_count": 0
  },
  "tokens": {
    "client_prompt_tokens": 2200,
    "server_prompt_tokens": 2200,
    "completion_tokens": 38,
    "total_tokens": 2238
  },
  "timing_ms": {
    "retrieval": 15.0,
    "packing": 1.0,
    "queue": null,
    "prefill": null,
    "ttft": 100.0,
    "mean_inter_event": 20.0,
    "p95_inter_event": 25.0,
    "client_decode_span": 800.0,
    "server_decode": null,
    "end_to_end": 920.0
  },
  "server": {
    "metric_provenance": "prometheus_exclusive_server_delta",
    "peak_kv_cache_usage_fraction": 0.3,
    "peak_running_requests": 1,
    "peak_waiting_requests": 0,
    "preemptions_delta": 0,
    "prompt_tokens_per_second_peak": 0.0,
    "generation_tokens_per_second_peak": 0.0
  },
  "generation": {
    "call_id": "...",
    "finish_reason": "stop",
    "generated_answer": "...",
    "parsed_answer": "...",
    "cited_document_ids": ["..."]
  },
  "error": null
}
```

When a value is unavailable, use null and add a reason to the relevant
provenance/error object. Do not use zero for missing timing or metrics.

### `calls.jsonl`

One row per model-call attempt, keyed by `call_id` and `workflow_id`:

- request start, HTTP response, first-content, last-content, and completion
  wall-clock timestamps in UTC nanoseconds;
- matching monotonic offsets/durations;
- raw SSE event sequence with text delta, token/logprob records when returned,
  event token count, and arrival offset;
- TTFT, mean/p95 inter-event latency, and inter-token latency only if timing
  granularity is verified;
- messages or prompt hash, model, seed, sampling settings, usage, finish reason,
  raw generated text, HTTP status, server request ID, attempt number, and typed
  error.

Keep all attempts. The workflow row points to the accepted attempt. Primary
latency analysis uses successful first attempts; retries and failures are
reported separately.

### `retrievals.jsonl` and `server_samples.jsonl`

Retrieval rows hold complete top-1000 chunk hits, scores, ranks, parent IDs,
latency, retriever config hash, and artifact hash. Budget rows refer to this
immutable artifact.

Server samples are long-form Prometheus observations keyed by run, call or
workflow when attribution is defensible, metric name, labels, monotonic offset,
and value. Preserve the raw `/metrics` catalog and its hash. Histogram bucket,
sum, and count deltas remain aggregates.

### `manifest.json`

Required sections:

- identity: schema version, run ID, created UTC, purpose, stage;
- Git: commit, dirty flag, and diff hash for diagnostics (measurement runs
  should be clean);
- dataset: both Hugging Face repository names/revisions, split seed/IDs/hash,
  validation counts, local artifact hashes, and decryption method version;
- models: generator repository/revision, served name, tokenizer repository and
  revision, model dtype, quantization, configured maximum length, any RoPE
  scaling, theoretical KV bytes/token; embedding model fields are null for BM25;
- software: Python, OS, package lock/hash, vLLM version/commit, CUDA, driver;
- hardware: GPU name/UUID/total memory, CPU, RAM, and exclusivity check;
- vLLM: exact sanitized launch argv, host/port, engine version, prefix-cache
  setting, KV dtype, block size/count, GPU memory utilization, maximum sequences,
  scheduler settings, and metrics catalog hash;
- experiment: workflow config, budgets, top N, query order hash, random seeds,
  output-token limits, retry policy, sampling interval;
- preparation: chunk config/hash/count, index config/hash, ranking config/hash;
- prompt: file path, byte SHA-256, rendered chat-template hash;
- commands: exact preparation, server, run, validation, and plotting commands.

Pin revisions at runtime. As of this plan's audit, the official Hugging Face
APIs reported query revision `144cff8e35b5eaef7e526346aa60774a9deb941f`
and corpus revision `b27b02bc3e45511b8b82a13e6f90ce761df726f6`;
do not silently follow `main` if those change.

## 7. Day-by-day Week 1 plan

### Day 1: active scaffold, data contract, and generator freeze

- Commit/archive the current `old/` move separately so later manifests can use
  a clean Git revision.
- Replace the stale root package/test configuration with the active package
  scaffold, dependency groups, artifact ignores, and thin CLI conventions.
- Implement strict BrowseComp-Plus query/corpus records and local decrypt/load
  paths pinned to exact revisions. Never print decrypted examples in logs.
- Validate 830 unique queries, 100,195 unique corpus documents for the audited
  revision, nonempty answer/evidence/gold fields, and membership of every
  labeled ID in the corpus. Check and report whether gold is a subset of
  evidence; do not assume it.
- Create and freeze the hash-based 100-query dev/730-query held-out split.
- Validate the pinned Qwen3.5-27B model with the required recent vLLM build and
  FlashInfer on `solab-g3`; inspect `/metrics`, calculate hybrid cache capacity,
  and freeze the model/tokenizer/server launch command by end of day.

Deliverable: validated dataset/split manifest, selected generator decision
record, exact vLLM launch command, raw metric catalog, and minimal contract
tests.

### Day 2: chunking, index, and frozen rankings

- Implement generator-token 512/64 fixed-window chunking with token offsets,
  stable IDs, streaming corpus processing, and pathological-document guards.
- Build a small-corpus fixture and a 1% corpus resource pilot before the full
  build. Record chunk count, size distribution, build time, RAM, and disk.
- Port BM25 behavior into a persistent chunk index with index metadata and
  deterministic tie-breaking. Do not materialize all full documents twice.
- Build the full local index and freeze top-1000 retrieved rankings for all dev
  queries. Freeze evidence-only oracle rankings separately.
- Collapse chunk rankings to parents and create official-format retrieval runs
  for a sanity comparison.

Deliverable: hashed chunk/index/ranking artifacts and deterministic rebuild
tests.

### Day 3: packer, prompt, quality metrics, and artifact writer

- Implement exact rendered-token greedy prefix packing and source-label maps.
- Implement context coverage, redundancy, normalized answer, refusal detection,
  output parsing, and deterministic citation metrics.
- Freeze the single-pass prompt and hash its bytes/rendered chat template.
- Implement versioned, atomic/resumable workflow/call/retrieval writers and the
  run manifest. Resume logic skips only an already-valid unique experiment key.
- Add a validator for row cardinality, artifact hashes, token bounds, ID
  membership, selected-prefix invariants, and null/error semantics.

Deliverable: unit-tested pack/evaluate/write path using only in-memory fixtures.

### Day 4: streamed chat and vLLM telemetry

- Implement `VllmChatClient` using `urllib` and an incremental SSE parser that
  handles fragmented lines, multiple data fields, usage chunks, `[DONE]`, HTTP
  errors, malformed JSON, disconnects, and timeouts.
- Use wall-clock UTC nanoseconds for event timestamps and
  `perf_counter_ns()` for durations. Record raw event granularity.
- Implement `/metrics` discovery/parser and a background sampler. Map semantic
  concepts to the names actually present in the frozen vLLM version.
- Enable and verify per-request server metrics if supported. Otherwise document
  which queue/prefill/decode fields remain null and why.
- Verify prefix caching is disabled from both launch argv and cache-config
  metrics. Run dedicated-server 2K and 32K integration checks.

Deliverable: fake-SSE unit tests, real-server smoke trace, metric mapping, and
prompt-token agreement report.

### Day 5: end-to-end smoke and acceptance review

- Choose 20-30 dev IDs before looking at outputs.
- Run retrieved/oracle x 2K/8K/32K (120-180 workflow rows), randomized within
  query, after excluded warmups.
- Validate and regenerate failures only through the recorded retry mechanism.
- Produce the six required plots plus error/cardinality and realized-budget
  diagnostics.
- Manually inspect at least five rows per budget/source combination for prompt
  formatting, source mapping, answer parsing, and trace plausibility.
- Calculate the Week 2-4 GPU-hour forecast and freeze Week 1 configuration.

Deliverable: complete smoke run directory, validation report, plots, freeze
record, and go/no-go decision.

## 8. Week 1 acceptance tests

The baseline is complete only when all applicable checks pass.

### Dataset and split

- Exactly 830 unique query IDs and 100,195 unique corpus document IDs for the
  pinned audited revisions, or a documented upstream revision change.
- Every evidence and gold ID exists in the corpus; duplicates are rejected or
  canonically deduplicated with counts.
- The same seed/revision always yields the same 100 dev and 730 held-out IDs;
  split IDs are disjoint and exhaustive.
- No decrypted question, answer, document, generated answer, JSONL result, or
  trace is tracked by Git.

### Chunk/index/ranking

- Every non-tail chunk has at most 512 tokens and adjacent windows overlap by
  exactly 64 source token IDs; token spans remain within the parent document.
- Chunk IDs and serialized metadata are byte-stable on fixture rebuild.
- BM25 fixture rankings match expected order and deterministic ID tie-breaks.
- Index metadata rejects a mismatched corpus, tokenizer, or chunk config.
- Every dev query has exactly one retrieved and one oracle ranking artifact;
  every ranked chunk exists and has the expected parent.
- Oracle rankings contain only evidence-parent chunks and do not use the
  reference answer.
- All budget arms for a query/source refer to the same ranking hash, and no
  retrieval code path runs during a budget arm.

### Packing and evaluation

- Rendered context never exceeds its requested budget, including headers.
- Selected rankings are order-preserving prefixes; no partial or skipped chunk.
- Selected chunk/document IDs and counts recompute exactly from rendered source
  maps.
- Evidence/gold recall, complete coverage, redundancy, answer normalization,
  containment, F1, citations, invalid citations, and parse failures have edge
  case tests (including empty sets/output and duplicate chunks).
- Client preflight proves the 32K arm plus total overhead/output fits the frozen
  server limit.

### Streaming and telemetry

- Fake-server tests cover fragmented SSE frames, Unicode splits, multi-token
  events, usage, stop, error, timeout, and disconnect behavior.
- Request, first-content, event, and completion timestamps are monotonic; TTFT,
  event gaps, decode span, and E2E recompute from raw timestamps.
- Client and server prompt/completion token counts agree, or the difference is
  understood, bounded, and the server count is designated authoritative.
- The saved metric catalog and launch configuration verify prefix caching is
  disabled.
- KV usage, running/waiting request gauges, token counters, and preemption
  counters are available. If they are not, change/freeze the vLLM version before
  measurement; this is a hard gate for the core systems hypothesis.
- Missing per-request queue/prefill/decode values are null with provenance, not
  copied from server-wide quantiles.

### End-to-end smoke

- Expected row count is exactly `queries x 2 sources x 3 budgets`, including
  typed error rows; unique keys have no duplicates.
- At least 95% of successful 32K rows realize 95% of their requested context,
  or each underfill is explained by exhausted candidates.
- At least 95% of calls succeed on first attempt; all attempts/errors remain
  visible.
- Repeating five inputs yields identical parsed answers and selected IDs.
- All required plots can be regenerated from validated artifacts alone.
- Manual inspection finds no leaked reference answer in prompts or retrieval.

## 9. Weeks 2-4 experiment schedule

### Week 2: full single-pass factorial on development data

**Days 6-7:** Run all 100 dev queries under retrieved/oracle x
2K/4K/8K/16K/32K. Use one quality pass with query order randomized and arm
order randomized within query. Use 10 excluded warmups after every server
restart. Run three additional timing repeats on a preselected 30-query subset
to estimate systems variance without tripling all generation cost.

**Day 8:** Validate rows and run parent-collapsed official retrieval metrics.
Analyze retrieval-to-context loss: top-1000 document recall, context recall at
each token budget, underfill, document duplication, and oracle localization
loss.

**Day 9:** Analyze retrieved versus oracle quality and cost. Select and freeze
one single-pass comparator for later stages using a predeclared rule: highest
mean dev answer containment, then token F1, then lower median E2E latency; also
retain 2K and 32K as lower/upper systems anchors.

**Day 10:** Freeze Stage 2 call, round, retrieval, context, and output budgets.
Reranking is allowed only if all Stage 1 work is complete and it can be a
strictly optional side experiment: rerank a frozen pool and compare a small
context to larger unreranked context while including reranker latency. It must
not change the main comparator or delay Week 3.

### Week 3: bounded iterative single agent

Implement a simple state machine, not a general agent framework:

```text
question
  -> initial BM25 search
  -> planning call emits strict JSON {follow_up_query, stop, rationale}
  -> deduplicated BM25 follow-up search
  -> repeat up to 4 retrieval rounds
  -> final fixed-budget synthesis call
```

Use the same corpus, chunk/index artifact, generator, prompt family, and source
format. Set `max_rounds=4`, `top_chunks_per_round=5`, a fixed observation budget
per round (start at 4K), a fixed 16K deduplicated final evidence budget, and
small planning outputs (128 tokens). Stop only on valid model `stop=true`, the
round limit, or a typed failure. Preserve every query, ranking, model call,
state transition, and deduplication decision.

**Days 11-12:** Build/test the state machine, strict JSON repair policy (at most
one recorded retry), deduplication, aggregate accounting, and resume behavior.

**Day 13:** Run a 20-query smoke. Verify cumulative tokens equal the sum of all
calls and peak live tokens never incorrectly equal that sum for sequential
calls.

**Days 14-15:** Run all 100 dev queries. Compare with the frozen single-pass 16K
and 32K arms. Report both unconstrained quality and cost-matched views: final
evidence budget matched to 16K, and post-hoc strata by cumulative input tokens.
Freeze the iterative protocol before Week 4.

Do not choose the number of rounds per query using gold labels. Record search
diversity, repeated queries, duplicated chunks/documents, evidence/gold union
recall by round, calls, cumulative tokens, E2E latency, and peak server metrics.

### Week 4: bounded parallel agents, serving replay, and final evaluation

Use one controlled architecture:

```text
coordinator call -> N independent search briefs
N workers in parallel:
    one BM25 retrieval -> one evidence-summary call
coordinator final synthesis over worker summaries and cited source map
```

Test `N in {1, 2, 4}` on dev. Each worker gets the same 4K retrieved-context and
128-token summary limit; the coordinator final answer gets a fixed summary
budget and 256 output tokens. Agent count, not per-agent budget, changes. An
8-worker condition is a systems stress arm only if capacity calculations and a
small safety smoke pass.

**Day 16:** Implement coordinator/worker contracts, `asyncio` scheduling,
failure propagation, deterministic source union, and workflow-level cumulative
and peak-live accounting. Complete fixture tests and a 20-query smoke.

**Day 17:** Run 100 dev queries for N=1/2/4. Analyze evidence union recall,
query/result diversity, duplicated retrieval, answer quality, calls, cumulative
tokens, wall time, and peak concurrency; freeze one multi-agent condition.

**Day 18:** Replay the exact saved worker requests at concurrency 1/2/4 and, if
safe, 8. Use at least three randomized repeats per concurrency, excluded warmup,
fixed server state, and 100 ms metric sampling. Measure maximum simultaneous
requests, aggregate live prompt+generated tokens over time, KV utilization,
running/waiting requests, throughput, queueing, preemptions/recompute, TTFT, and
E2E. Replay keeps prompt content and token counts identical across concurrency.

**Day 19:** Freeze all decisions, then run the held-out evaluation. Minimum
conditions are the chosen retrieved single-pass arm, the 32K oracle diagnostic,
the frozen iterative agent, and the frozen multi-agent system. Use all 730 only
if the Week 1 forecast shows that it will complete by Day 20; otherwise use the
already-precommitted 200-query held-out hash sample and reserve full-730
execution as the first follow-up. Never choose the sample after seeing outcomes.

**Day 20:** Validate the held-out run, produce final plots/statistics, run
official judge-based scoring only if local time/resources remain, write
limitations and appendix links to the old scientific-retrieval work, and
package exact manifests/commands. The four-week deliverable favors a complete,
auditable 200-query held-out result over a partial full-corpus run.

## 10. Plots and statistical analyses

### Week 1 required plots

1. evidence-document recall versus realized context tokens;
2. gold-document recall versus realized context tokens;
3. answer token F1 and containment/EM versus realized context tokens;
4. client TTFT versus total server prompt tokens;
5. server prefill time versus total prompt tokens, only if per-request values
   are actually available;
6. peak KV-cache utilization versus total prompt tokens.

Add realized versus requested context, selected chunks/documents, redundancy,
latency distribution, error rate, and client/server token-count agreement as
pipeline diagnostics.

### Final plots

- paired retrieved/oracle budget curves for evidence recall, gold recall,
  complete evidence coverage, F1, containment, TTFT, E2E, and KV use;
- evidence/gold recall and answer quality by iterative round;
- answer quality versus cumulative input/output tokens and E2E latency for all
  workflows (Pareto plot);
- cumulative workflow tokens versus peak concurrently live tokens;
- multi-agent count versus evidence union recall, unique sources, duplicate
  retrieval fraction, answer quality, and calls;
- replay concurrency versus peak KV utilization, running/waiting requests,
  preemptions, throughput, TTFT, and E2E;
- timeline for representative concurrency runs showing live requests, estimated
  live tokens, KV usage, waiting requests, and preemption events;
- failure/retry/truncation rates by workflow.

### Statistical plan

- The query is the paired sampling unit. Never treat arms or calls from one
  query as independent observations.
- Report means for benchmark metrics and medians/p95 for skewed latency/cost,
  each with query-clustered bootstrap 95% confidence intervals (10,000 resamples
  for final results; 2,000 is sufficient for smoke/dev iteration).
- Use paired bootstrap confidence intervals for budget, oracle/retrieved, and
  workflow differences. For binary answer outcomes, also report McNemar's test;
  for continuous F1/latency, use paired permutation or Wilcoxon as a sensitivity
  check rather than relying on normality.
- Correct the small predeclared family of primary pairwise comparisons with
  Holm's method. Label all other slicing exploratory.
- Fit simple descriptive regressions with query-clustered errors: TTFT/prefill
  versus prompt tokens, answer success versus evidence/gold coverage and
  context source, and queue/TTFT versus peak live tokens/concurrency. Plot raw
  data and avoid causal language.
- Report macro average recall and complete-set coverage together. The latter is
  important because the average query has multiple evidence documents.
- Quality runs need one deterministic pass. Systems replay gets at least three
  repeats; report run-order and restart effects.
- Define primary Week 4 comparisons before held-out execution: selected
  single-pass versus iterative, iterative versus N=4 multi-agent, and retrieved
  versus oracle at the chosen single-pass budget.

## 11. Major risks and mitigations

| Risk | Consequence | Mitigation / gate |
| --- | --- | --- |
| Root migration is uncommitted and config is stale | Manifests cannot identify reproducible code; imports/tests are misleading | Commit archive move separately; scaffold active package before implementation. Never import `old/`. |
| Decrypted benchmark leakage | Violates benchmark canary and contaminates Git/history | Strict artifact root, expanded ignores, pre-commit/status check, never log examples, only IDs/hashes in tracked docs. |
| 32K arm exceeds total model length | Rejections or silent truncation invalidate budget comparison | Preflight exact rendered chat tokens plus output; require >=40K configured total or lower no arm (do not silently clamp). |
| Temperature 0 harms a Qwen3 candidate | Low/repetitive answers confound workflow comparison | Two-model Week 1 gate; freeze one that is stable under the mandated decoding settings. |
| Full chunk corpus is too large for in-memory legacy BM25 | Week 1 index build OOM or takes days | Stream chunking, persistent index, 1% resource pilot, record RAM/disk/time, use official prebuilt document BM25 only as an explicit MVP fallback. |
| Generator and chunk tokenizers differ | Budgets and offsets are wrong | Use the frozen generator tokenizer for chunk and pack accounting; hash tokenizer revision. |
| Oracle pages exceed budget | Oracle underperforms for localization reasons | Define it as an evidence-document oracle, freeze question-only within-evidence ranking, report evidence coverage and caveat. |
| Incomplete or semantic labels | Useful retrieved text counted irrelevant | Use label metrics as lower-bound diagnostics; retain answer metrics/citations; manually inspect disagreements without changing labels. |
| Deterministic QA metrics miss paraphrases | Understates quality | Keep them required and reproducible; add official judge later as a separately labeled result. |
| Stream chunk is not one token | Inter-token latency claim is false | Store raw SSE events/token records; expose timing granularity; report inter-event latency unless verified. |
| Prometheus metrics change by vLLM version | Missing/misnamed KV and timing fields | Discover catalog at startup, semantic mapping with types, pin vLLM; hard-gate core gauges/counters and null unavailable request metrics. |
| Server-wide histograms are attributed to a request | Misleading queue/prefill/decode analysis | Use per-request metrics if supported; otherwise aggregate deltas only on exclusive server and state provenance. |
| Metric sampling misses short KV peaks | Peak pressure is underestimated | Sample at 100 ms for concurrency, retain request timeline/estimated live tokens, compare to block/capacity calculation. |
| Prefix cache or warm state biases long shared prompts | Prefill/TTFT no longer reflect raw workload | Disable and verify prefix caching; excluded warmups; fixed restart policy and randomized order. |
| Retries censor slow/failing requests | Latency looks better than reality | Keep attempt rows, report failure/retry rates, use successful first attempts for primary latency with failure sensitivity. |
| More agents also means more calls/tokens | Cannot attribute gains to parallelism | Report cumulative budget, use N=1 architecture control, and use identical-request concurrency replay for systems effects. |
| No preemption occurs | Hypothesis about preemption cannot be evaluated | Scale replay concurrency/context safely until waiting/KV pressure or precommitted max; report a negative result rather than forcing OOM. |
| GPU is not exclusive | Server metrics and latency are contaminated | Check active processes before every run, dedicated host policy, abort rather than use the shared H200 service. |
| Development/test leakage | Inflated final claims | Hash-only split, protocol freeze, predeclared held-out conditions, no result-driven held-out sample changes. |
| Four-week compute overrun | Agent/final runs incomplete | Forecast from smoke, use 100-query dev and selected held-out arms, precommit 200-query fallback, keep output/round limits small. |

## 12. Minimal viable path

If integration or implementation runs late, preserve the research question by
reducing breadth, not instrumentation.

1. **BrowseComp query integration delayed:** use the official decrypt script and
   direct local JSONL schema rather than building a generic Hugging Face adapter.
   For code-path tests only, build a 20-query local corpus from labeled positive
   and released negative documents. Do not report its retrieval scores as the
   fixed-corpus benchmark.
2. **Full chunk index delayed:** use the official/prebuilt document BM25 ranking,
   freeze top documents, then deterministically chunk their text and retain
   ranked-document order. Label this `document_retrieval_then_chunking`; it is
   not interchangeable with the planned chunk BM25 baseline.
3. **Generator quality too low:** keep the 32K evidence-document oracle arm and
   run WixQA only to demonstrate that loading, packing, generation, timing, and
   evaluation work. Report BrowseComp quality as a floor; do not switch the
   primary benchmark silently.
4. **Per-request vLLM metrics unavailable:** pin/upgrade vLLM if possible. If
   not, retain TTFT/E2E/event traces and core server KV/running/waiting/preemption
   metrics, leave queue/prefill/decode null, and narrow the claims.
5. **Iterative agent delayed:** implement exactly two sequential follow-up
   rounds with a strict JSON query planner and one final synthesis. This is
   enough to measure cumulative versus peak cost.
6. **Multi-agent orchestration delayed:** run N independently generated search
   briefs with one retrieval/summary call each and one final coordinator. If
   even that slips, replay fixed Stage 2 prompts concurrently to complete the
   KV/queueing experiment, clearly separating it from multi-agent quality.
7. **Compute budget exceeded:** finish all 100 dev factorial arms and use the
   precommitted 200-query held-out sample for the four selected conditions.
   Defer the 730-query confirmation; never drop instrumentation or oracle data
   to increase nominal sample size.

## 13. Explicitly not built in Week 1

- query rewriting or decomposition;
- repeated/adaptive retrieval;
- reranking or context compression models;
- search-tool or browser interfaces;
- conversation history or memory;
- iterative-agent state machines;
- subagents, coordinators, or concurrency scheduling;
- load generators, preemption stress, or prefix-cache comparisons;
- retriever, chunk-size, overlap, prompt, or generator sweeps after the freeze;
- LLM-judge scoring as a required dependency;
- QASPER, TREC RAG, continued scientific-retrieval optimization, or WixQA unless
  the explicit fallback gate fires;
- use of the shared 4x H200 Qwen3-Coder endpoint for any measurement;
- UI, dashboard, distributed task queue, generic agent framework, or production
  service hardening.

## Sources checked for this plan

- [Official BrowseComp-Plus repository](https://github.com/texttron/BrowseComp-Plus)
  for data preparation, run format, official qrels evaluation, and agent assets.
- [Official BrowseComp-Plus corpus card](https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus)
  for the `docid`/`text`/`url` corpus schema and 100,195-document count.
- [Official BrowseComp-Plus query dataset](https://huggingface.co/datasets/Tevatron/browsecomp-plus)
  for the 830-row `query_id`, `query`, `answer`, `gold_docs`, `negative_docs`,
  and `evidence_docs` schema.
- [BrowseComp-Plus paper/project](https://texttron.github.io/BrowseComp-Plus/)
  for the benchmark construction and evidence/gold definitions.
- [vLLM metrics documentation](https://docs.vllm.ai/en/latest/design/metrics/)
  for server-wide metric types and version-sensitive metric behavior.
- [vLLM per-request metrics documentation](https://docs.vllm.ai/en/latest/features/per_request_metrics/)
  for the distinction between per-request data and aggregate Prometheus
  histograms.
- [Qwen3.5-27B model card](https://huggingface.co/Qwen/Qwen3.5-27B) for the
  pinned architecture, native context, reasoning behavior, vLLM requirement,
  text-only serving flag, and reported deep-search results.
- [vLLM supported-model documentation](https://docs.vllm.ai/en/latest/models/supported_models/)
  and [FlashInfer installation documentation](https://docs.flashinfer.ai/installation.html)
  for language-only Qwen3.5 serving and explicit FlashInfer installation.
