# Qwen3-Embedding-8B retrieval baseline handoff

Status date: 2026-07-22

This is the current session handoff and takes precedence over stale model and
retriever choices in the original four-week plan. It contains no decrypted
questions, answers, or corpus text.

## Current research decision

The first end-to-end baseline will follow the BrowseComp-Plus **Standard**
scaffold as closely as practical:

- generator: `Qwen/Qwen3.6-27B` served by vLLM with FlashInfer;
- retriever: `Qwen/Qwen3-Embedding-8B` and the official precomputed FAISS
  document index;
- tool surface: `search` only, with no `get_document`;
- tool result: top 5 documents per search call;
- snippet: at most the first 512 tokens of each returned document, measured
  with the upstream `Qwen/Qwen3-0.6B` tokenizer;
- initial execution: one query at a time, with no context-budget sweep, judge,
  concurrency, or custom chunking until the Standard baseline works.

The `Qwen/Qwen3-0.6B` **weights are not required**. The upstream search tool
loads only its tokenizer to reproduce 512-token snippet truncation.

## Host topology and observed paths

`solab-p7` has no GPU and owns benchmark preparation, results, and the current
CPU FAISS evaluation. The repository and selected result root are:

```text
/mnt/nvme2/mlee/rag-system
/mnt/nvme2/mlee/rag-system/results
```

The transferred dataset cache has an additional `datasets/` level. The paths
used during troubleshooting were based on:

```text
/mnt/nvme2/mlee/.cache/huggingface/hub/datasets/
  datasets--Tevatron--browsecomp-plus/
  datasets--Tevatron--browsecomp-plus-corpus/
  datasets--Tevatron--browsecomp-plus-indexes/
```

Always locate and validate the actual snapshot before running; do not assume a
standard Hugging Face cache layout. The query snapshot must expose six
`data/*.parquet` files and the corpus snapshot seven when followed with
`find -L`.

`solab-g3` has two A100 80 GB GPUs. The PCIe A100 is used for the 8B embedding
model; the SXM4 A100 is reserved for Qwen3.6 generation. Observed paths are:

```text
/mnt/nvme3n1/mlee/BrowseComp-Plus
/mnt/nvme3n1/mlee/BrowseComp-Plus/.venv
/mnt/nvme3n1/mlee/BrowseComp-Plus/embeddings
```

The large g3 cache is configured under
`/mnt/nvme3n1/labuser/.cache`. Set `UV_CACHE_DIR` explicitly to
`/mnt/nvme3n1/labuser/.cache/uv` because uv otherwise used
`/home/labuser/.cache/uv` and exhausted that filesystem while extracting
FlashAttention. Also set temporary build directories to the large cache.

## Pinned upstream assets

| Asset | Revision used |
| --- | --- |
| `Tevatron/browsecomp-plus` | `144cff8e35b5eaef7e526346aa60774a9deb941f` |
| `Tevatron/browsecomp-plus-corpus` | `b27b02bc3e45511b8b82a13e6f90ce761df726f6` |
| `Tevatron/browsecomp-plus-indexes` | `b3f37f70c33829eb09d04784a54277a31871fd63` |
| `Qwen/Qwen3-Embedding-8B` | `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af` |
| inspected BrowseComp-Plus code | `046949032b0328319cc9a02663a759ec601d9402` |

Confirm the checked-out upstream commit and local model snapshot before a
reported run; the commit checkout was instructed but not independently
verified in this session.

## Work completed

1. The offline query and corpus snapshots were transferred to p7 and the
   official 8B embedding checkpoint to g3.
2. `scripts/prepare_browsecomp_plus.py` was used on p7 to decrypt only the
   required query fields, validate the benchmark, and create the deterministic
   development/held-out split. The intended audited counts are 830 queries,
   100,195 documents, 100 development IDs, and 730 held-out IDs.
3. Prepared queries were converted to Tevatron JSONL with `query_id` and
   `query`, then copied to g3.
4. Tevatron encoded all query vectors on g3 with the local
   Qwen3-Embedding-8B checkpoint. The frozen retrieval settings were:
   `query_max_len=512`, normalized vectors, EOS pooling, FP16, and the prefix:

   ```text
   Instruct: Given a web search query, retrieve relevant passages that answer the query
   Query:
   ```

5. `query.qwen3-8b.pkl` was copied back to p7. Tevatron performed sharded CPU
   FAISS search over the four official index shards to depth 1,000 and emitted
   `top1000.tsv` and `top1000.trec`.
6. Pyserini evaluated the TREC run against the upstream document-level evidence
   and gold qrels.

The successful encoding command did not explicitly override Tevatron's
attention backend. Verify and record whether `flash_attention_2` or SDPA was
actually active before calling the run an exact reproduction.

## Measured retrieval results

All 830 queries were evaluated.

| Metric | Published Qwen3-8B | Local result | Difference |
| --- | ---: | ---: | ---: |
| Evidence nDCG@10 | 0.2026 | 0.2075 | +0.0049 |
| Evidence Recall@5 | 0.1453 | 0.1494 | +0.0041 |
| Evidence Recall@100 | 0.4773 | 0.4866 | +0.0093 |
| Evidence Recall@1000 | 0.7672 | 0.7738 | +0.0066 |
| Gold nDCG@10 | 0.1946 | 0.1947 | +0.0001 |
| Gold Recall@5 | 0.1848 | 0.1903 | +0.0055 |
| Gold Recall@100 | 0.5575 | 0.5642 | +0.0067 |
| Gold Recall@1000 | 0.8354 | 0.8346 | -0.0008 |

This passes the retrieval reproduction gate. The small differences may come
from the Tevatron, Transformers, FAISS, attention-backend, or qrels revisions.
Do not silently tune settings to force exact equality. Record those revisions
and treat this run as the frozen local retrieval baseline.

## Artifacts that should be retained

Under the gitignored p7 result root, retain at minimum:

```text
results/
  datasets/browsecomp-plus/
  retrieval/query.qwen3-8b.pkl
  retrieval/qwen3-embedding-8b/
    top1000.tsv
    top1000.trec
    evidence-metrics.txt
    gold-metrics.txt
    metadata/
```

The metrics files, artifact/index SHA-256 files, `pip freeze`, Java version,
upstream Git commit, GPU inventory, and exact commands were requested after the
successful evaluation but were not confirmed as saved. A new session should
check and fill these gaps before moving or regenerating artifacts.

## Why record revisions, hashes, commands, and hardware?

The records serve four concrete purposes:

1. **Reproduction:** they identify the exact dataset, labels, checkpoint,
   index, software, command, and output that produced a number.
2. **Diagnosis:** if a later score changes, hashes and versions reveal whether
   the cause was the model, query vectors, qrels, FAISS, code, or experiment.
3. **Fair comparisons:** later Standard, `get_document`, chunking, context, and
   agent experiments can change one declared factor while holding the
   retriever baseline fixed.
4. **Research integrity:** they prevent accidental held-out tuning, silent
   upstream drift, dropped failures, or a leaderboard comparison made under a
   different scaffold.

This is not a request to log everything. The minimum useful record is asset and
code revisions, exact configuration/commands, software and hardware versions,
artifact hashes, raw run output, and final metrics.

## Document encoding, snippets, and qrels

These are three different layers and must not be conflated.

### 1. Offline document representation

The upstream Qwen3 embedding recipe supplies each full corpus document to the
encoder with `passage_max_len=4096`, normalized EOS pooling, and no passage
prefix. Under the normal right-truncation path, at most the leading 4,096
tokens contribute to the single stored vector for that document. The hosted
8B index contains one vector per document, not one vector per 512-token chunk.

The checked-in recipe explicitly demonstrates the 0.6B Qwen3 embedding model;
the hosted 0.6B, 4B, and 8B indexes strongly imply the same recipe, but the
8B index does not contain a complete build manifest proving every argument.
Keep that provenance limitation in the write-up.

### 2. Online Standard search result

After vector search returns a document ID, the Standard tool looks up the
original document text and returns at most its first 512 tokens. The upstream
implementation uses the `Qwen/Qwen3-0.6B` tokenizer for this truncation. It is
only a tokenizer choice: the 0.6B model does not retrieve, rerank, or generate.

The 4,096-token limit therefore controls what influenced the stored document
vector; the 512-token limit controls what text the generator sees after a hit.

### 3. Evidence and gold labels

BrowseComp-Plus qrels label **document IDs**, not embeddings, token windows, or
snippets. Evidence documents are human-verified documents needed to answer the
question. Gold documents are needed documents that also semantically contain
the final answer. Labels were assigned against the document, so they do not
guarantee that the first 512 returned tokens contain the answer.

For example, suppose `D42` is a gold document and the answer occurs around
tokens 800-820:

```text
D42 tokens 1..4096  -> one official document embedding
D42 is top-5        -> Gold Recall@5 receives credit for document ID D42
D42 tokens 1..512   -> Standard tool text shown to the LLM
answer at 800..820  -> LLM may still fail because the answer was truncated
```

That is not a qrels mismatch. It is a deliberate retrieval-to-context loss:
document discovery succeeded, but the Standard scaffold did not expose the
answer-bearing part of the document.

### What happens with custom chunking?

Custom chunks need stable chunk IDs plus their original parent document ID.
Official document-level retrieval metrics remain valid after collapsing the
chunk ranking to unique parent IDs by the first/highest-ranked occurrence.

Continuing the example, 512-token chunks with 64-token overlap might produce:

```text
D42#0: tokens   1..512
D42#1: tokens 449..960  <- contains the answer at 800..820
D42#2: tokens 897..1408
```

If `D42#1` is retrieved, collapse its parent to `D42` for official Gold
Recall/nDCG, while showing the answer-bearing chunk to the LLM. This is a
legitimate custom scaffold and may outperform Standard, but it must be labeled
Custom because the returned context changed.

One limitation remains: retrieving any chunk from parent `D42` can receive
document-level gold credit after collapse even if that particular chunk does
not contain the answer. Therefore report two metric families separately:

- official document metrics after parent collapse; and
- experimental context metrics for the exact chunks/text shown to the model.

The released qrels do not provide exhaustive answer-bearing token spans, so a
parent gold match is not itself proof that the selected chunk was sufficient.

## Immediate next step

Do not run the judge or context-budget experiments yet. First implement and
validate one dynamic Standard search request:

1. Preserve the retrieval baseline and its metadata.
2. Make the official index and corpus available to a search service on g3, or
   implement the alternative p7-FAISS/g3-embedding service boundary.
3. Keep Qwen3-Embedding-8B loaded on the PCIe A100.
4. Provide the official 0.6B tokenizer files locally for 512-token snippets;
   do not download its weights.
5. Verify one `search(query)` call returns exactly five document IDs/scores and
   no snippet exceeds 512 tokenizer tokens.
6. Then stage and serve Qwen3.6-27B on the SXM4 A100 and connect the search-only
   agent loop.

The current `rag-system` code prepares/validates the benchmark but does not yet
implement the complete Standard agent runner. Do not claim the end-to-end
baseline is runnable until the live search smoke and generator tool loop pass.

### Dynamic search implementation added after the retrieval reproduction

The active package now contains the first p7/g3 dynamic-search boundary:

- `scripts/serve_query_encoder.py` loads the local Qwen3-Embedding-8B model in
  the existing Tevatron environment on g3 and exposes `/health` and `/encode`;
- `RemoteQueryEncoder` sends one query to g3 and validates the returned vector;
- `FaissDocumentBackend` loads the official Tevatron index shards on p7,
  searches the returned query vector, and joins hits to the canonical corpus;
- `StandardSearchTool` requires exactly five unique document hits and truncates
  each agent-facing snippet to at most 512 local tokenizer tokens; and
- `scripts/smoke_standard_search.py` permits only frozen development IDs and
  prints no decrypted question or snippet text.

The unit suite passes with 20 tests. The live smoke was subsequently run
against the real p7 corpus/index and g3 Qwen3-Embedding-8B service. It returned
one development query ID, `top_k=5`, `snippet_max_tokens=512`, and five hits,
each containing rank, document ID, score, and a snippet token count no greater
than 512. No decrypted query or snippet text was recorded here.

This passes the dynamic Standard-search gate. The next gate is a persistent p7
search service and one Qwen3.6-27B search-only agent trajectory using the same
top-5/512 contract. Do not begin context-budget sweeps or the judge before that
trajectory is saved and validated.

### Persistent search and Standard agent implementation

The next runtime pieces are now implemented and pass 27 offline unit tests:

- `scripts/serve_standard_search.py` loads the official FAISS shards, canonical
  corpus, and 0.6B snippet tokenizer once on p7, then serves `/health` and
  `/search` on port 8012;
- `StandardSearchClient` rejects any response other than exactly five unique
  hits with snippet counts no greater than 512, and removes diagnostic token
  counts before giving tool output to the model;
- `VllmChatClient` uses vLLM's OpenAI-compatible Chat Completions endpoint with
  thinking enabled, automatic but strict tool choice, and parallel tool calls
  disabled;
- `StandardAgentWorkflow` uses the pinned upstream no-get-document prompt,
  permits only `search`, unions all retrieved document IDs, and emits the
  official run shape; and
- `scripts/run_standard_agent.py` accepts only frozen development IDs and
  writes a private resumable JSON record, including an error row on failure.

`scripts/serve_generator.sh` now enables vLLM automatic tool choice with the
`qwen3_coder` tool parser and retains the `qwen3` reasoning parser. These are
required for Qwen3.6 native tool calls. The live persistent server and agent
trajectory have not yet been run; exact commands are in the root README.

The first generator launch exposed vLLM CLI drift: the installed build rejected
`--attention-backend`, `--language-model-only`, and
`--enable-per-request-metrics`. The launcher is now version-aware. FlashInfer
is selected only through `VLLM_ATTENTION_BACKEND`; language-only mode falls
back to zero image/video limits when supported; and optional request/load
metrics flags are included only when present in `vllm serve --help`. Missing
telemetry remains explicitly warned and must not be synthesized later.

The installed g3 vLLM build also rejected `qwen3_coder` and listed only older
tool parsers. This is a hard incompatibility with the Qwen3.6 baseline, not a
flag-name difference: upgrade vLLM in a separate generator environment and
verify that `vllm serve --help` lists `qwen3_coder`. Do not substitute the
`hermes` parser. The launcher now checks this before loading model weights.

The first live query (`703`) completed two valid Standard search round trips
but its third generation stopped at exactly 10,000 completion tokens while
still reasoning. The saved record is correctly `status=incomplete` and is not
a passing smoke result. The runner now records each response's `finish_reason`,
accepts both vLLM reasoning field names, and reports an explicit termination
reason. Preserve this failed artifact and rerun in a new output directory with
a 20,000-token per-turn allowance before changing any other baseline setting.

## Meaning of agent leaderboard Recall (%)

The end-to-end leaderboard's `Recall (%)` has no fixed `K`. For query `q`, the
official evaluator computes:

```text
unique_retrieved_q = union of document IDs returned by every search call
recall_q = |unique_retrieved_q intersect evidence_q| / |evidence_q|
Recall (%) = 100 * macro-average(recall_q over queries)
```

It uses the evidence qrels, not the narrower gold qrels. With Standard top-5
search, `s` calls can expose at most `5*s` results before duplicates, so the
effective number of unique retrieved documents varies by trajectory. This
must be called trajectory evidence recall or agent recall in local reports,
not Recall@5. Retrieval-only `Evidence Recall@5`, `@100`, and `@1000` remain
separate fixed-ranking metrics.
