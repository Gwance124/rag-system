# LaTeX Chunking Pipeline (Pilot) — Design Spec

## Context

This is the first sub-project of a larger effort to build a reasoning-focused RAG system over an arxiv-latex-derived corpus (~650k papers filtered to CXL hardware / kv-cache / AI/ML/IR categories). The eventual system targets reasoning-heavy queries (e.g. "implement this algorithm from the paper"), which existing benchmarks (BEIR, MTEB, BRIGHT, LitSearch) don't represent well — LitSearch in particular caps papers at ~2000 tokens and is built from citation-context questions, not implementation-style reasoning queries.

Full papers in this corpus run far longer than any of these benchmarks assume: a 50k-paper sample showed **avg ~39k tokens, median ~24k tokens** per paper. The embedding model available (`llama-nv-embed-reason-3b`, served via vLLM) was trained at a 512-token max and only evaluated up to 8192 — nowhere near full-paper length. This means papers must be chunked before embedding; that chunking step is the subject of this spec.

Building a proper reasoning eval benchmark requires human-labeled positive/negative documents, which isn't feasible right now — so this sub-project produces no benchmark. Its only goal is to get from raw parquet rows to inspectable, well-formed chunks so quality can be judged by hand and used to unblock the next sub-project (embedding + indexing).

## Scope

**In scope:**
- Filter the arxiv-latex parquet files (parts 0001-0044, already cs-category filtered) down to a pilot set: `cs.IR` papers from 2020 onward (~18,727 papers).
- Parse each paper's `latex` field (already a compiled, Markdown-style tree per the arxiv-latex dataset card — not raw `.tex`) into a structured document: title, abstract, ordered sections with heading text/level/path, and ordered content blocks (paragraph, code/algorithm, table, figure-caption, equation) per section.
- Strip bibliography/references sections and residual LaTeX bloat (`\cite{}`, `\ref{}`, `\label{}`, comments) that may leak through the compiled markdown.
- Chunk each paper's blocks into ~512-token chunks (measured with the real `llama-nv-embed-reason-3b` tokenizer, loaded from the local vLLM model directory — no new downloads), respecting block boundaries where possible.
- Prefix each chunk with document context (title + abstract + section path) before the chunk body, per the "contextual retrieval" pattern, to reduce ambiguity of isolated passages.
- Write chunk records to parquet for manual inspection.
- Log parse failures without crashing the run.

**Out of scope (deferred to later sub-projects):**
- Embedding chunks or indexing into Qdrant.
- Any eval/benchmark construction.
- The two-stage coarse/fine retrieval architecture (only needed once corpus size makes flat chunk-level indexing impractical; not needed at 18,727-paper pilot scale).
- Handling pre-2020 papers with messy/legacy LaTeX (the pilot's 2020+ filter sidesteps this).

## Architecture

```
parquet files (0001-0044, cs-filtered)
   │
   ▼
[Stage 0] Filter to pilot set
   - filter rows where `categories` contains "cs.IR" AND year (derived from `yymm_id`) >= 2020
   - keep only needed columns: id, title, abstract, categories, yymm_id, latex
   - output: pilot_papers.parquet (~18,727 rows)
   │
   ▼
[Stage 1] Inspect + Parse
   - FIRST: sample and manually eyeball raw `latex` field values before finalizing the parser,
     since it's unconfirmed whether the field is clean Markdown or still has LaTeX residue
   - parse each `latex` string into a structured document tree:
       title, abstract, ordered sections (heading text, level, section path e.g. "3.2 Related Work")
       each section -> ordered blocks: paragraph | code/algorithm | table | figure-caption | equation
   - drop bibliography/references section and residual LaTeX bloat
   │
   ▼
[Stage 2] Chunk
   - walk the document tree, greedily group blocks into chunks up to ~512 tokens
     (token count via the real nv-embed-reason-3b tokenizer)
   - never split a code/algorithm block or equation across a chunk boundary if avoidable
   - if a single block alone exceeds ~512 tokens, split it at sentence boundaries (paragraphs)
     or blank-line boundaries (code) rather than silently truncating
   - each chunk gets a context prefix: "{title}\n{abstract}\n{section_path}\n\n" + chunk body
   │
   ▼
[Stage 3] Write output
   - one row per chunk -> chunks.parquet (schema below)
   │
   ▼
[Stage 4] Manual inspection / spot-check
   - print full chunk sequences for a handful of papers; check for broken code blocks,
     leftover LaTeX junk, sane token counts, and that bibliography was actually excluded
```

Stages 0-3 are meant to run unattended against real data on the work laptop/servers (this environment can't reach the parquet files or the vLLM tokenizer). Stage 1's manual sample inspection is the first implementation task and may force small parser adjustments before the rest is built out.

## Data Schemas

**Source** (arxiv-latex dataset, relevant columns only):
| Column | Type | Notes |
|---|---|---|
| `id` | string | arXiv identifier |
| `yymm_id` | string | normalized YYMM id, used to derive year |
| `title` | string | |
| `categories` | string | space-separated, e.g. `cs.IR cs.CL` |
| `abstract` | string | |
| `latex` | large_string | compiled, Markdown-style tree bundling all `.tex`/`.bib`/`.sty` source |

**`pilot_papers.parquet`** (Stage 0 output): `id, title, abstract, categories, yymm_id, latex`

**`chunks.parquet`** (Stage 3 output, one row per chunk):
| Column | Type | Notes |
|---|---|---|
| `id` | string | arxiv id, matches source `id` |
| `chunk_index` | int | 0-based position within paper |
| `section_path` | string | e.g. `"3.2 Related Work"` |
| `text_with_context` | string | `"{title}\n{abstract}\n{section_path}\n\n{body}"` |
| `text_raw` | string | body only, no prefix |

Token count and block-type composition aren't persisted — they're transient checks computed during Stage 4's manual inspection (via the real tokenizer) rather than stored fields, since nothing downstream consumes them yet.

**`parse_failures.jsonl`**: one line per paper that failed to parse — `{id, error}`.

## Error Handling

A paper that fails to parse (unexpected structure, empty body, etc.) is skipped and logged to `parse_failures.jsonl` rather than crashing the run. With 18,727 papers, a handful of outliers shouldn't block the pilot; the failure log quantifies how big the problem actually is and informs whether the parser needs hardening before scaling past the pilot.

## Testing Strategy

- Parser and chunker logic get unit tests against small synthetic/hand-written Markdown-tree-style fixtures (mimicking expected structure: headings, paragraphs, code blocks, a fake bibliography section) — these can run on this laptop without access to real data.
- Token-counting and chunk-boundary logic (oversized block splitting, never-split-code-blocks rule) get unit tests with fixtures crafted to hit those edge cases specifically.
- Real-data validation (Stage 1's sample inspection, Stage 4's spot-check) happens later, run by the user on the work laptop/server against actual parquet files and the actual tokenizer — this is a manual verification step, not an automated test, and is called out explicitly as an implementation task.

## Open Question Flagged for Implementation

Whether the `latex` field is genuinely clean Markdown or still contains raw LaTeX commands (`\cite{}`, `\ref{}`, math macros, etc.) is unconfirmed as of this spec. The parser should be written defensively — assume Markdown-like structure but include regex cleanup for common leftover LaTeX commands — and the first implementation task must inspect real samples before the parser design is locked in further.