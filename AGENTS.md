# Repository Guidelines

## Current Project Boundary

This repository is being rebuilt as a local RAG systems research project. The
active design is documented in
`docs/plans/2026-07-21-rag-systems-four-week-plan.md`.

The current execution state and the latest decisions that supersede stale
parts of that plan are documented in
`docs/progress/2026-07-23-pinned-vllm-and-single-pass.md` (which itself
supersedes only the generator sections of
`docs/progress/2026-07-22-qwen3-embedding-baseline-handoff.md`). Read both
before proposing commands or changing the experiment. The Qwen3-Embedding-8B
retrieval reproduction and the persistent p7 Standard search service have
passed. The generator baseline is now gpt-oss-20b; its Harmony tool-calling
failures were diagnosed as bugs in the bare-metal g3 vLLM build. The next
gates are (a) a dev-100 recall parity batch against the pinned upstream vLLM
image and (b) the no-tool single-pass baseline over the frozen ranking, per
`docs/oss-20b-pinned-generator-parity.md`.

`old/` is an archive of the scientific-retrieval and LaTeX-chunking work. Do
not add features there, import it from new code, or use QASPER as a benchmark
for the new study. Port a small legacy primitive into the active package only
when its behavior is covered by a new regression test. The root `README.md` and
`pyproject.toml` describe the new active scaffold; legacy build instructions
belong only under `old/`.

The intended active layout is:

- `src/rag_system/`: dataset, chunking, retrieval, packing, generation,
  telemetry, evaluation, artifact, and workflow modules.
- `scripts/`: thin entry points for preparing data, building indexes, running
  experiments, validating artifacts, and plotting results.
- `tests/`: active pytest suite, organized by module.
- `configs/` and `prompts/`: frozen experiment configuration and prompt files.
- `docs/plans/`: current designs and implementation plans.
- `old/`: read-only legacy background and appendix material.

Generated datasets, decrypted benchmark text, caches, indexes, rankings,
traces, results, and plots belong under a gitignored artifact root, never
inside `src/` or `tests/`. BrowseComp-Plus questions and answers are
obfuscated upstream and must not be committed in decrypted form.

## Research Guardrails

BrowseComp-Plus is the primary benchmark. Preserve a deterministic development
split and do not inspect held-out results while tuning. WixQA is only a backup
pipeline sanity check. Do not introduce QASPER or TREC RAG into the new
experiments.

Week 1 now targets two BrowseComp-Plus baselines with gpt-oss-20b and
Qwen3-Embedding-8B on the official precomputed document index: (a) the
Standard agent baseline (search-only tool, top 5 results, at most 512 tokens
per result) for leaderboard recall parity, run on the recorded current g3
vLLM (0.19.1) with the pinned upstream image as the escalation fallback,
and (b) the no-tool single-pass baseline over the frozen top-1000
ranking with a top-k in {5, 10, 20} sweep. Do not add `get_document`, custom
chunking, reranking, subagents, concurrency experiments, or an LLM judge
before those baselines pass. Keep parity rows (upstream server defaults) and
measurement rows (instrumented sequential scaffold, prefix caching disabled)
in separate run directories; never report latency or KV metrics from a
parity server.

Measurement hardware is the two-A100 80 GB `solab-g3` host: use the PCIe A100
for Qwen3-Embedding-8B retrieval and the SXM4 A100 for Qwen3.6-27B generation.
Dataset preparation, orchestration, and result ownership remain on CPU-only
`solab-p7`. Run a Qwen3-32B judge later and sequentially, not concurrently with
the generator/retriever baseline. Never use the shared 4x H200 endpoint for
reported measurements. Prefix caching is disabled initially. Discover and
save the installed vLLM `/metrics` surface; do not hard-code metric names
without version-aware validation.

Every run must be reproducible from a manifest. Pin dataset, model, tokenizer,
and software revisions; hash chunk, index, ranking, prompt, and split artifacts;
record the exact server launch command and GPU configuration; retain error rows
rather than silently dropping failed requests. A context-budget sweep must
reuse the same frozen ranking for every budget.

## Build, Test, and Development Commands

- `python -m pip install -e ".[dev,eval]"` installs active code and test/eval
  dependencies.
- `pytest` runs the active unit and integration suite.
- `pytest tests/test_browsecomp_plus.py -v` runs the current dataset-focused
  tests while iterating.
- `python scripts/stage_offline_assets.py --output-root <path>` builds the
  pinned laptop-side transfer bundle.
- `python scripts/prepare_browsecomp_plus.py --queries-repo <path>
  --corpus-repo <path> --output-dir <path>` validates and prepares the
  transferred benchmark on `solab-p7`.

Legacy commands and tests under `old/` are archival references, not release
gates for the new package.

## Coding Style and Interfaces

Target Python 3.10 or newer. Use four-space indentation,
standard-library-first imports, type hints for public interfaces, and small,
focused modules. Use `snake_case` for functions/files, `PascalCase` for
classes, and `UPPER_SNAKE_CASE` for constants. Keep network clients
dependency-light and injectable so tests can use local fake HTTP servers.

Use immutable dataclasses for cross-module records and explicit units in field
names (`*_tokens`, `*_ms`, `*_utc_ns`, `*_fraction`). Use wall-clock time only
for timestamps and a monotonic clock for durations. Missing telemetry is
`null` with a reason/provenance field; never synthesize unavailable
request-level metrics from server-wide aggregates.

Shell scripts must remain Bash-compatible and retain `set -euo pipefail`.
Keep CLI files thin: orchestration logic belongs in `src/rag_system/` and must
be callable directly from tests.

## Testing and Experiment Verification

Name test files `test_*.py` and functions `test_<behavior>`. Add a regression
test for each behavior change. Prefer in-memory fixtures, `tmp_path`,
`monkeypatch`, and local fake HTTP/SSE servers over network or service calls.
Mark real A100, vLLM, Qdrant, and full-corpus checks as explicit integration or
smoke tests.

Before a measurement run, require a clean committed code state (or record and
hash the diff for a diagnostic run), a validated manifest, an exclusive GPU,
the expected model revision, disabled prefix caching, and a saved `/metrics`
catalog. After a run, validate unique experiment keys, expected row counts,
selected IDs, token-budget bounds, monotonic timestamps, and artifact hashes.

## Commit and Pull Request Guidelines

Use short, task-focused commit subjects. Keep archive moves, active scaffolding,
runtime features, and experiment outputs separate. Pull requests should state
the motivation, implementation, verification commands, and any new data,
cache, Qdrant, vLLM, or GPU requirements. For experiment changes, include the
dataset revision, split, workflow, model/configuration, and a concise metric
sample. Never commit decrypted benchmark content, large artifacts, or secrets.
