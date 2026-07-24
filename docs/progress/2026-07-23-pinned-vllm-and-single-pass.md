# Pinned-vLLM parity plan and single-pass baseline handoff

Status date: 2026-07-23

This handoff supersedes the stale generator sections of
`docs/progress/2026-07-22-qwen3-embedding-baseline-handoff.md`. The retrieval
reproduction, host topology, pinned assets, and qrels/snippet discussion in
that document remain authoritative.

## Decisions made today

1. **Stay on BrowseComp-Plus.** BRIGHT/BRIGHT-PRO was rejected because it is
   retrieval-only: encoders and rerankers barely exercise KV cache, so it has
   no decode-side story for the KV/latency/CXL systems study. ResearchQA was
   rejected because it is rubric-LLM-judge scored with no fixed corpus or
   qrels, which breaks deterministic systems measurement. BCP already
   provides the fixed corpus, document qrels, a passed retrieval
   reproduction, and a generation-heavy agentic workload.
2. **The generator baseline is gpt-oss-20b; the server is vLLM 0.19.1**
   (corrected the same evening: the earlier "broken build, pin to v0.10.1"
   diagnosis was wrong). The failures decomposed into (a) client-side
   strictness — the runner demanded argument keys exactly `{"user_query"}`
   and hard-aborted the run on any imperfect call — fixed the same day to
   match upstream's tolerant parse-and-feed-back-errors behavior; (b)
   transient `unexpected tokens remaining in message header` 500s
   (vllm#23567 class): at temperature 0.7 the model occasionally emits
   Harmony tokens the parser rejects, on any version, and bounded retries
   absorb them; (c) native-browser `mcp_call` aliases, already normalized.
   After the fixes, queries 703 and 215 completed with real searches on
   0.19.1. The pinned upstream images (`vllm/vllm-openai:v0.10.1`,
   `:gptoss`) are the escalation fallback only. Runbook and gates:
   `docs/oss-20b-pinned-generator-parity.md`.
3. **Parity run vs measurement run.** Leaderboard parity uses upstream
   defaults (prefix caching on, no telemetry flags) and its rows are never
   used for latency/KV reporting. The instrumented sequential scaffold with
   prefix caching disabled remains the only source of measurement rows.
4. **Recall gate only for parity.** Dev-100 mean trajectory evidence recall
   in 0.44–0.54 (BrowseComp-Plus paper Table 4, `gpt-oss-20B-high` +
   `Qwen3-Embed-8B`: 49.29% recall on 830 queries, mean 23.87 searches/query)
   passes; the Qwen3-32B judge accuracy check is deferred. (Corrected from an
   earlier 32.17%/43.0%/12.6-searches figure that didn't match any row in
   Table 4.)
5. **Single-pass first.** The first systems deliverable is the no-tool
   single-pass baseline over the frozen `top1000.trec` (top-k documents,
   512-token Standard snippets, one Responses call). It is immune to Harmony
   tool bugs and produces the first context-budget (k ∈ {5, 10, 20}) versus
   recall/quality rows even if agent parity stalls.

## Code added today (61 unit tests pass)

- `src/rag_system/workflows/single_pass.py`: `SinglePassWorkflow` (no-tool
  Responses generation over provided candidates, official run-record shape,
  same Explanation/Exact Answer/Confidence contract), `load_trec_ranking`,
  `select_context_document_ids`, `build_document_lookup`.
- `src/rag_system/evaluation/run_summary.py`: `summarize_run_directory`
  macro-averages trajectory evidence/gold recall, status counts, search
  calls, and answer-format validity over any `run_*.json` directory (agent
  and single-pass runs share the record shape).
- `scripts/run_single_pass.py`: resumable dev-split single-pass batch; loads
  only the ranked documents actually needed, hashes the ranking file, writes
  `batch_summary.json` and `recall_summary.json`.
- `scripts/summarize_agent_runs.py`: recall summary CLI for existing agent
  run directories (the Phase A parity gate).
- `prefix_snippet` extracted in `src/rag_system/retrieval/standard.py`
  (shared by the search tool and single-pass); `preflight_generator`
  extracted in `src/rag_system/workflows/oss_standard_batch.py`.
- `docs/oss-20b-pinned-generator-parity.md`: full runbook (pinned server,
  smoke gates, dev-100 batch, single-pass commands, escalation ladder).

## Next actions (on the lab hosts)

1. g3: stop the bare-metal generator; serve gpt-oss-20b from
   `vllm/vllm-openai:v0.10.1` (or `vllm==0.10.1` venv). Record `/version`.
2. p7: one-query smoke via `run_oss_standard_agent.py`; gates in the runbook.
3. p7: dev-100 batch into `development-pinned-v0101`; check
   `summarize_agent_runs.py` recall against the 0.44–0.54 band.
4. p7: single-pass smoke, then dev-100 at k=5, then k=10/k=20 sweeps.
5. Escalate per the runbook ladder if gates fail; single-pass proceeds
   regardless.

No decrypted benchmark content appears in any of today's artifacts or docs.
