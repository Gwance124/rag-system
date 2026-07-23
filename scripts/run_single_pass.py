#!/usr/bin/env python3
"""Run the no-tool single-pass RAG baseline over the frozen ranking."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from rag_system.contracts import SearchCandidate
from rag_system.datasets.browsecomp_plus import (
    DEFAULT_DEVELOPMENT_COUNT,
    iter_corpus_repository,
    load_prepared_development_query,
    load_prepared_development_query_ids,
)
from rag_system.evaluation.run_summary import summarize_run_directory
from rag_system.generation.vllm_responses import VllmResponsesClient
from rag_system.workflows.oss_standard_batch import (
    atomic_private_json,
    preflight_generator,
    run_resumable_development_batch,
)
from rag_system.workflows.single_pass import (
    SinglePassWorkflow,
    build_document_lookup,
    load_trec_ranking,
    select_context_document_ids,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument(
        "--ranking-trec",
        type=Path,
        required=True,
        help="Frozen top-1000 TREC run from the retrieval reproduction",
    )
    parser.add_argument("--corpus-repo", type=Path, required=True)
    parser.add_argument("--datasets-cache", type=Path)
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        required=True,
        help="Local Qwen3-0.6B tokenizer directory (Standard snippet contract)",
    )
    parser.add_argument("--generator-url", required=True)
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reasoning-effort", choices=["low", "medium", "high"], default="high"
    )
    parser.add_argument("--max-output-tokens", type=int, default=10_000)
    parser.add_argument("--max-generation-retries", type=int, default=2)
    parser.add_argument("--generator-timeout-seconds", type=float, default=2400.0)
    parser.add_argument(
        "--query-id",
        help="Run one frozen development query instead of the whole split",
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        parser.error(f"Transformers is required: {exc}")

    prepared_dir = args.prepared_dir.expanduser().resolve()
    ranking_path = args.ranking_trec.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(output_dir, 0o700)

    if args.query_id is not None:
        query_ids = (args.query_id,)
        # Raises if the ID is not part of the frozen development split.
        load_prepared_development_query(prepared_dir, args.query_id)
    else:
        query_ids = load_prepared_development_query_ids(prepared_dir)
        if len(query_ids) != DEFAULT_DEVELOPMENT_COUNT:
            parser.error(
                f"prepared split has {len(query_ids)} development queries; "
                f"expected {DEFAULT_DEVELOPMENT_COUNT}"
            )

    preflight_generator(args.generator_url, args.model)

    ranking_sha256 = _sha256_file(ranking_path)
    ranking = load_trec_ranking(ranking_path)
    needed_document_ids = select_context_document_ids(
        ranking, query_ids, args.top_k
    )
    print(
        f"loading {len(needed_document_ids)} context documents from the corpus",
        file=sys.stderr,
        flush=True,
    )
    document_lookup = build_document_lookup(
        iter_corpus_repository(
            args.corpus_repo.expanduser().resolve(),
            cache_dir=args.datasets_cache,
        ),
        needed_document_ids,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer_path.expanduser().resolve()),
        local_files_only=True,
    )
    responses_client = VllmResponsesClient(
        base_url=args.generator_url,
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
        timeout_seconds=args.generator_timeout_seconds,
    )

    metadata = {
        "model": args.model,
        "generator_url": args.generator_url,
        "api": "responses",
        "reasoning_effort": args.reasoning_effort,
        "scaffold": "single_pass_topk_first512",
        "top_k": args.top_k,
        "ranking_trec": str(ranking_path),
        "ranking_sha256": ranking_sha256,
        "max_output_tokens": args.max_output_tokens,
        "max_generation_retries": args.max_generation_retries,
        "generator_timeout_seconds": args.generator_timeout_seconds,
    }

    batch_progress_path = output_dir / "batch.progress.jsonl"
    batch_progress_path.touch(exist_ok=True)
    os.chmod(batch_progress_path, 0o600)
    started_at = time.monotonic()

    def progress(event: dict) -> None:
        durable_event = {
            "elapsed_seconds": round(time.monotonic() - started_at, 3),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            **event,
        }
        with batch_progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(durable_event, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(
            f"[{time.monotonic() - started_at:8.1f}s] "
            + json.dumps(event, ensure_ascii=False, sort_keys=True),
            file=sys.stderr,
            flush=True,
        )

    def execute_query(query_id: str) -> dict:
        query = load_prepared_development_query(prepared_dir, query_id)
        candidates = [
            SearchCandidate(
                document_id=document_id,
                score=score,
                text=document_lookup[document_id],
            )
            for document_id, score in ranking[query_id][: args.top_k]
        ]
        workflow = SinglePassWorkflow(
            responses_client=responses_client,
            tokenizer=tokenizer,
            max_generation_retries=args.max_generation_retries,
            progress_callback=lambda event: progress(
                {"query_id": query_id, **event}
            ),
        )
        record = workflow.run(query, candidates)
        record["metadata"] = {
            **metadata,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        return record

    summary = run_resumable_development_batch(
        query_ids,
        output_dir,
        execute_query,
        progress_callback=progress,
    )
    summary_payload = {
        **summary.to_dict(),
        **metadata,
        "schema_version": "1.0",
        "sequential": True,
        "resume_policy": "skip_valid_run_artifacts",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_private_json(output_dir / "batch_summary.json", summary_payload)

    try:
        recall_summary = summarize_run_directory(
            output_dir,
            functools.partial(load_prepared_development_query, prepared_dir),
        )
        atomic_private_json(output_dir / "recall_summary.json", recall_summary)
        headline = {
            key: recall_summary[key]
            for key in (
                "run_count",
                "status_counts",
                "evidence_recall_mean",
                "gold_recall_mean",
                "final_answer_format_valid_count",
            )
        }
    except ValueError as exc:
        headline = {"recall_summary_error": str(exc)}
    print(json.dumps({**summary.to_dict(), **headline}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
