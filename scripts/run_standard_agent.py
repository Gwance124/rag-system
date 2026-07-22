#!/usr/bin/env python3
"""Run and save one dev-only Qwen3.6 BrowseComp-Plus Standard trajectory."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from rag_system.datasets.browsecomp_plus import load_prepared_development_query
from rag_system.generation.vllm_chat import VllmChatClient
from rag_system.retrieval.search_service import StandardSearchClient
from rag_system.workflows.standard_agent import StandardAgentWorkflow


def atomic_private_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            os.chmod(temporary, 0o600)
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--query-id", required=True)
    parser.add_argument("--search-url", default="http://127.0.0.1:8012")
    parser.add_argument("--generator-url", required=True)
    parser.add_argument("--model", default="qwen3.6-27b")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-search-calls", type=int, default=20)
    parser.add_argument("--max-output-tokens", type=int, default=10_000)
    parser.add_argument("--generator-timeout-seconds", type=float, default=2400.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--min-p", type=float)
    parser.add_argument("--presence-penalty", type=float)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--preserve-thinking", action="store_true")
    parser.add_argument(
        "--thinking-token-budget",
        type=int,
        help="optional vLLM reasoning cap; omit for the upstream-comparable baseline",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet-progress", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_path = args.output_dir.expanduser().resolve() / f"run_{args.query_id}.json"
    progress_path = output_path.with_suffix(".progress.jsonl")
    if output_path.exists() and not args.force:
        parser.error(f"output already exists: {output_path}; pass --force to replace it")
    if progress_path.exists() and not args.force:
        parser.error(
            f"progress log already exists: {progress_path}; pass --force to replace it"
        )
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text("", encoding="utf-8")
    os.chmod(progress_path, 0o600)

    query = load_prepared_development_query(args.prepared_dir, args.query_id)
    chat_client = VllmChatClient(
        base_url=args.generator_url,
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        preserve_thinking=args.preserve_thinking,
        seed=args.seed,
        timeout_seconds=args.generator_timeout_seconds,
        thinking_token_budget=args.thinking_token_budget,
    )
    started_at = time.monotonic()

    def metric(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    def progress(event: dict) -> None:
        elapsed = time.monotonic() - started_at
        durable_event = {
            "elapsed_seconds": round(elapsed, 3),
            "query_id": query.query_id,
            **event,
        }
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(durable_event, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if args.quiet_progress:
            return
        name = event["event"]
        if name == "generation_started":
            message = (
                f"turn {event['turn']}: generating "
                f"({event['completed_search_calls']} searches completed)"
            )
        elif name == "run_started":
            message = (
                f"run started max_output_tokens={event['max_output_tokens']} "
                f"thinking_token_budget={event['thinking_token_budget']} "
                f"generator_timeout_seconds={event['generator_timeout_seconds']}"
            )
        elif name == "generation_completed":
            usage = event.get("usage") or {}
            message = (
                f"turn {event['turn']}: generation finished "
                f"reason={event.get('finish_reason')} "
                f"prompt_tokens={usage.get('prompt_tokens', 'n/a')} "
                f"completion_tokens={usage.get('completion_tokens', 'n/a')} "
                f"tool_calls={event['tool_call_count']}"
            )
        elif name == "generation_failed":
            error = event["error"]
            message = (
                f"turn {event['turn']}: generation failed "
                f"{error['type']}: {error['message']}"
            )
        elif name == "search_started":
            message = f"search {event['search_call']}: {event['query']}"
        elif name == "search_completed":
            evidence = event["evidence"]
            gold = event["gold"]
            message = (
                f"search {event['search_call']}: returned={event['returned_documents']} "
                f"unique_total={event['unique_documents_cumulative']} "
                f"evidence_hits={evidence['turn_hits']}/{evidence['relevant_documents']} "
                f"evidence_new={evidence['new_hits']} "
                f"evidence_recall@5={metric(evidence['turn_recall_at_5'])} "
                f"evidence_ndcg@5={metric(evidence['turn_ndcg_at_5'])} "
                f"evidence_cumulative={metric(evidence['cumulative_recall'])} "
                f"gold_hits={gold['turn_hits']}/{gold['relevant_documents']} "
                f"gold_new={gold['new_hits']} "
                f"gold_recall@5={metric(gold['turn_recall_at_5'])} "
                f"gold_ndcg@5={metric(gold['turn_ndcg_at_5'])} "
                f"gold_cumulative={metric(gold['cumulative_recall'])}"
            )
        elif name == "search_failed":
            error = event["error"]
            message = (
                f"search {event['search_call']}: failed "
                f"{error['type']}: {error['message']}"
            )
        elif name == "run_finished":
            message = (
                f"run finished status={event['status']} "
                f"termination={event['termination_reason']} "
                f"searches={event['search_calls']} "
                f"unique_documents={event['unique_retrieved_documents']}"
            )
        else:
            message = json.dumps(event, sort_keys=True)
        print(f"[{elapsed:8.1f}s] {message}", file=sys.stderr, flush=True)

    workflow = StandardAgentWorkflow(
        chat_client=chat_client,
        search_client=StandardSearchClient(args.search_url),
        max_search_calls=args.max_search_calls,
        progress_callback=progress,
    )
    progress(
        {
            "event": "run_started",
            "max_output_tokens": args.max_output_tokens,
            "thinking_token_budget": args.thinking_token_budget,
            "generator_timeout_seconds": args.generator_timeout_seconds,
            "sampling": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "min_p": args.min_p,
                "presence_penalty": args.presence_penalty,
                "repetition_penalty": args.repetition_penalty,
            },
            "preserve_thinking": args.preserve_thinking,
        }
    )
    try:
        record = workflow.run(query)
    except Exception as exc:
        record = {
            "schema_version": "1.0",
            "query_id": query.query_id,
            "tool_call_counts": {"search": 0},
            "status": "error",
            "retrieved_docids": [],
            "result": [],
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    diagnostics = record.get("diagnostics") or {}
    progress(
        {
            "event": "run_finished",
            "status": record["status"],
            "termination_reason": diagnostics.get("termination_reason", "exception"),
            "search_calls": record["tool_call_counts"].get("search", 0),
            "unique_retrieved_documents": len(record["retrieved_docids"]),
        }
    )
    record["metadata"] = {
        "model": args.model,
        "generator_url": args.generator_url,
        "search_url": args.search_url,
        "scaffold": "standard_search_only_top5_first512",
        "seed": args.seed,
        "max_output_tokens": args.max_output_tokens,
        "generator_timeout_seconds": args.generator_timeout_seconds,
        "thinking_token_budget": args.thinking_token_budget,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "preserve_thinking": args.preserve_thinking,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_private_json(output_path, record)
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "progress_path": str(progress_path),
                "query_id": query.query_id,
                "status": record["status"],
                "search_calls": record["tool_call_counts"].get("search", 0),
                "unique_retrieved_documents": len(record["retrieved_docids"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
