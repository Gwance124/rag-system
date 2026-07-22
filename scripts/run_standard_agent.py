#!/usr/bin/env python3
"""Run and save one dev-only Qwen3.6 BrowseComp-Plus Standard trajectory."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_path = args.output_dir.expanduser().resolve() / f"run_{args.query_id}.json"
    if output_path.exists() and not args.force:
        parser.error(f"output already exists: {output_path}; pass --force to replace it")

    query = load_prepared_development_query(args.prepared_dir, args.query_id)
    chat_client = VllmChatClient(
        base_url=args.generator_url,
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        seed=args.seed,
    )
    workflow = StandardAgentWorkflow(
        chat_client=chat_client,
        search_client=StandardSearchClient(args.search_url),
        max_search_calls=args.max_search_calls,
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
    record["metadata"] = {
        "model": args.model,
        "generator_url": args.generator_url,
        "search_url": args.search_url,
        "scaffold": "standard_search_only_top5_first512",
        "seed": args.seed,
        "max_output_tokens": args.max_output_tokens,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_private_json(output_path, record)
    print(
        json.dumps(
            {
                "output_path": str(output_path),
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
