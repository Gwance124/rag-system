#!/usr/bin/env python3
"""Run or resume the frozen GPT-OSS-20B development split sequentially."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from rag_system.datasets.browsecomp_plus import (
    DEFAULT_DEVELOPMENT_COUNT,
    load_prepared_development_query_ids,
)
from rag_system.workflows.oss_standard_agent import (
    BROWSECOMP_PLUS_OSS_COMMIT,
    BROWSECOMP_PLUS_OSS_REPOSITORY,
    BROWSECOMP_PLUS_OSS_RUNNER,
)
from rag_system.workflows.oss_standard_batch import (
    atomic_private_json,
    preflight_oss_standard_services,
    run_resumable_development_batch,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--search-url", default="http://127.0.0.1:8012")
    parser.add_argument("--generator-url", required=True)
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default="high",
    )
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--max-search-calls", type=int, default=100)
    parser.add_argument("--max-generation-retries", type=int, default=2)
    parser.add_argument("--max-output-tokens", type=int, default=10_000)
    parser.add_argument("--context-budget-tokens", type=int, default=128_000)
    parser.add_argument("--generator-timeout-seconds", type=float, default=2400.0)
    parser.add_argument("--quiet-query-progress", action="store_true")
    args = parser.parse_args()

    prepared_dir = args.prepared_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(output_dir, 0o700)
    query_ids = load_prepared_development_query_ids(prepared_dir)
    if len(query_ids) != DEFAULT_DEVELOPMENT_COUNT:
        parser.error(
            "prepared split has "
            f"{len(query_ids)} development queries; expected "
            f"{DEFAULT_DEVELOPMENT_COUNT}"
        )
    preflight_oss_standard_services(
        args.search_url,
        args.generator_url,
        args.model,
    )

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

        name = event["event"]
        if name == "batch_started":
            message = f"batch started total={event['total_query_count']}"
        elif name == "query_skipped":
            message = (
                f"[{event['batch_index']}/{len(query_ids)}] "
                f"query={event['query_id']} skipped status={event['status']}"
            )
        elif name == "query_started":
            message = (
                f"[{event['batch_index']}/{len(query_ids)}] "
                f"query={event['query_id']} started"
            )
        elif name == "query_finished":
            message = (
                f"[{event['batch_index']}/{len(query_ids)}] "
                f"query={event['query_id']} finished status={event['status']}"
            )
        elif name == "batch_finished":
            message = (
                f"batch finished executed={event['executed_count']} "
                f"skipped={event['skipped_existing_count']} "
                f"remaining={event['remaining_query_count']} "
                f"statuses={event['status_counts']}"
            )
        elif name == "batch_stopping":
            message = (
                f"batch stopping after query={event['query_id']} "
                f"reason={event['reason']}"
            )
        else:
            message = json.dumps(event, sort_keys=True)
        print(
            f"[{time.monotonic() - started_at:8.1f}s] {message}",
            file=sys.stderr,
            flush=True,
        )

    single_runner = Path(__file__).resolve().with_name("run_oss_standard_agent.py")

    def execute_query(query_id: str) -> dict:
        command = [
            sys.executable,
            str(single_runner),
            "--prepared-dir",
            str(prepared_dir),
            "--query-id",
            query_id,
            "--search-url",
            args.search_url,
            "--generator-url",
            args.generator_url,
            "--model",
            args.model,
            "--output-dir",
            str(output_dir),
            "--reasoning-effort",
            args.reasoning_effort,
            "--max-iterations",
            str(args.max_iterations),
            "--max-search-calls",
            str(args.max_search_calls),
            "--max-generation-retries",
            str(args.max_generation_retries),
            "--max-output-tokens",
            str(args.max_output_tokens),
            "--context-budget-tokens",
            str(args.context_budget_tokens),
            "--generator-timeout-seconds",
            str(args.generator_timeout_seconds),
            # The batch skips completed artifacts. Force only clears an
            # interrupted query's orphaned progress file before resuming it.
            "--force",
        ]
        if args.quiet_query_progress:
            command.append("--quiet-progress")
        completed = subprocess.run(command, check=False)
        output_path = output_dir / f"run_{query_id}.json"
        if completed.returncode != 0:
            raise RuntimeError(
                f"query runner exited with status {completed.returncode}"
            )
        try:
            return json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"query runner did not create a valid artifact at {output_path}"
            ) from exc

    summary = run_resumable_development_batch(
        query_ids,
        output_dir,
        execute_query,
        progress_callback=progress,
    )
    summary_payload = {
        **summary.to_dict(),
        "schema_version": "1.0",
        "model": args.model,
        "generator_url": args.generator_url,
        "search_url": args.search_url,
        "reasoning_effort": args.reasoning_effort,
        "max_output_tokens": args.max_output_tokens,
        "max_iterations": args.max_iterations,
        "max_search_calls": args.max_search_calls,
        "max_generation_retries": args.max_generation_retries,
        "context_budget_tokens": args.context_budget_tokens,
        "generator_timeout_seconds": args.generator_timeout_seconds,
        "sequential": True,
        "resume_policy": "skip_valid_run_artifacts",
        "upstream_reference": {
            "repository": BROWSECOMP_PLUS_OSS_REPOSITORY,
            "commit": BROWSECOMP_PLUS_OSS_COMMIT,
            "runner": BROWSECOMP_PLUS_OSS_RUNNER,
        },
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = output_dir / "batch_summary.json"
    atomic_private_json(summary_path, summary_payload)
    print(json.dumps({**summary_payload, "summary_path": str(summary_path)}, indent=2))
    if summary.stopped_on_error or summary.remaining_query_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
