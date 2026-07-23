"""Aggregate a directory of agent run artifacts into leaderboard-style metrics.

The official BrowseComp-Plus agent evaluator macro-averages trajectory
evidence recall over queries. This module reproduces that recall aggregation
locally (without the LLM judge) so a run directory can be checked against a
leaderboard recall band before any accuracy scoring.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rag_system.contracts import BenchmarkQuery
from rag_system.evaluation.retrieval import trajectory_evidence_recall

QueryLoader = Callable[[str], BenchmarkQuery]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_run_directory(
    run_dir: str | Path,
    query_loader: QueryLoader,
) -> dict[str, Any]:
    """Summarize every ``run_<query_id>.json`` in ``run_dir``.

    The summary contains only IDs, statuses, and metrics; no decrypted
    question, answer, or document text is copied from the run records.
    """

    root = Path(run_dir).expanduser().resolve()
    run_paths = sorted(root.glob("run_*.json"))
    if not run_paths:
        raise ValueError(f"no run_*.json artifacts found in {root}")

    per_query: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    evidence_recalls: list[float] = []
    gold_recalls: list[float] = []
    completed_evidence_recalls: list[float] = []
    search_calls_values: list[float] = []
    unique_retrieved_values: list[float] = []
    format_valid_count = 0

    for path in run_paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        query_id = record.get("query_id")
        expected_query_id = path.name[len("run_") : -len(".json")]
        if not isinstance(query_id, str) or query_id != expected_query_id:
            raise ValueError(
                f"{path.name}: record query_id {query_id!r} does not match filename"
            )
        query = query_loader(query_id)
        status = record.get("status")
        if not isinstance(status, str) or not status:
            raise ValueError(f"{path.name}: record has no status")
        status_counts[status] = status_counts.get(status, 0) + 1

        retrieved = record.get("retrieved_docids")
        if not isinstance(retrieved, list):
            raise ValueError(f"{path.name}: record has no retrieved_docids list")
        evidence_recall = trajectory_evidence_recall(
            retrieved, query.evidence_document_ids
        )
        gold_recall = trajectory_evidence_recall(retrieved, query.gold_document_ids)
        evidence_recalls.append(evidence_recall)
        gold_recalls.append(gold_recall)
        if status == "completed":
            completed_evidence_recalls.append(evidence_recall)

        tool_call_counts = record.get("tool_call_counts") or {}
        search_calls = tool_call_counts.get("search")
        if isinstance(search_calls, (int, float)):
            search_calls_values.append(float(search_calls))
        unique_retrieved_values.append(float(len(set(retrieved))))

        diagnostics = record.get("diagnostics") or {}
        validation = diagnostics.get("final_answer_validation") or {}
        format_valid = validation.get("valid")
        if format_valid is True:
            format_valid_count += 1

        per_query.append(
            {
                "query_id": query_id,
                "status": status,
                "termination_reason": diagnostics.get("termination_reason"),
                "search_calls": search_calls,
                "unique_retrieved": len(set(retrieved)),
                "evidence_recall": evidence_recall,
                "gold_recall": gold_recall,
                "final_answer_format_valid": format_valid,
            }
        )

    return {
        "run_dir": str(root),
        "run_count": len(per_query),
        "status_counts": status_counts,
        "evidence_recall_mean": _mean(evidence_recalls),
        "gold_recall_mean": _mean(gold_recalls),
        "completed_evidence_recall_mean": _mean(completed_evidence_recalls),
        "search_calls_mean": _mean(search_calls_values),
        "unique_retrieved_mean": _mean(unique_retrieved_values),
        "final_answer_format_valid_count": format_valid_count,
        "per_query": per_query,
    }
