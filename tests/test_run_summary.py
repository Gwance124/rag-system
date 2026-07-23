"""Tests for aggregating agent run directories into leaderboard-style metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_system.contracts import BenchmarkQuery
from rag_system.evaluation.run_summary import summarize_run_directory


def _query(query_id: str, evidence: list[str], gold: list[str]) -> BenchmarkQuery:
    return BenchmarkQuery(
        query_id=query_id,
        question=f"question {query_id}",
        reference_answer="answer",
        evidence_document_ids=tuple(evidence),
        gold_document_ids=tuple(gold),
    )


def _write_run(
    run_dir: Path,
    query_id: str,
    *,
    status: str,
    retrieved_docids: list[str],
    search_calls: int,
    format_valid: bool | None = True,
) -> None:
    validation = None if format_valid is None else {"valid": format_valid}
    record = {
        "schema_version": "1.0",
        "query_id": query_id,
        "tool_call_counts": {"search": search_calls},
        "status": status,
        "retrieved_docids": retrieved_docids,
        "result": [],
        "diagnostics": {
            "termination_reason": "final_answer",
            "final_answer_validation": validation,
        },
    }
    path = run_dir / f"run_{query_id}.json"
    path.write_text(json.dumps(record), encoding="utf-8")


@pytest.fixture
def queries() -> dict[str, BenchmarkQuery]:
    return {
        "q1": _query("q1", evidence=["d1", "d2", "d3", "d4"], gold=["d1", "d2"]),
        "q2": _query("q2", evidence=["d5", "d6"], gold=["d5"]),
    }


def test_summarize_run_directory_macro_averages_trajectory_recall(
    tmp_path: Path, queries: dict[str, BenchmarkQuery]
) -> None:
    _write_run(
        tmp_path,
        "q1",
        status="completed",
        retrieved_docids=["d1", "d2", "d9"],
        search_calls=4,
    )
    _write_run(
        tmp_path,
        "q2",
        status="incomplete",
        retrieved_docids=["d6"],
        search_calls=2,
        format_valid=None,
    )

    summary = summarize_run_directory(tmp_path, queries.__getitem__)

    assert summary["run_count"] == 2
    assert summary["status_counts"] == {"completed": 1, "incomplete": 1}
    # q1: 2/4 evidence, q2: 1/2 evidence -> macro mean 0.5
    assert summary["evidence_recall_mean"] == pytest.approx(0.5)
    # q1: 2/2 gold, q2: 0/1 gold -> macro mean 0.5
    assert summary["gold_recall_mean"] == pytest.approx(0.5)
    assert summary["completed_evidence_recall_mean"] == pytest.approx(0.5)
    assert summary["search_calls_mean"] == pytest.approx(3.0)
    assert summary["unique_retrieved_mean"] == pytest.approx(2.0)
    assert summary["final_answer_format_valid_count"] == 1

    per_query = {row["query_id"]: row for row in summary["per_query"]}
    assert per_query["q1"]["evidence_recall"] == pytest.approx(0.5)
    assert per_query["q1"]["gold_recall"] == pytest.approx(1.0)
    assert per_query["q2"]["status"] == "incomplete"
    # Decrypted benchmark content must never leak into the summary artifact.
    assert "question" not in json.dumps(summary)


def test_summarize_run_directory_recomputes_format_validity_from_answer_text(
    tmp_path: Path, queries: dict[str, BenchmarkQuery]
) -> None:
    # A record saved by an older runner build may carry a stale validation
    # verdict; the summary must re-validate the stored final answer text.
    record = {
        "schema_version": "1.0",
        "query_id": "q1",
        "tool_call_counts": {"search": 2},
        "status": "completed",
        "retrieved_docids": ["d1"],
        "result": [
            {
                "type": "output_text",
                "tool_name": None,
                "arguments": None,
                "output": (
                    "**Explanation**: stated in the document [d1].\n"
                    "**Exact Answer**: blue\n"
                    "**Confidence**: 90%"
                ),
            }
        ],
        "diagnostics": {"final_answer_validation": {"valid": False}},
    }
    (tmp_path / "run_q1.json").write_text(json.dumps(record), encoding="utf-8")

    summary = summarize_run_directory(tmp_path, queries.__getitem__)

    assert summary["final_answer_format_valid_count"] == 1
    assert summary["per_query"][0]["final_answer_format_valid"] is True


def test_summarize_run_directory_rejects_mismatched_query_ids(
    tmp_path: Path, queries: dict[str, BenchmarkQuery]
) -> None:
    record = {
        "query_id": "q1",
        "status": "completed",
        "retrieved_docids": [],
        "tool_call_counts": {"search": 0},
    }
    (tmp_path / "run_q2.json").write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="query_id"):
        summarize_run_directory(tmp_path, queries.__getitem__)


def test_summarize_run_directory_requires_run_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no run_"):
        summarize_run_directory(tmp_path, lambda query_id: None)
