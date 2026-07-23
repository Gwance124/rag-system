from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rag_system.workflows.oss_standard_batch import (
    atomic_private_json,
    preflight_oss_standard_services,
    run_resumable_development_batch,
)


def record(query_id: str, status: str = "completed") -> dict:
    return {
        "schema_version": "1.0",
        "query_id": query_id,
        "tool_call_counts": {"search": 1},
        "status": status,
        "retrieved_docids": ["d1"],
        "result": [],
    }


def test_resumable_batch_skips_existing_and_retains_executor_errors(tmp_path):
    output_dir = tmp_path / "runs"
    atomic_private_json(output_dir / "run_q1.json", record("q1"))
    executed = []
    events = []

    def execute(query_id: str) -> dict:
        executed.append(query_id)
        if query_id == "q3":
            raise TimeoutError("generator timed out")
        return record(query_id)

    summary = run_resumable_development_batch(
        ("q1", "q2", "q3", "q4"),
        output_dir,
        execute,
        progress_callback=events.append,
    )

    assert executed == ["q2", "q3"]
    assert summary.to_dict() == {
        "output_dir": str(output_dir.resolve()),
        "total_query_count": 4,
        "skipped_existing_count": 1,
        "executed_count": 2,
        "remaining_query_count": 1,
        "stopped_on_error": True,
        "status_counts": {"completed": 2, "error": 1},
    }
    assert not (output_dir / "run_q4.json").exists()
    error_record = json.loads((output_dir / "run_q3.json").read_text())
    assert error_record["status"] == "error"
    assert error_record["error"] == {
        "type": "TimeoutError",
        "message": "generator timed out",
    }
    assert error_record["diagnostics"]["termination_reason"] == (
        "batch_query_executor_error"
    )
    assert os.stat(output_dir / "run_q2.json").st_mode & 0o777 == 0o600
    assert [event["event"] for event in events] == [
        "batch_started",
        "query_skipped",
        "query_started",
        "query_finished",
        "query_started",
        "query_finished",
        "batch_stopping",
        "batch_finished",
    ]


def test_resumable_batch_refuses_corrupt_existing_artifact(tmp_path):
    output_dir = tmp_path / "runs"
    output_dir.mkdir()
    (output_dir / "run_q1.json").write_text("not json", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot resume from invalid artifact"):
        run_resumable_development_batch(
            ("q1",),
            output_dir,
            lambda query_id: record(query_id),
        )


def test_resumable_batch_continues_after_incomplete_query(tmp_path):
    executed = []

    def execute(query_id: str) -> dict:
        executed.append(query_id)
        status = "incomplete" if query_id == "q1" else "completed"
        return record(query_id, status)

    summary = run_resumable_development_batch(
        ("q1", "q2"),
        tmp_path / "runs",
        execute,
    )

    assert executed == ["q1", "q2"]
    assert summary.stopped_on_error is False
    assert summary.remaining_query_count == 0
    assert summary.to_dict()["status_counts"] == {
        "completed": 1,
        "incomplete": 1,
    }


def test_resumable_batch_skips_existing_error_and_continues(tmp_path):
    output_dir = tmp_path / "runs"
    atomic_private_json(output_dir / "run_703.json", record("703", "error"))
    executed = []

    def execute(query_id: str) -> dict:
        executed.append(query_id)
        return record(query_id)

    summary = run_resumable_development_batch(
        ("703", "704"),
        output_dir,
        execute,
    )

    assert executed == ["704"]
    assert summary.stopped_on_error is False
    assert summary.remaining_query_count == 0
    assert summary.to_dict()["status_counts"] == {
        "completed": 1,
        "error": 1,
    }


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def test_batch_preflight_validates_search_contract_and_served_model(monkeypatch):
    requested_urls = []

    def fake_urlopen(request, timeout):
        requested_urls.append((request.full_url, timeout))
        if request.full_url == "http://search.test/health":
            return FakeResponse(
                {"status": "ok", "top_k": 5, "snippet_max_tokens": 512}
            )
        return FakeResponse({"data": [{"id": "openai/gpt-oss-20b"}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    preflight_oss_standard_services(
        "http://search.test",
        "http://generator.test/v1",
        "openai/gpt-oss-20b",
        timeout_seconds=3.0,
    )

    assert requested_urls == [
        ("http://search.test/health", 3.0),
        ("http://generator.test/v1/models", 3.0),
    ]


def test_batch_cli_rejects_nonstandard_development_count(tmp_path):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    (prepared_dir / "split.json").write_text(
        json.dumps({"development_query_ids": ["703"]}),
        encoding="utf-8",
    )
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_oss_standard_batch.py"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--prepared-dir",
            str(prepared_dir),
            "--generator-url",
            "http://generator.test/v1",
            "--output-dir",
            str(tmp_path / "runs"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "has 1 development queries; expected 100" in completed.stderr
