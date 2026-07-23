"""Resume-safe sequential execution of the frozen OSS development split."""

from __future__ import annotations

import json
import os
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


BatchProgressCallback = Callable[[dict[str, Any]], None]
QueryExecutor = Callable[[str], dict[str, Any]]


@dataclass(frozen=True)
class OssDevelopmentBatchSummary:
    output_dir: str
    total_query_count: int
    skipped_existing_count: int
    executed_count: int
    remaining_query_count: int
    stopped_on_error: bool
    status_counts: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status_counts"] = dict(self.status_counts)
        return payload


def atomic_private_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a mode-0600 JSON artifact."""

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


def _output_path(output_dir: Path, query_id: str) -> Path:
    if (
        not query_id
        or query_id in {".", ".."}
        or "/" in query_id
        or "\\" in query_id
    ):
        raise ValueError(f"unsafe query ID {query_id!r}")
    return output_dir / f"run_{query_id}.json"


def _validate_record(record: Any, query_id: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"query {query_id}: runner returned a non-object record")
    if str(record.get("query_id")) != query_id:
        raise ValueError(f"query {query_id}: runner returned the wrong query ID")
    status = record.get("status")
    if not isinstance(status, str) or not status:
        raise ValueError(f"query {query_id}: runner returned no status")
    return record


def _read_existing_record(path: Path, query_id: str) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot resume from invalid artifact {path}") from exc
    return _validate_record(record, query_id)


def _get_json(url: str, timeout_seconds: float) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"service preflight failed for {url}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"service preflight returned a non-object for {url}")
    return payload


def preflight_oss_standard_services(
    search_url: str,
    generator_url: str,
    model: str,
    *,
    timeout_seconds: float = 10.0,
) -> None:
    """Fail before the batch if either Standard service is unavailable or wrong."""

    search_health_url = f"{search_url.rstrip('/')}/health"
    search_health = _get_json(search_health_url, timeout_seconds)
    if (
        search_health.get("status") != "ok"
        or search_health.get("top_k") != 5
        or search_health.get("snippet_max_tokens") != 512
    ):
        raise RuntimeError(
            "search preflight did not report the Standard top-5/512 contract"
        )

    models_url = f"{generator_url.rstrip('/')}/models"
    models = _get_json(models_url, timeout_seconds).get("data")
    served_model_ids: set[str] = set()
    if isinstance(models, list):
        served_model_ids = {
            item["id"]
            for item in models
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
    if model not in served_model_ids:
        raise RuntimeError(
            f"generator preflight did not find served model {model!r}"
        )


def run_resumable_development_batch(
    query_ids: Sequence[str],
    output_dir: str | Path,
    execute_query: QueryExecutor,
    *,
    progress_callback: BatchProgressCallback | None = None,
) -> OssDevelopmentBatchSummary:
    """Run every missing query once and retain an error row for executor failures."""

    ordered_ids = tuple(query_ids)
    if not ordered_ids or len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("query_ids must be a non-empty unique sequence")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    skipped_existing_count = 0
    executed_count = 0
    stopped_on_error = False
    status_counts: dict[str, int] = {}

    def progress(event: str, **details: Any) -> None:
        if progress_callback is not None:
            progress_callback({"event": event, **details})

    progress("batch_started", total_query_count=len(ordered_ids))
    for batch_index, query_id in enumerate(ordered_ids, start=1):
        path = _output_path(root, query_id)
        if path.exists():
            record = _read_existing_record(path, query_id)
            skipped_existing_count += 1
            status = record["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
            progress(
                "query_skipped",
                batch_index=batch_index,
                query_id=query_id,
                status=status,
                reason="existing_artifact",
            )
            continue

        progress(
            "query_started",
            batch_index=batch_index,
            query_id=query_id,
        )
        try:
            record = _validate_record(execute_query(query_id), query_id)
        except Exception as exc:
            record = {
                "schema_version": "1.0",
                "query_id": query_id,
                "tool_call_counts": {"search": 0},
                "status": "error",
                "retrieved_docids": [],
                "result": [],
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "diagnostics": {
                    "termination_reason": "batch_query_executor_error",
                },
            }
        atomic_private_json(path, record)
        executed_count += 1
        status = record["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
        progress(
            "query_finished",
            batch_index=batch_index,
            query_id=query_id,
            status=status,
            output_path=str(path),
        )
        if status == "error":
            stopped_on_error = True
            progress(
                "batch_stopping",
                batch_index=batch_index,
                query_id=query_id,
                reason="fresh_error_row",
            )
            break

    remaining_query_count = len(ordered_ids) - (
        skipped_existing_count + executed_count
    )
    summary = OssDevelopmentBatchSummary(
        output_dir=str(root),
        total_query_count=len(ordered_ids),
        skipped_existing_count=skipped_existing_count,
        executed_count=executed_count,
        remaining_query_count=remaining_query_count,
        stopped_on_error=stopped_on_error,
        status_counts=tuple(sorted(status_counts.items())),
    )
    progress("batch_finished", **summary.to_dict())
    return summary
