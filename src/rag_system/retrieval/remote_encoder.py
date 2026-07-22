"""Dependency-light client for the private g3 query-encoding service."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass


class QueryEncoderError(RuntimeError):
    """Raised when the remote query encoder returns an invalid response."""


@dataclass(frozen=True)
class RemoteQueryEncoder:
    base_url: str
    timeout_seconds: float = 120.0

    def encode(self, query: str) -> list[float]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        body = json.dumps({"query": normalized_query}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/encode",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
            raise QueryEncoderError(f"query encoder request failed: {exc}") from exc

        embedding = payload.get("embedding") if isinstance(payload, dict) else None
        if not isinstance(embedding, list) or not embedding:
            raise QueryEncoderError("query encoder response has no embedding")
        if not all(isinstance(value, (int, float)) for value in embedding):
            raise QueryEncoderError("query encoder returned a non-numeric embedding")
        declared_dimension = payload.get("dimension")
        if declared_dimension is not None and declared_dimension != len(embedding):
            raise QueryEncoderError(
                "query encoder dimension does not match the embedding length"
            )
        return [float(value) for value in embedding]
