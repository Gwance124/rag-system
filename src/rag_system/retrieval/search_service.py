"""Client for the persistent p7 Standard search service."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class SearchServiceError(RuntimeError):
    """Raised when the Standard search service violates its contract."""


@dataclass(frozen=True)
class StandardSearchClient:
    base_url: str
    timeout_seconds: float = 180.0

    def search(self, query: str) -> list[dict[str, Any]]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/search",
            data=json.dumps({"query": normalized_query}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
            raise SearchServiceError(f"search request failed: {exc}") from exc

        if not isinstance(payload, dict):
            raise SearchServiceError("search response is not an object")
        if payload.get("top_k") != 5 or payload.get("snippet_max_tokens") != 512:
            raise SearchServiceError("search response is not Standard top-5/512")
        hits = payload.get("hits")
        if not isinstance(hits, list) or len(hits) != 5:
            raise SearchServiceError("search response must contain exactly five hits")

        seen: set[str] = set()
        agent_hits = []
        for hit in hits:
            if not isinstance(hit, dict):
                raise SearchServiceError("search hit is not an object")
            docid = hit.get("docid")
            score = hit.get("score")
            snippet = hit.get("snippet")
            token_count = hit.get("snippet_token_count")
            if not isinstance(docid, str) or not docid or docid in seen:
                raise SearchServiceError("search hit has an invalid or duplicate docid")
            if not isinstance(score, (int, float)):
                raise SearchServiceError("search hit has an invalid score")
            if not isinstance(snippet, str):
                raise SearchServiceError("search hit has an invalid snippet")
            if not isinstance(token_count, int) or not 0 <= token_count <= 512:
                raise SearchServiceError("search hit exceeds the 512-token contract")
            seen.add(docid)
            agent_hits.append(
                {"docid": docid, "score": float(score), "snippet": snippet}
            )
        return agent_hits
