from __future__ import annotations

import json

import pytest

from rag_system.retrieval.search_service import SearchServiceError, StandardSearchClient


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def standard_payload() -> dict:
    return {
        "top_k": 5,
        "snippet_max_tokens": 512,
        "hits": [
            {
                "docid": f"d{index}",
                "score": 1 - index / 10,
                "snippet": f"text {index}",
                "snippet_token_count": 2,
            }
            for index in range(5)
        ],
    }


def test_standard_search_client_hides_diagnostic_token_count_from_agent(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse(standard_payload())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = StandardSearchClient("http://search.test", timeout_seconds=7)
    hits = client.search(" query ")

    assert json.loads(captured["request"].data) == {"query": "query"}
    assert captured["timeout"] == 7
    assert hits[0] == {"docid": "d0", "score": 1.0, "snippet": "text 0"}
    assert all("snippet_token_count" not in hit for hit in hits)


def test_standard_search_client_rejects_nonstandard_response(monkeypatch):
    payload = standard_payload()
    payload["top_k"] = 10
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: FakeResponse(payload),
    )
    with pytest.raises(SearchServiceError, match="not Standard"):
        StandardSearchClient("http://search.test").search("query")
