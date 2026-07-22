from __future__ import annotations

import json
import pytest

from rag_system.retrieval.remote_encoder import QueryEncoderError, RemoteQueryEncoder


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def test_remote_query_encoder_posts_query_and_validates_dimension(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return FakeResponse({"dimension": 3, "embedding": [0.1, 0.2, 0.3]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    encoder = RemoteQueryEncoder("http://encoder.test", timeout_seconds=9)
    assert encoder.encode(" query ") == [0.1, 0.2, 0.3]
    assert len(requests) == 1
    request, timeout = requests[0]
    assert request.full_url == "http://encoder.test/encode"
    assert json.loads(request.data) == {"query": "query"}
    assert timeout == 9


def test_remote_query_encoder_rejects_mismatched_dimension(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: FakeResponse({"dimension": 2, "embedding": [0.1]}),
    )
    encoder = RemoteQueryEncoder("http://encoder.test")
    with pytest.raises(QueryEncoderError, match="dimension"):
        encoder.encode("query")
