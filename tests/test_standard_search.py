from __future__ import annotations

import pytest

from rag_system.contracts import SearchCandidate
from rag_system.retrieval.standard import (
    StandardSearchContractError,
    StandardSearchTool,
)


class CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
    ) -> str:
        assert skip_special_tokens is True
        return "".join(chr(token_id) for token_id in token_ids)


class FakeBackend:
    def __init__(self, candidates: list[SearchCandidate]) -> None:
        self.candidates = candidates
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> list[SearchCandidate]:
        self.calls.append((query, top_k))
        return self.candidates[:top_k]


def candidates(count: int = 5) -> list[SearchCandidate]:
    return [
        SearchCandidate(
            document_id=f"d{index}",
            score=1.0 - index / 10,
            text=("x" * 600 if index == 0 else f"document {index}"),
        )
        for index in range(count)
    ]


def test_standard_search_returns_five_unique_prefix_snippets():
    backend = FakeBackend(candidates())
    tool = StandardSearchTool(backend, CharacterTokenizer())

    trace = tool.search("  dynamic query  ")

    assert backend.calls == [("dynamic query", 5)]
    assert trace.top_k == 5
    assert trace.snippet_max_tokens == 512
    assert [hit.document_id for hit in trace.hits] == ["d0", "d1", "d2", "d3", "d4"]
    assert trace.hits[0].snippet == "x" * 512
    assert trace.hits[0].snippet_token_count == 512
    assert all(hit.snippet_token_count <= 512 for hit in trace.hits)


def test_standard_search_rejects_short_backend_result():
    tool = StandardSearchTool(FakeBackend(candidates(4)), CharacterTokenizer())
    with pytest.raises(StandardSearchContractError, match="expected exactly 5"):
        tool.search("query")


def test_standard_search_rejects_duplicate_documents():
    duplicate = candidates()
    duplicate[-1] = SearchCandidate("d0", 0.1, "duplicate")
    tool = StandardSearchTool(FakeBackend(duplicate), CharacterTokenizer())
    with pytest.raises(StandardSearchContractError, match="duplicate document ID"):
        tool.search("query")
