"""BrowseComp-Plus Standard search-tool boundary.

The underlying retriever ranks complete corpus documents. This adapter owns
the separately defined agent-facing contract: five unique documents and a
prefix of at most 512 tokens from each document.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from rag_system.contracts import (
    SearchCandidate,
    StandardSearchHit,
    StandardSearchTrace,
)


STANDARD_TOP_K = 5
STANDARD_SNIPPET_MAX_TOKENS = 512


class SearchBackend(Protocol):
    """Document-level retriever used behind the Standard tool."""

    def search(self, query: str, top_k: int) -> Sequence[SearchCandidate]: ...


class SnippetTokenizer(Protocol):
    """Subset of the Hugging Face tokenizer interface used for snippets."""

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool,
    ) -> str: ...


class StandardSearchContractError(ValueError):
    """Raised when a backend result cannot be exposed as a Standard result."""


class StandardSearchTool:
    """Validate and truncate one dynamic document-level search response."""

    def __init__(
        self,
        backend: SearchBackend,
        tokenizer: SnippetTokenizer,
        *,
        top_k: int = STANDARD_TOP_K,
        snippet_max_tokens: int = STANDARD_SNIPPET_MAX_TOKENS,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if snippet_max_tokens <= 0:
            raise ValueError("snippet_max_tokens must be positive")
        self.backend = backend
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.snippet_max_tokens = snippet_max_tokens

    def search(self, query: str) -> StandardSearchTrace:
        normalized_query = query.strip()
        if not normalized_query:
            raise StandardSearchContractError("search query must not be empty")

        candidates = tuple(self.backend.search(normalized_query, self.top_k))
        if len(candidates) != self.top_k:
            raise StandardSearchContractError(
                f"expected exactly {self.top_k} candidates, found {len(candidates)}"
            )

        seen_document_ids: set[str] = set()
        hits = []
        for candidate in candidates:
            if not candidate.document_id:
                raise StandardSearchContractError("candidate has an empty document ID")
            if candidate.document_id in seen_document_ids:
                raise StandardSearchContractError(
                    f"duplicate document ID in search result: {candidate.document_id}"
                )
            if not math.isfinite(candidate.score):
                raise StandardSearchContractError(
                    f"candidate {candidate.document_id} has a non-finite score"
                )
            if not isinstance(candidate.text, str):
                raise StandardSearchContractError(
                    f"candidate {candidate.document_id} has invalid text"
                )

            snippet, token_count = self._prefix_snippet(candidate.text)
            hits.append(
                StandardSearchHit(
                    document_id=candidate.document_id,
                    score=candidate.score,
                    snippet=snippet,
                    snippet_token_count=token_count,
                )
            )
            seen_document_ids.add(candidate.document_id)

        return StandardSearchTrace(
            query=normalized_query,
            top_k=self.top_k,
            snippet_max_tokens=self.snippet_max_tokens,
            hits=tuple(hits),
        )

    def _prefix_snippet(self, text: str) -> tuple[str, int]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= self.snippet_max_tokens:
            return text, len(token_ids)

        prefix_length = self.snippet_max_tokens
        while prefix_length > 0:
            snippet = self.tokenizer.decode(
                token_ids[:prefix_length],
                skip_special_tokens=True,
            )
            rendered_count = len(
                self.tokenizer.encode(snippet, add_special_tokens=False)
            )
            if rendered_count <= self.snippet_max_tokens:
                return snippet, rendered_count
            prefix_length -= 1

        return "", 0
