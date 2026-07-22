"""Index-agnostic sparse, dense, and hybrid query orchestration."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable

from old.src.retrieval.fusion import rrf_fuse
from old.src.retrieval.types import RetrievalConfig, RetrievalResult, SearchHit


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


class HybridRetriever:
    def __init__(
        self,
        *,
        sparse_index=None,
        dense_index=None,
        config: RetrievalConfig | None = None,
        query_rewriter: Callable[[str], str] | None = None,
    ) -> None:
        self.sparse_index = sparse_index
        self.dense_index = dense_index
        self.config = config or RetrievalConfig()
        self.query_rewriter = query_rewriter
        if sparse_index is None and dense_index is None:
            raise ValueError("at least one retrieval index is required")

    def search(
        self,
        query: str,
        excluded_ids: Iterable[str] = (),
        allowed_ids: Iterable[str] | None = None,
    ) -> RetrievalResult:
        timings: dict[str, float] = {}
        allowed = None if allowed_ids is None else set(allowed_ids)
        rewritten = query
        if self.config.rewrite:
            if self.query_rewriter is None:
                raise ValueError("rewrite=True requires query_rewriter")
            start = time.perf_counter()
            rewritten = self.query_rewriter(query)
            timings["rewrite_ms"] = _elapsed_ms(start)

        rankings: dict[str, list[SearchHit]] = {}
        if self.sparse_index is not None:
            start = time.perf_counter()
            if allowed is None:
                rankings["sparse"] = self.sparse_index.search(rewritten, self.config.top_n)
            else:
                rankings["sparse"] = self.sparse_index.search(
                    rewritten,
                    self.config.top_n,
                    allowed,
                )
            timings["sparse_search_ms"] = _elapsed_ms(start)

        if self.dense_index is not None:
            start = time.perf_counter()
            if hasattr(self.dense_index, "embed_query") and hasattr(self.dense_index, "search_vector"):
                vector = self.dense_index.embed_query(rewritten)
                timings["embed_ms"] = _elapsed_ms(start)
                start = time.perf_counter()
                if allowed is None:
                    rankings["dense"] = self.dense_index.search_vector(vector, self.config.top_n)
                else:
                    rankings["dense"] = self.dense_index.search_vector(
                        vector,
                        self.config.top_n,
                        allowed,
                    )
                timings["dense_search_ms"] = _elapsed_ms(start)
            else:
                if allowed is None:
                    rankings["dense"] = self.dense_index.search(rewritten, self.config.top_n)
                else:
                    rankings["dense"] = self.dense_index.search(
                        rewritten,
                        self.config.top_n,
                        allowed,
                    )
                timings["dense_search_ms"] = _elapsed_ms(start)

        start = time.perf_counter()
        fused = rrf_fuse(
            rankings,
            weights={"sparse": self.config.sparse_weight, "dense": self.config.dense_weight},
            rrf_k=self.config.rrf_k,
        )
        excluded = set(excluded_ids)
        hits = [
            hit
            for hit in fused
            if hit.doc_id not in excluded and (allowed is None or hit.doc_id in allowed)
        ][: self.config.top_k]
        timings["fuse_ms"] = _elapsed_ms(start)
        return RetrievalResult(hits=hits, timings_ms=timings)
