from __future__ import annotations

from collections.abc import Mapping, Sequence

from retrieval.types import Document, SearchHit


def rrf_fuse(
    rankings: Mapping[str, Sequence[SearchHit]],
    *,
    weights: Mapping[str, float] | None = None,
    rrf_k: int = 60,
    top_k: int | None = None,
) -> list[SearchHit]:
    """Fuse ranked lists with weighted reciprocal rank fusion."""
    scores: dict[str, float] = {}
    metadata: dict[str, str | None] = {}
    weights = weights or {name: 1.0 for name in rankings}

    for name, hits in rankings.items():
        weight = weights.get(name, 1.0)
        seen: set[str] = set()
        for rank, hit in enumerate(hits, 1):
            if hit.doc_id in seen:
                continue
            seen.add(hit.doc_id)
            scores[hit.doc_id] = scores.get(hit.doc_id, 0.0) + weight / (rrf_k + rank)
            if hit.doc_id not in metadata or metadata[hit.doc_id] is None:
                metadata[hit.doc_id] = hit.paper_id

    fused = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
    if top_k is not None:
        fused = fused[:top_k]
    return [SearchHit(doc_id, scores[doc_id], metadata[doc_id]) for doc_id in fused]


def aggregate_to_papers(
    hits: Sequence[SearchHit],
    documents: Mapping[str, Document],
    top_k: int | None = None,
) -> list[SearchHit]:
    """Collapse chunk hits to papers by taking each paper's best score."""
    best: dict[str, SearchHit] = {}
    for hit in hits:
        document = documents.get(hit.doc_id)
        paper_id = hit.paper_id or (document.paper_key if document else hit.doc_id)
        candidate = SearchHit(paper_id, hit.score, paper_id)
        if paper_id not in best or candidate.score > best[paper_id].score:
            best[paper_id] = candidate

    paper_ids = sorted(best, key=lambda paper_id: (-best[paper_id].score, paper_id))
    if top_k is not None:
        paper_ids = paper_ids[:top_k]
    return [best[paper_id] for paper_id in paper_ids]
