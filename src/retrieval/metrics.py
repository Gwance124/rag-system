from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence


def recall_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    return len(set(ranking[:k]) & relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank(ranking: Sequence[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(ranking, 1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranking: Sequence[str], relevant: set[str], k: int = 10) -> float:
    if not relevant:
        return 0.0

    dcg = sum(
        (1.0 if doc_id in relevant else 0.0) / math.log2(rank + 2)
        for rank, doc_id in enumerate(ranking[:k])
    )
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def evaluate_run(
    run: Mapping[str, Sequence[str]],
    qrels: Mapping[str, set[str]],
    ks: Iterable[int] = (10, 50, 100),
) -> dict[str, float]:
    """Return macro-averaged paper-level retrieval metrics."""
    query_ids = [query_id for query_id in qrels if query_id in run]
    if not query_ids:
        return {f"recall@{k}": 0.0 for k in ks} | {"ndcg@10": 0.0, "mrr": 0.0}

    ks = tuple(ks)
    metrics = {
        f"recall@{k}": sum(recall_at_k(run[q], qrels[q], k) for q in query_ids) / len(query_ids)
        for k in ks
    }
    metrics["ndcg@10"] = sum(ndcg_at_k(run[q], qrels[q], 10) for q in query_ids) / len(query_ids)
    metrics["mrr"] = sum(reciprocal_rank(run[q], qrels[q]) for q in query_ids) / len(query_ids)
    return metrics
