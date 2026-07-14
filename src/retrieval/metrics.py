from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

LITSEARCH_PAPER_BM25 = {
    "inline-citation": {
        "broad": {"recall@20": 0.374},
        "specific": {"recall@5": 0.385, "recall@20": 0.558},
    },
    "author-written": {
        "broad": {"recall@20": 0.486},
        "specific": {"recall@5": 0.626, "recall@20": 0.735},
    },
    "average": {
        "broad": {"recall@20": 0.399},
        "specific": {"recall@5": 0.500},
    },
}

LITSEARCH_PAPER_NDCG10 = {
    "GTR-T5-large": {"broad": 0.233, "specific": 0.304},
    "Instructor-XL": {"broad": 0.328, "specific": 0.412},
    "E5-large-v2": {"broad": 0.271, "specific": 0.453},
    "GritLM-7B": {"broad": 0.441, "specific": 0.603},
}


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


def evaluate_litsearch_comparison(benchmark, run):
    groups = {}
    for query_id, metadata in (benchmark.query_metadata or {}).items():
        query_set = str(metadata.get("query_set", "unknown")).lower().replace("_", "-")
        if query_set.startswith("inline"):
            query_set = "inline-citation"
        elif query_set.startswith("author") or query_set.startswith("manual-"):
            query_set = "author-written"
        specificity = "broad" if int(metadata.get("specificity", 0)) == 0 else "specific"
        groups.setdefault((query_set, specificity), []).append(query_id)

    def evaluate(query_ids):
        query_ids = [query_id for query_id in query_ids if query_id in run and query_id in benchmark.qrels]
        return evaluate_run(
            {query_id: run[query_id] for query_id in query_ids},
            {query_id: benchmark.qrels[query_id] for query_id in query_ids},
            ks=(5, 20),
        )

    ours = {}
    for (query_set, specificity), query_ids in groups.items():
        ours.setdefault(query_set, {})[specificity] = {"queries": len(query_ids), **evaluate(query_ids)}
    for specificity in ("broad", "specific"):
        query_ids = [query_id for (query_set, kind), ids in groups.items() if kind == specificity for query_id in ids]
        if query_ids:
            ours.setdefault("average", {})[specificity] = {"queries": len(query_ids), **evaluate(query_ids)}

    deltas = {}
    for query_set, subsets in ours.items():
        if query_set not in LITSEARCH_PAPER_BM25:
            continue
        for specificity, metrics in subsets.items():
            reference = LITSEARCH_PAPER_BM25[query_set].get(specificity, {})
            matching = {key: metrics[key] - value for key, value in reference.items() if key in metrics}
            if matching:
                deltas.setdefault(query_set, {})[specificity] = matching
    return {
        "paper_bm25": LITSEARCH_PAPER_BM25,
        "paper_ndcg@10": LITSEARCH_PAPER_NDCG10,
        "ours": ours,
        "delta_vs_paper_bm25": deltas,
    }
