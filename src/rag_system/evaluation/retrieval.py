"""Retrieval metrics used by the BrowseComp-Plus agent leaderboard."""

from __future__ import annotations

from collections.abc import Iterable


def trajectory_evidence_recall(
    retrieved_document_ids: Iterable[str],
    evidence_document_ids: Iterable[str],
) -> float:
    """Return recall over all unique documents retrieved during one trajectory.

    This is intentionally not named ``recall_at_k``: the official agent
    evaluator takes the union of documents from every search call. The number
    of unique retrieved documents therefore varies with the trajectory.
    """
    retrieved = set(retrieved_document_ids)
    evidence = set(evidence_document_ids)
    if not evidence:
        raise ValueError("evidence_document_ids must not be empty")
    return len(retrieved & evidence) / len(evidence)


def mean_trajectory_evidence_recall(
    trajectories: Iterable[tuple[Iterable[str], Iterable[str]]],
) -> float:
    """Macro-average trajectory evidence recall across benchmark queries."""
    recalls = [
        trajectory_evidence_recall(retrieved, evidence)
        for retrieved, evidence in trajectories
    ]
    if not recalls:
        raise ValueError("at least one trajectory is required")
    return sum(recalls) / len(recalls)
