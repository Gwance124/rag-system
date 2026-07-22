from __future__ import annotations

import pytest

from rag_system.evaluation.retrieval import (
    mean_trajectory_evidence_recall,
    trajectory_evidence_recall,
)


def test_trajectory_recall_uses_union_across_calls_not_fixed_k():
    retrieved_across_calls = ["d1", "d2", "d2", "d8", "d9", "d3"]
    evidence = ["d1", "d3", "d7"]
    assert trajectory_evidence_recall(retrieved_across_calls, evidence) == pytest.approx(
        2 / 3
    )


def test_mean_trajectory_recall_is_macro_average():
    trajectories = [
        (["d1"], ["d1"]),
        (["d2"], ["d2", "d3", "d4"]),
    ]
    assert mean_trajectory_evidence_recall(trajectories) == pytest.approx((1 + 1 / 3) / 2)


def test_trajectory_recall_requires_evidence_labels():
    with pytest.raises(ValueError, match="must not be empty"):
        trajectory_evidence_recall(["d1"], [])
