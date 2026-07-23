#!/usr/bin/env python3
"""Summarize a Standard agent run directory into leaderboard-style recall."""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

from rag_system.datasets.browsecomp_plus import load_prepared_development_query
from rag_system.evaluation.run_summary import summarize_run_directory
from rag_system.workflows.oss_standard_batch import atomic_private_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the private JSON summary artifact",
    )
    args = parser.parse_args()

    query_loader = functools.partial(
        load_prepared_development_query, args.prepared_dir
    )
    summary = summarize_run_directory(args.run_dir, query_loader)
    if args.output is not None:
        atomic_private_json(args.output.expanduser().resolve(), summary)

    headline = {
        key: summary[key]
        for key in (
            "run_dir",
            "run_count",
            "status_counts",
            "evidence_recall_mean",
            "gold_recall_mean",
            "completed_evidence_recall_mean",
            "search_calls_mean",
            "unique_retrieved_mean",
            "final_answer_format_valid_count",
        )
    }
    print(json.dumps(headline, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
