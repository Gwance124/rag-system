#!/usr/bin/env python3
"""Build the CP1 model/mode decision table from cached result JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


BENCHMARKS = (
    ("litsearch", "LitSearch"),
    ("mteb-scifact", "SciFact"),
    ("mteb-scidocs", "SciDocs"),
    ("mteb-nfcorpus", "NFCorpus"),
    ("mteb-trec-covid", "TREC-COVID"),
)
def _result_files(results_dir: Path):
    for path in sorted(results_dir.glob("*-sparse.json")):
        yield "BM25", "sparse", path
    sparse_dir = results_dir / "sparse"
    if sparse_dir.is_dir():
        for path in sorted(sparse_dir.glob("*.json")):
            yield "BM25", "sparse", path
    for model_dir in sorted(path for path in results_dir.iterdir() if path.is_dir() and path.name != "sparse"):
        for path in sorted(model_dir.glob("*-dense.json")):
            yield model_dir.name, "dense", path
        for path in sorted(model_dir.glob("*-hybrid.json")):
            yield model_dir.name, "hybrid", path


def _benchmark_name(path: Path) -> str:
    for suffix in ("-sparse.json", "-dense.json", "-hybrid.json"):
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def _variant_label(model: str, mode: str) -> str:
    return "BM25" if mode == "sparse" else f"{model} / {mode}"


def _metric(variant: dict, benchmark: str, name: str):
    return variant.get("benchmarks", {}).get(benchmark, {}).get("metrics", {}).get(name)


def _delta_values(variant: dict) -> dict[str, float]:
    comparison = variant.get("benchmarks", {}).get("litsearch", {}).get("litsearch_paper_comparison", {})
    values = {}
    for query_set, subsets in comparison.get("delta_vs_paper_bm25", {}).items():
        for specificity, metrics in subsets.items():
            for metric, value in metrics.items():
                values[f"{query_set}/{specificity}/{metric}"] = value
    return values


def build_report(results_dir: str | Path) -> dict:
    results_dir = Path(results_dir)
    variants = {}
    for model, mode, path in _result_files(results_dir):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read result file {path}: {exc}") from exc
        label = _variant_label(model, mode)
        variant = variants.setdefault(label, {"variant": label, "model": model, "mode": mode, "benchmarks": {}})
        variant["benchmarks"][_benchmark_name(path)] = data

    ordered = sorted(
        variants.values(),
        key=lambda row: (row["mode"] != "sparse", row["model"], row["mode"]),
    )
    eligible = [
        row
        for row in ordered
        if _metric(row, "litsearch", "recall@5") is not None
        and _metric(row, "litsearch", "recall@20") is not None
        and _metric(row, "mteb-scifact", "ndcg@10") is not None
    ]
    winner = max(
        eligible,
        key=lambda row: (
            _metric(row, "litsearch", "recall@5"),
            _metric(row, "litsearch", "recall@20"),
            _metric(row, "mteb-scifact", "ndcg@10"),
        ),
    )["variant"] if eligible else None
    return {
        "results_dir": str(results_dir),
        "winner_by_litsearch": winner,
        "variants": ordered,
    }


def _value(value) -> str:
    return "—" if value is None else f"{value:.3f}"


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def to_markdown(report: dict) -> str:
    variants = report["variants"]
    lines = [
        "# Checkpoint 1 benchmark report",
        "",
        "This is the CP1 title+abstract retrieval comparison. The winner is a mechanical recommendation by LitSearch R@5, then R@20, then SciFact nDCG@10; review the full table before freezing it.",
        "",
        f"Recommended winner: **{report['winner_by_litsearch'] or 'no complete variant yet'}**",
        "",
        "## Decision table",
        "",
    ]
    rows = []
    for variant in variants:
        rows.append([
            variant["variant"],
            _value(_metric(variant, "litsearch", "recall@5")),
            _value(_metric(variant, "litsearch", "recall@20")),
            _value(_metric(variant, "mteb-scifact", "ndcg@10")),
        ])
    lines.append(_table(["Variant", "LitSearch R@5", "LitSearch R@20", "SciFact nDCG@10"], rows))
    lines.extend(["", "## Recall@100 funnel coverage", ""])
    rows = []
    for variant in variants:
        rows.append([variant["variant"]] + [_value(_metric(variant, benchmark, "recall@100")) for benchmark, _ in BENCHMARKS])
    lines.append(_table(["Variant"] + [label for _, label in BENCHMARKS], rows))
    lines.extend(["", "## Sanity-check nDCG@10", ""])
    rows = []
    for variant in variants:
        rows.append([variant["variant"]] + [_value(_metric(variant, benchmark, "ndcg@10")) for benchmark, _ in BENCHMARKS[1:]])
    lines.append(_table(["Variant"] + [label for _, label in BENCHMARKS[1:]], rows))
    lines.extend(["", "## LitSearch deltas vs published BM25 references", ""])
    delta_keys = sorted({key for variant in variants for key in _delta_values(variant)})
    if delta_keys:
        rows = []
        for variant in variants:
            deltas = _delta_values(variant)
            rows.append([variant["variant"]] + [_value(deltas.get(key)) for key in delta_keys])
        lines.append(_table(["Variant"] + delta_keys, rows))
    else:
        lines.append("No LitSearch comparison sections were found in the result files.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Report the CP1 retrieval decision table from results/public.")
    parser.add_argument("--results-dir", default="results/public")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.results_dir)
    content = json.dumps(report, indent=2) + "\n" if args.format == "json" else to_markdown(report)
    if args.output:
        args.output.write_text(content)
    else:
        sys.stdout.write(content)


if __name__ == "__main__":
    main()
