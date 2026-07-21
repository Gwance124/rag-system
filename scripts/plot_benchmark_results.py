#!/usr/bin/env python3
"""Plot one model/pipeline comparison chart per benchmark dataset."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_METRICS = ("recall@5", "recall@20", "recall@50", "ndcg@10")


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", value.lower()).strip("-")


def _benchmark_key(data: dict, path: Path) -> str:
    config = data.get("config", {})
    benchmark = config.get("benchmark")
    dataset = config.get("dataset")
    if benchmark in {"mteb", "beir"} and dataset:
        return f"mteb-{dataset}"
    if benchmark == "bright" and config.get("domain"):
        return f"bright-{config['domain']}"
    if benchmark == "qasper":
        scope = config.get("qasper_scope")
        if scope in {"global", "paper", "two-stage"}:
            return f"qasper-{scope}"
    if benchmark:
        return str(benchmark)
    return re.sub(r"-(?:sparse|dense|hybrid)$", "", path.stem)


def _variant(data: dict, path: Path) -> tuple[str, str, str]:
    config = data.get("config", {})
    mode = str(config.get("mode") or "unknown")
    if mode == "sparse":
        model = "BM25"
    else:
        model = str(config.get("embedding_model") or path.parent.name or "unknown")
        model = model.rsplit("/", 1)[-1]
    label = f"{model} / {mode}"
    if config.get("benchmark") == "qasper" and config.get("qasper_scope") == "two-stage":
        paper_top_k = config.get("qasper_paper_top_k")
        if paper_top_k is not None:
            label = f"{label} / paper-k={paper_top_k}"
    return model, mode, label


def load_results(results_dir: str | Path) -> dict[str, list[dict]]:
    """Load result JSONs and group them by benchmark, model, and pipeline."""
    root = Path(results_dir)
    grouped: dict[tuple[str, str], dict] = {}
    for path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or not isinstance(data.get("metrics"), dict):
            continue
        benchmark = _benchmark_key(data, path)
        model, mode, label = _variant(data, path)
        row = {
            "benchmark": benchmark,
            "model": model,
            "mode": mode,
            "label": label,
            "metrics": data["metrics"],
            "source": str(path),
        }
        grouped[(benchmark, label)] = row

        if benchmark == "litsearch":
            averages = (
                data.get("litsearch_paper_comparison", {})
                .get("ours", {})
                .get("average", {})
            )
            for specificity in ("broad", "specific"):
                metrics = averages.get(specificity)
                if not isinstance(metrics, dict):
                    continue
                subset = f"litsearch-average-{specificity}"
                grouped[(subset, label)] = {
                    **row,
                    "benchmark": subset,
                    "metrics": metrics,
                }

        if benchmark == "qasper-two-stage":
            metrics = (
                data.get("qasper", {})
                .get("paper_retrieval", {})
                .get("metrics")
            )
            if isinstance(metrics, dict):
                subset = "qasper-two-stage-papers"
                paper_label = f"{model} / {mode}"
                existing = grouped.get((subset, paper_label))
                grouped[(subset, paper_label)] = {
                    **row,
                    "benchmark": subset,
                    "label": paper_label,
                    "metrics": {
                        **(existing["metrics"] if existing else {}),
                        **metrics,
                    },
                }

    output: dict[str, list[dict]] = {}
    for row in sorted(grouped.values(), key=lambda item: (item["benchmark"], item["label"])):
        output.setdefault(row["benchmark"], []).append(row)
    return output


def plot_results(
    results: dict[str, list[dict]],
    output_dir: str | Path,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    *,
    file_format: str = "png",
    dpi: int = 160,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for benchmark, rows in results.items():
        available = [metric for metric in metrics if any(metric in row["metrics"] for row in rows)]
        if not available:
            continue

        figure, axis = plt.subplots(figsize=(max(8, 1.35 * len(rows) + 2), 5.5))
        positions = list(range(len(rows)))
        width = 0.8 / len(available)
        for index, metric in enumerate(available):
            values = [row["metrics"].get(metric, float("nan")) for row in rows]
            offsets = [position + (index - (len(available) - 1) / 2) * width for position in positions]
            bars = axis.bar(offsets, values, width=width, label=metric)
            for bar, value in zip(bars, values):
                if value == value:
                    axis.text(
                        bar.get_x() + bar.get_width() / 2,
                        value + 0.015,
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=90,
                    )

        axis.set_title(f"{benchmark}: model / pipeline comparison")
        axis.set_ylabel("Score")
        axis.set_ylim(0, 1.08)
        axis.set_xticks(positions, [row["label"] for row in rows], rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
        figure.tight_layout()
        path = output / f"{_safe_name(benchmark)}.{file_format}"
        figure.savefig(path, dpi=dpi)
        plt.close(figure)
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot one comparison graph per benchmark dataset.")
    parser.add_argument("--results-dir", default="results/public")
    parser.add_argument("--output-dir", default="results/plots")
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="comma-separated metrics (default: recall@5,recall@20,recall@50,ndcg@10)",
    )
    parser.add_argument("--format", choices=("png", "svg"), default="png")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()
    metrics = tuple(metric.strip() for metric in args.metrics.split(",") if metric.strip())
    written = plot_results(load_results(args.results_dir), args.output_dir, metrics, file_format=args.format, dpi=args.dpi)
    if not written:
        parser.error(f"no benchmark result JSONs with the requested metrics found under {args.results_dir}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
