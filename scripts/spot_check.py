import argparse
import json
import pandas as pd
from chunking.tokenizer import HFTokenizer
from chunking.stats import (
    estimate_tokens_from_chars,
    summarize_tokens,
    paper_cleaned_token_count,
    build_token_stats_report,
)


def main():
    parser = argparse.ArgumentParser(
        description="Print full chunk sequences for a handful of papers (Stage 4)."
    )
    parser.add_argument("--chunks", required=True, help="Path to chunks.parquet")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--n-papers", type=int, default=5)
    parser.add_argument(
        "--pilot-papers",
        help="Path to pilot_papers.parquet - if given, also compares raw (uncleaned) "
        "latex token estimate vs cleaned chunk token usage per paper",
    )
    parser.add_argument(
        "--plot-output", default="token_stats.png",
        help="Where to write the raw-vs-cleaned token stats bar chart (requires --pilot-papers)",
    )
    parser.add_argument(
        "--stats-output", default="token_stats.json",
        help="Where to write per-paper raw/cleaned token counts and summary stats as JSON "
        "(requires --pilot-papers), so this data can be reviewed without rerunning the script",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.chunks)
    tokenizer = HFTokenizer(args.tokenizer_path)

    pilot_df = None
    if args.pilot_papers:
        pilot_df = pd.read_parquet(args.pilot_papers).set_index("id")

    sample_ids = df["id"].drop_duplicates().sample(
        n=min(args.n_papers, df["id"].nunique()), random_state=0
    )

    chunk_token_counts = []
    raw_token_estimates = []
    cleaned_token_counts = []
    sampled_paper_ids = []

    for paper_id in sample_ids:
        sampled_paper_ids.append(paper_id)
        paper_chunks = df[df["id"] == paper_id].sort_values("chunk_index")
        print("=" * 80)
        print(f"paper: {paper_id}  ({len(paper_chunks)} chunks)")
        for _, chunk in paper_chunks.iterrows():
            tokens = tokenizer.count_tokens(chunk["text_with_context"])
            chunk_token_counts.append(tokens)
            print("-" * 80)
            print(f"chunk {chunk['chunk_index']} | section: {chunk['section_path']} | tokens: {tokens}")
            print(chunk["text_raw"][:1000])
        print()

        cleaned_token_counts.append(
            paper_cleaned_token_count(list(paper_chunks["text_raw"]), tokenizer)
        )
        if pilot_df is not None:
            raw_latex = pilot_df.loc[paper_id, "latex"]
            raw_token_estimates.append(estimate_tokens_from_chars(raw_latex))

    chunk_stats = summarize_tokens(chunk_token_counts)
    print("=" * 80)
    print(f"chunks sampled: {len(chunk_token_counts)}")
    print(f"total tokens: {chunk_stats['total']}")
    print(f"avg tokens/chunk: {chunk_stats['avg']:.1f}")

    if pilot_df is not None:
        raw_stats = summarize_tokens(raw_token_estimates)
        cleaned_stats = summarize_tokens(cleaned_token_counts)
        print()
        print("per-paper token usage: raw (uncleaned, char/4 estimate) vs cleaned (actual, post-parse)")
        print(f"{'':10}{'total':>12}{'avg':>12}{'median':>12}{'min':>12}{'max':>12}")
        print(
            f"{'raw':10}"
            f"{raw_stats['total']:>12}{raw_stats['avg']:>12.1f}{raw_stats['median']:>12.1f}"
            f"{raw_stats['min']:>12}{raw_stats['max']:>12}"
        )
        print(
            f"{'cleaned':10}"
            f"{cleaned_stats['total']:>12}{cleaned_stats['avg']:>12.1f}{cleaned_stats['median']:>12.1f}"
            f"{cleaned_stats['min']:>12}{cleaned_stats['max']:>12}"
        )

        plot_raw_vs_cleaned(raw_stats, cleaned_stats, args.plot_output)
        print(f"\nWrote token stats chart to {args.plot_output}")

        report = build_token_stats_report(
            paper_ids=sampled_paper_ids,
            raw_token_estimates=raw_token_estimates,
            cleaned_token_counts=cleaned_token_counts,
            chunk_token_counts=chunk_token_counts,
        )
        with open(args.stats_output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote token stats data to {args.stats_output}")


def plot_raw_vs_cleaned(raw_stats: dict, cleaned_stats: dict, output_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["avg", "median", "min", "max"]
    raw_values = [raw_stats[m] for m in metrics]
    cleaned_values = [cleaned_stats[m] for m in metrics]

    x = range(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], raw_values, width, label="raw (estimate)")
    ax.bar([i + width / 2 for i in x], cleaned_values, width, label="cleaned (actual)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylabel("tokens per paper")
    ax.set_title("Raw vs cleaned per-paper token usage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)


if __name__ == "__main__":
    main()
