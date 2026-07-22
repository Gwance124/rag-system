import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Print raw `latex` field samples for manual inspection (Stage 1)."
    )
    parser.add_argument("--pilot-papers", required=True)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--chars", type=int, default=3000, help="Max characters to print per sample")
    args = parser.parse_args()

    df = pd.read_parquet(args.pilot_papers)
    sample = df.sample(n=min(args.n, len(df)), random_state=0)
    for _, row in sample.iterrows():
        print("=" * 80)
        print(f"id: {row['id']}  title: {row['title']}")
        print("-" * 80)
        print(row["latex"][: args.chars])
        print()


if __name__ == "__main__":
    main()
