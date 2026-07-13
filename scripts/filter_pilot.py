import argparse
import glob
import pandas as pd
from chunking.pipeline import filter_pilot_papers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-glob", required=True,
        help="Glob pattern for source parquet files, e.g. 'data/part-*.parquet'",
    )
    parser.add_argument("--output", required=True, help="Output path for pilot_papers.parquet")
    parser.add_argument("--category", default="cs.IR")
    parser.add_argument("--min-year", type=int, default=2020)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.input_glob))
    filtered_frames = []
    kept_so_far = 0

    for i, path in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] reading {path}")
        df = pd.read_parquet(path)
        filtered = filter_pilot_papers(df, category=args.category, min_year=args.min_year)
        del df
        filtered_frames.append(filtered)
        kept_so_far += len(filtered)
        print(f"[{i}/{len(paths)}]   kept {len(filtered)} rows (running total: {kept_so_far})")

    pilot = pd.concat(filtered_frames, ignore_index=True)
    pilot.to_parquet(args.output, index=False)
    print(f"Wrote {len(pilot)} papers to {args.output}")


if __name__ == "__main__":
    main()
