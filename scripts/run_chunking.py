import argparse
import pandas as pd
from chunking.tokenizer import HFTokenizer
from chunking.pipeline import run_chunking, write_chunks, write_failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-papers", required=True, help="Path to pilot_papers.parquet")
    parser.add_argument(
        "--tokenizer-path", required=True,
        help="Local path to the nv-embed-reason-3b tokenizer/model dir (vLLM's model dir)",
    )
    parser.add_argument("--chunks-output", required=True)
    parser.add_argument("--failures-output", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    df = pd.read_parquet(args.pilot_papers)
    tokenizer = HFTokenizer(args.tokenizer_path)
    records, failures = run_chunking(df, tokenizer, max_tokens=args.max_tokens)

    write_chunks(records, args.chunks_output)
    write_failures(failures, args.failures_output)

    print(f"Wrote {len(records)} chunks to {args.chunks_output}")
    print(f"Logged {len(failures)} parse failures to {args.failures_output}")


if __name__ == "__main__":
    main()
