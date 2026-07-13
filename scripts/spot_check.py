import argparse
import pandas as pd
from chunking.tokenizer import HFTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Print full chunk sequences for a handful of papers (Stage 4)."
    )
    parser.add_argument("--chunks", required=True, help="Path to chunks.parquet")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--n-papers", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_parquet(args.chunks)
    tokenizer = HFTokenizer(args.tokenizer_path)

    sample_ids = df["id"].drop_duplicates().sample(
        n=min(args.n_papers, df["id"].nunique()), random_state=0
    )
    for paper_id in sample_ids:
        paper_chunks = df[df["id"] == paper_id].sort_values("chunk_index")
        print("=" * 80)
        print(f"paper: {paper_id}  ({len(paper_chunks)} chunks)")
        for _, chunk in paper_chunks.iterrows():
            tokens = tokenizer.count_tokens(chunk["text_with_context"])
            print("-" * 80)
            print(f"chunk {chunk['chunk_index']} | section: {chunk['section_path']} | tokens: {tokens}")
            print(chunk["text_raw"][:1000])
        print()


if __name__ == "__main__":
    main()
