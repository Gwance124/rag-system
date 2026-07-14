import argparse
import pandas as pd
from chunking.tokenizer import HFTokenizer
from chunking.latex_parse import _split_into_files, _BEGIN_DOCUMENT_RE


def main():
    parser = argparse.ArgumentParser(
        description="Dump bundle structure for a paper and raw content for a chunk - "
        "defaults to auto-finding the current worst offenders by token count."
    )
    parser.add_argument("--pilot-papers", default="pilot_papers.parquet")
    parser.add_argument("--chunks", default="chunks.parquet")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--paper-id", help="Defaults to the paper with the highest total token count")
    parser.add_argument(
        "--chunk-id", help="Paper id for the worst chunk to dump (defaults to same as --paper-id's "
        "worst-paper unless --chunk-only-id is given)",
    )
    parser.add_argument("--chunk-index", type=int, help="Chunk index within --chunk-id/--paper-id")
    args = parser.parse_args()

    pilot_df = pd.read_parquet(args.pilot_papers).set_index("id")
    chunks_df = pd.read_parquet(args.chunks)
    tokenizer = HFTokenizer(args.tokenizer_path)

    chunks_df["_tokens"] = chunks_df["text_with_context"].apply(tokenizer.count_tokens)

    paper_id = args.paper_id
    if paper_id is None:
        paper_id = chunks_df.groupby("id")["_tokens"].sum().idxmax()
        print(f"(auto) worst paper by total tokens: {paper_id}")

    chunk_paper_id = args.chunk_id or paper_id
    chunk_index = args.chunk_index
    if chunk_index is None:
        worst = chunks_df[chunks_df["id"] == chunk_paper_id].loc[
            chunks_df[chunks_df["id"] == chunk_paper_id]["_tokens"].idxmax()
        ]
        chunk_index = int(worst["chunk_index"])
        print(f"(auto) worst chunk within {chunk_paper_id}: chunk_index={chunk_index} tokens={worst['_tokens']}")

    print("=" * 80)
    print(f"paper bundle structure: {paper_id}")
    latex = pilot_df.loc[paper_id, "latex"]
    for name, content in _split_into_files(latex):
        has_doc = bool(_BEGIN_DOCUMENT_RE.search(content))
        print(f"{name!r:40} begin{{document}}={has_doc}  len={len(content)}")

    print("=" * 80)
    print(f"chunk content: {chunk_paper_id} chunk {chunk_index}")
    row = chunks_df[(chunks_df["id"] == chunk_paper_id) & (chunks_df["chunk_index"] == chunk_index)].iloc[0]
    print("section_path:", row["section_path"])
    print("tokens:", row["_tokens"])
    print(row["text_raw"][:2000])
    print("... [total len:", len(row["text_raw"]), "chars]")


if __name__ == "__main__":
    main()
