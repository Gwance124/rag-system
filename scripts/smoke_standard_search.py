#!/usr/bin/env python3
"""Run one dev-only Standard search across the p7/g3 service boundary."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rag_system.datasets.browsecomp_plus import iter_corpus_repository
from rag_system.retrieval.faiss_documents import FaissDocumentBackend
from rag_system.retrieval.remote_encoder import RemoteQueryEncoder
from rag_system.retrieval.standard import StandardSearchTool


def load_development_query(
    prepared_dir: Path,
    query_id: str,
) -> str:
    split = json.loads((prepared_dir / "split.json").read_text(encoding="utf-8"))
    development_ids = set(split["development_query_ids"])
    if query_id not in development_ids:
        raise ValueError(f"query ID {query_id!r} is not in the frozen development split")
    with (prepared_dir / "queries.decrypted.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("query_id") == query_id:
                return row["question"]
    raise ValueError(f"query ID {query_id!r} is absent from the prepared query artifact")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--query-id", required=True)
    parser.add_argument("--corpus-repo", type=Path, required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--encoder-url", required=True)
    parser.add_argument("--datasets-cache", type=Path)
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        parser.error(f"Transformers is required: {exc}")

    question = load_development_query(
        args.prepared_dir.expanduser().resolve(),
        args.query_id,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer_path.expanduser().resolve()),
        local_files_only=True,
    )
    encoder = RemoteQueryEncoder(args.encoder_url)
    documents = iter_corpus_repository(
        args.corpus_repo,
        cache_dir=args.datasets_cache,
    )
    backend = FaissDocumentBackend(args.index_path, encoder, documents)
    trace = StandardSearchTool(backend, tokenizer).search(question)
    summary = {
        "query_id": args.query_id,
        "top_k": trace.top_k,
        "snippet_max_tokens": trace.snippet_max_tokens,
        "hits": [
            {
                "rank": rank,
                "document_id": hit.document_id,
                "score": hit.score,
                "snippet_token_count": hit.snippet_token_count,
            }
            for rank, hit in enumerate(trace.hits, start=1)
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
