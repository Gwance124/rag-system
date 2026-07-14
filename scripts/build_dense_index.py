#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retrieval.benchmarks import load_bright_hf, load_litsearch_hf, load_mteb_hf
from retrieval.dense import QdrantIndex, VllmEmbeddingClient
from retrieval.types import Document


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Qdrant dense index from the local HF cache or JSONL.")
    parser.add_argument("--benchmark", choices=("bright", "litsearch", "beir", "jsonl"), default="litsearch")
    parser.add_argument("--domain", default="biology")
    parser.add_argument("--dataset-id", default="xlangai/BRIGHT")
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--split", default="test")
    parser.add_argument("--long-documents", action="store_true")
    parser.add_argument("--cache-dir")
    parser.add_argument("--documents", help="JSONL documents for --benchmark jsonl")
    parser.add_argument("--qdrant-url", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--embedding-url", default="http://solab-g3:8000/v1")
    parser.add_argument("--embedding-model", default="nvidia/llama-nv-embed-reasoning-3b")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.benchmark == "bright":
        benchmark = load_bright_hf(
            args.domain,
            dataset_id=args.dataset_id,
            long_documents=args.long_documents,
            cache_dir=args.cache_dir,
        )
        documents = benchmark.documents
    elif args.benchmark == "litsearch":
        documents = load_litsearch_hf(cache_dir=args.cache_dir).documents
    elif args.benchmark == "beir":
        dataset_id = args.dataset_id if args.dataset_id != "xlangai/BRIGHT" else f"mteb/{args.dataset}"
        documents = load_mteb_hf(dataset_id, split=args.split, cache_dir=args.cache_dir).documents
    else:
        if not args.documents:
            parser.error("jsonl indexing requires --documents")
        with open(args.documents) as handle:
            documents = [
                Document(
                    str(row["doc_id"]),
                    row["text"],
                    row.get("paper_id"),
                    row.get("metadata", {}),
                )
                for line in handle
                if line.strip()
                for row in [json.loads(line)]
            ]

    index = QdrantIndex(
        args.collection,
        VllmEmbeddingClient(args.embedding_url, args.embedding_model),
        args.qdrant_url,
    )
    index.create(documents, batch_size=args.batch_size)
    print(f"Indexed {len(documents)} documents into {args.collection}")


if __name__ == "__main__":
    main()
