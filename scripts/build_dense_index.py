#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retrieval.benchmarks import (
    DEFAULT_MTEB_DATASET,
    load_bright_hf,
    load_litsearch_hf,
    load_mteb_hf,
    load_scholargym_benchmark,
    mteb_dataset_id,
)
from retrieval.dense import QdrantIndex, VllmEmbeddingClient
from retrieval.types import Document


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Qdrant dense index from the local HF cache or JSONL.")
    parser.add_argument(
        "--benchmark",
        choices=("bright", "litsearch", "mteb", "beir", "scholargym", "jsonl"),
        default="litsearch",
        help="benchmark family; beir is a backward-compatible alias for mteb",
    )
    parser.add_argument("--domain", default="biology")
    parser.add_argument("--dataset-id", help="Hugging Face dataset ID override")
    parser.add_argument("--dataset", default=DEFAULT_MTEB_DATASET, help="MTEB retrieval dataset (default: scidocs)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--long-documents", action="store_true")
    parser.add_argument("--cache-dir", help="Hugging Face root containing hub/ and datasets/")
    parser.add_argument("--documents", help="JSONL documents for --benchmark jsonl")
    parser.add_argument("--scholargym-paper-db", help="ScholarGym scholargym_paper_db.json")
    parser.add_argument("--scholargym-benchmark", help="ScholarGym scholargym_bench.jsonl")
    parser.add_argument("--qdrant-url", required=True)
    parser.add_argument("--collection", required=True, help="Unique name for this corpus and embedding model")
    parser.add_argument("--embedding-url", default="http://192.168.3.4:8000/v1")
    parser.add_argument("--embedding-model", default="nvidia/llama-nv-embed-reasoning-3b")
    parser.add_argument("--embedding-api-model", help="vLLM served model name, if different from the checkpoint name")
    parser.add_argument("--query-prefix", default="query: ")
    parser.add_argument("--passage-prefix", default="passage: ")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    is_mteb = args.benchmark in ("mteb", "beir")

    if args.benchmark == "bright":
        benchmark = load_bright_hf(
            args.domain,
            dataset_id=args.dataset_id or "xlangai/BRIGHT",
            long_documents=args.long_documents,
            cache_dir=args.cache_dir,
        )
        documents = benchmark.documents
    elif args.benchmark == "litsearch":
        documents = load_litsearch_hf(
            dataset_id=args.dataset_id or "princeton-nlp/LitSearch",
            cache_dir=args.cache_dir,
        ).documents
    elif is_mteb:
        dataset_id = args.dataset_id or mteb_dataset_id(args.dataset)
        documents = load_mteb_hf(dataset_id, split=args.split, cache_dir=args.cache_dir).documents
    elif args.benchmark == "scholargym":
        if not args.scholargym_paper_db or not args.scholargym_benchmark:
            parser.error("ScholarGym requires --scholargym-paper-db and --scholargym-benchmark")
        documents = load_scholargym_benchmark(
            args.scholargym_paper_db,
            args.scholargym_benchmark,
        ).documents
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
        VllmEmbeddingClient(
            args.embedding_url,
            args.embedding_api_model or args.embedding_model,
            args.query_prefix,
            args.passage_prefix,
        ),
        args.qdrant_url,
    )
    index.create(documents, batch_size=args.batch_size)
    print(f"Indexed {len(documents)} documents into {args.collection}")


if __name__ == "__main__":
    main()
