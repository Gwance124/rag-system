#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retrieval.benchmarks import load_bright_hf, load_jsonl_benchmark, load_litsearch_hf, load_mteb_hf
from retrieval.dense import QdrantIndex, VllmEmbeddingClient
from retrieval.metrics import evaluate_run
from retrieval.pipeline import HybridRetriever
from retrieval.sparse import BM25Index
from retrieval.types import RetrievalConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the first retrieval baseline on BRIGHT or JSONL data.")
    parser.add_argument("--benchmark", choices=("bright", "litsearch", "beir", "jsonl"), default="litsearch")
    parser.add_argument("--domain", default="biology", help="BRIGHT domain/configuration")
    parser.add_argument("--dataset-id", default="xlangai/BRIGHT")
    parser.add_argument("--dataset", default="scifact", help="MTEB/BEIR dataset name, e.g. scifact or trec-covid")
    parser.add_argument("--split", default="test")
    parser.add_argument("--long-documents", action="store_true")
    parser.add_argument("--cache-dir", help="Local Hugging Face cache; missing files fail instead of downloading")
    parser.add_argument("--documents")
    parser.add_argument("--queries")
    parser.add_argument("--qrels")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--mode", choices=("sparse", "dense", "hybrid"), default="sparse")
    parser.add_argument("--embedding-url", default="http://solab-g3:8000/v1")
    parser.add_argument("--embedding-model", default="nvidia/llama-nv-embed-reasoning-3b")
    parser.add_argument("--qdrant-url", help="Qdrant REST base URL on the benchmark host")
    parser.add_argument("--collection", help="Qdrant collection for the selected corpus")
    args = parser.parse_args()

    if args.benchmark == "bright":
        benchmark = load_bright_hf(
            args.domain,
            dataset_id=args.dataset_id,
            long_documents=args.long_documents,
            cache_dir=args.cache_dir,
        )
    elif args.benchmark == "litsearch":
        benchmark = load_litsearch_hf(cache_dir=args.cache_dir)
    elif args.benchmark == "beir":
        dataset_id = args.dataset_id if args.dataset_id != "xlangai/BRIGHT" else f"mteb/{args.dataset}"
        benchmark = load_mteb_hf(dataset_id, split=args.split, cache_dir=args.cache_dir)
    else:
        if not all((args.documents, args.queries, args.qrels)):
            parser.error("jsonl benchmarks require --documents, --queries, and --qrels")
        benchmark = load_jsonl_benchmark(args.documents, args.queries, args.qrels)

    sparse_index = BM25Index(benchmark.documents) if args.mode in ("sparse", "hybrid") else None
    dense_index = None
    if args.mode in ("dense", "hybrid"):
        if not args.qdrant_url or not args.collection:
            parser.error("dense and hybrid modes require --qdrant-url and --collection")
        dense_index = QdrantIndex(
            args.collection,
            VllmEmbeddingClient(args.embedding_url, args.embedding_model),
            args.qdrant_url,
        )
    retriever = HybridRetriever(
        sparse_index=sparse_index,
        dense_index=dense_index,
        config=RetrievalConfig(
            top_n=args.top_n,
            top_k=args.top_k,
            sparse_weight=1.0 if args.mode == "sparse" else 0.5,
            dense_weight=1.0 if args.mode == "dense" else 0.5,
        ),
    )

    run = {}
    timings = []
    for query_id, query in benchmark.queries.items():
        result = retriever.search(query, benchmark.excluded_ids.get(query_id, ()))
        run[query_id] = [hit.doc_id for hit in result.hits]
        timings.append(result.timings_ms)

    print(json.dumps({"metrics": evaluate_run(run, benchmark.qrels), "queries": len(run), "timings_ms": timings}, indent=2))


if __name__ == "__main__":
    main()
