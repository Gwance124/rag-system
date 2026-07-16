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
    load_jsonl_benchmark,
    load_litsearch_hf,
    load_mteb_hf,
    load_scholargym_benchmark,
    mteb_dataset_id,
    scholargym_paths,
)
from retrieval.dense import QdrantIndex, VllmEmbeddingClient
from retrieval.metrics import evaluate_litsearch_comparison, evaluate_run
from retrieval.pipeline import HybridRetriever
from retrieval.sparse import BM25Index
from retrieval.types import RetrievalConfig

def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval benchmarks from the local cache or JSONL files.")
    parser.add_argument(
        "--benchmark",
        choices=("bright", "litsearch", "mteb", "beir", "scholargym", "jsonl"),
        default="litsearch",
        help="benchmark family; beir is a backward-compatible alias for mteb",
    )
    parser.add_argument("--domain", default="biology", help="BRIGHT domain/configuration")
    parser.add_argument("--dataset-id", help="Hugging Face dataset ID override")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_MTEB_DATASET,
        help="MTEB/BEIR dataset name or full dataset ID (default: scidocs)",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--long-documents", action="store_true")
    parser.add_argument(
        "--cache-dir",
        help="Hugging Face root containing hub/ and datasets/; missing files fail instead of downloading",
    )
    parser.add_argument("--documents")
    parser.add_argument("--queries")
    parser.add_argument("--qrels")
    parser.add_argument("--scholargym-paper-db", help="ScholarGym scholargym_paper_db.json")
    parser.add_argument("--scholargym-benchmark", help="ScholarGym scholargym_bench.jsonl")
    parser.add_argument("--scholargym-dir", help="ScholarGym dataset directory; defaults to <cache-dir>/datasets/scholargym")
    parser.add_argument("--scholargym-query-limit", type=int)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--mode", choices=("sparse", "dense", "hybrid"), default="sparse")
    parser.add_argument("--embedding-url", default="http://192.168.3.4:8000/v1")
    parser.add_argument("--embedding-model", default="nvidia/llama-nv-embed-reasoning-3b")
    parser.add_argument("--embedding-api-model", help="vLLM served model name, if different from the checkpoint name")
    parser.add_argument("--query-prefix", default="query: ")
    parser.add_argument("--passage-prefix", default="passage: ")
    parser.add_argument("--qdrant-url", help="Qdrant REST base URL on the benchmark host")
    parser.add_argument("--collection", help="Qdrant collection for this corpus and embedding model")
    args = parser.parse_args()
    is_mteb = args.benchmark in ("mteb", "beir")

    dataset_id = None
    if args.benchmark == "bright":
        dataset_id = args.dataset_id or "xlangai/BRIGHT"
        benchmark = load_bright_hf(
            args.domain,
            dataset_id=dataset_id,
            long_documents=args.long_documents,
            cache_dir=args.cache_dir,
        )
    elif args.benchmark == "litsearch":
        dataset_id = args.dataset_id or "princeton-nlp/LitSearch"
        benchmark = load_litsearch_hf(dataset_id=dataset_id, cache_dir=args.cache_dir)
    elif is_mteb:
        dataset_id = args.dataset_id or mteb_dataset_id(args.dataset)
        benchmark = load_mteb_hf(dataset_id, split=args.split, cache_dir=args.cache_dir)
    elif args.benchmark == "scholargym":
        paper_db_path, benchmark_path = scholargym_paths(
            args.cache_dir,
            args.scholargym_dir,
            args.scholargym_paper_db,
            args.scholargym_benchmark,
        )
        if not paper_db_path.is_file() or not benchmark_path.is_file():
            parser.error(
                "ScholarGym files not found; expected "
                f"{paper_db_path} and {benchmark_path}"
            )
        benchmark = load_scholargym_benchmark(
            paper_db_path,
            benchmark_path,
            query_limit=args.scholargym_query_limit,
        )
    else:
        if not all((args.documents, args.queries, args.qrels)):
            parser.error("jsonl benchmarks require --documents, --queries, and --qrels")
        benchmark = load_jsonl_benchmark(args.documents, args.queries, args.qrels)

    def report_progress(done: int, total: int, stage: str) -> None:
        print(f"[{args.benchmark}/{args.mode}] {stage}: {done}/{total}", file=sys.stderr, flush=True)

    print(
        f"[{args.benchmark}/{args.mode}] loaded {len(benchmark.documents)} documents "
        f"and {len(benchmark.queries)} queries",
        file=sys.stderr,
        flush=True,
    )
    sparse_index = (
        BM25Index(
            benchmark.documents,
            progress=lambda done, total: report_progress(done, total, "BM25 documents"),
        )
        if args.mode in ("sparse", "hybrid")
        else None
    )
    dense_index = None
    if args.mode in ("dense", "hybrid"):
        if not args.qdrant_url or not args.collection:
            parser.error("dense and hybrid modes require --qdrant-url and --collection")
        dense_index = QdrantIndex(
            args.collection,
            VllmEmbeddingClient(
                args.embedding_url,
                args.embedding_api_model or args.embedding_model,
                args.query_prefix,
                args.passage_prefix,
            ),
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
    total_queries = len(benchmark.queries)
    for query_number, (query_id, query) in enumerate(benchmark.queries.items(), 1):
        result = retriever.search(query, benchmark.excluded_ids.get(query_id, ()))
        run[query_id] = [hit.doc_id for hit in result.hits]
        timings.append(result.timings_ms)
        if query_number % 25 == 0 or query_number == total_queries:
            report_progress(query_number, total_queries, "queries")

    output = {
        "config": {
            "benchmark": "mteb" if is_mteb else args.benchmark,
            "dataset": dataset_id.rsplit("/", 1)[-1] if is_mteb else None,
            "dataset_id": dataset_id,
            "domain": args.domain if args.benchmark == "bright" else None,
            "split": args.split if is_mteb else None,
            "custom_extension": args.benchmark == "scholargym",
            "scholargym_paper_db": str(paper_db_path) if args.benchmark == "scholargym" else None,
            "scholargym_benchmark": str(benchmark_path) if args.benchmark == "scholargym" else None,
            "scholargym_query_limit": args.scholargym_query_limit if args.benchmark == "scholargym" else None,
            "mode": args.mode,
            "embedding_model": args.embedding_model if args.mode in ("dense", "hybrid") else None,
            "embedding_api_model": (
                args.embedding_api_model or args.embedding_model
                if args.mode in ("dense", "hybrid")
                else None
            ),
            "query_prefix": args.query_prefix if args.mode in ("dense", "hybrid") else None,
            "passage_prefix": args.passage_prefix if args.mode in ("dense", "hybrid") else None,
            "collection": args.collection if args.mode in ("dense", "hybrid") else None,
            "documents": {
                "bright": "content",
                "litsearch": "title+abstract",
                "mteb": "title+text",
                "beir": "title+text",
                "scholargym": "title+abstract",
                "jsonl": "text",
            }[args.benchmark],
        },
        "metrics": evaluate_run(run, benchmark.qrels, ks=(5, 10, 20, 50, 100)),
        "queries": len(run),
        "timings_ms": timings,
    }
    if args.benchmark == "litsearch":
        output["litsearch_paper_comparison"] = evaluate_litsearch_comparison(benchmark, run)
    elif args.benchmark == "scholargym":
        output["scholargym_static"] = {
            "label": "ScholarGym-static (custom extension)",
            "retrieval": "single-shot title+abstract",
            "primary_metrics": ["recall@20", "recall@50"],
            "secondary_metrics": ["ndcg@10"],
            "query_sources": sorted({metadata.get("source") for metadata in (benchmark.query_metadata or {}).values() if metadata.get("source")}),
        }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
