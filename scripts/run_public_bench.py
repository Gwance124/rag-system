#!/usr/bin/env python3
"""Run one offline retrieval benchmark and emit metrics as JSON."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retrieval.benchmarks import (
    Benchmark,
    DEFAULT_MTEB_DATASET,
    DEFAULT_QASPER_DATASET,
    DEFAULT_QASPER_RAW_DATASET,
    load_bright_hf,
    load_jsonl_benchmark,
    load_litsearch_hf,
    load_mteb_hf,
    load_qasper_hf,
    load_qasper_paper_benchmark_hf,
    load_scholargym_benchmark,
    mteb_dataset_id,
    qasper_chunk_candidates,
    scholargym_paths,
)
from retrieval.dense import QdrantIndex, VllmEmbeddingClient
from retrieval.metrics import evaluate_capped_recall, evaluate_litsearch_comparison, evaluate_run
from retrieval.pipeline import HybridRetriever
from retrieval.sparse import BM25Index
from retrieval.types import RetrievalConfig


_DENSE_MODES = {"dense", "hybrid"}
_DOCUMENT_TEXT = {
    "bright": "content",
    "litsearch": "title+abstract",
    "mteb": "title+text",
    "beir": "title+text",
    "qasper": "section+paragraph",
    "scholargym": "title+abstract",
    "jsonl": "text",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run retrieval benchmarks from the local cache or JSONL files.")
    parser.add_argument(
        "--benchmark",
        choices=("bright", "litsearch", "mteb", "beir", "qasper", "scholargym", "jsonl"),
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
        "--qasper-scope",
        choices=("global", "paper", "two-stage"),
        default="global",
        help="global searches all chunks; paper uses the known paper; two-stage retrieves papers then chunks",
    )
    parser.add_argument("--qasper-query-limit", type=int, help="limit QASPER queries for smoke tests")
    parser.add_argument(
        "--qasper-raw-dataset-id",
        default=DEFAULT_QASPER_RAW_DATASET,
        help="raw QASPER dataset containing paper titles and abstracts",
    )
    parser.add_argument(
        "--qasper-paper-collection",
        help="Qdrant title+abstract collection required by dense/hybrid two-stage QASPER",
    )
    parser.add_argument(
        "--qasper-paper-top-k",
        type=int,
        default=20,
        help="number of Stage 1 papers whose chunks are searched (default: 20)",
    )
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
    return parser


def _load_requested_benchmark(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> tuple[Benchmark, str | None, Path | None, Path | None]:
    """Resolve CLI source arguments and load one benchmark into memory."""
    is_mteb = args.benchmark in ("mteb", "beir")
    dataset_id = None
    paper_db_path = None
    benchmark_path = None
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
    elif args.benchmark == "qasper":
        dataset_id = args.dataset_id or DEFAULT_QASPER_DATASET
        benchmark = load_qasper_hf(
            scope="global" if args.qasper_scope == "two-stage" else args.qasper_scope,
            dataset_id=dataset_id,
            split=args.split,
            cache_dir=args.cache_dir,
            query_limit=args.qasper_query_limit,
        )
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
    return benchmark, dataset_id, paper_db_path, benchmark_path


def _report_progress(args: argparse.Namespace, done: int, total: int, stage: str) -> None:
    print(f"[{args.benchmark}/{args.mode}] {stage}: {done}/{total}", file=sys.stderr, flush=True)


def _build_retriever(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    benchmark: Benchmark,
    *,
    collection: str | None = None,
    top_n: int | None = None,
    top_k: int | None = None,
) -> HybridRetriever:
    sparse_index = None
    if args.mode in ("sparse", "hybrid"):
        sparse_index = BM25Index(
            benchmark.documents,
            progress=lambda done, total: _report_progress(args, done, total, "BM25 documents"),
        )

    dense_index = None
    if args.mode in _DENSE_MODES:
        collection = collection or args.collection
        if not args.qdrant_url or not collection:
            parser.error("dense and hybrid modes require --qdrant-url and --collection")
        embedder = VllmEmbeddingClient(
            args.embedding_url,
            args.embedding_api_model or args.embedding_model,
            args.query_prefix,
            args.passage_prefix,
        )
        dense_index = QdrantIndex(collection, embedder, args.qdrant_url)

    return HybridRetriever(
        sparse_index=sparse_index,
        dense_index=dense_index,
        config=RetrievalConfig(
            top_n=args.top_n if top_n is None else top_n,
            top_k=args.top_k if top_k is None else top_k,
            sparse_weight=1.0 if args.mode == "sparse" else 0.5,
            dense_weight=1.0 if args.mode == "dense" else 0.5,
        ),
    )


def _run_queries(
    args: argparse.Namespace,
    benchmark: Benchmark,
    retriever: HybridRetriever,
) -> tuple[dict[str, list[str]], list[dict[str, float]]]:
    run = {}
    timings = []
    total_queries = len(benchmark.queries)
    for query_number, (query_id, query) in enumerate(benchmark.queries.items(), 1):
        candidates = (
            benchmark.candidate_ids.get(query_id, set())
            if benchmark.candidate_ids is not None
            else None
        )
        result = retriever.search(
            query,
            benchmark.excluded_ids.get(query_id, ()),
            candidates,
        )
        run[query_id] = [hit.doc_id for hit in result.hits]
        timings.append(result.timings_ms)
        if query_number % 25 == 0 or query_number == total_queries:
            _report_progress(args, query_number, total_queries, "queries")
    return run, timings


def _build_output(
    args: argparse.Namespace,
    benchmark: Benchmark,
    dataset_id: str | None,
    paper_db_path: Path | None,
    benchmark_path: Path | None,
    run: dict,
    timings: list[dict[str, float]],
    qasper_two_stage: dict | None = None,
) -> dict:
    is_mteb = args.benchmark in ("mteb", "beir")
    uses_dense = args.mode in _DENSE_MODES
    output = {
        "config": {
            "benchmark": "mteb" if is_mteb else args.benchmark,
            "dataset": (
                dataset_id.rsplit("/", 1)[-1]
                if dataset_id and (is_mteb or args.benchmark == "qasper")
                else None
            ),
            "dataset_id": dataset_id,
            "domain": args.domain if args.benchmark == "bright" else None,
            "split": args.split if is_mteb or args.benchmark == "qasper" else None,
            "custom_extension": (
                args.benchmark == "scholargym"
                or (args.benchmark == "qasper" and args.qasper_scope != "paper")
            ),
            "scholargym_paper_db": str(paper_db_path) if args.benchmark == "scholargym" else None,
            "scholargym_benchmark": str(benchmark_path) if args.benchmark == "scholargym" else None,
            "scholargym_query_limit": args.scholargym_query_limit if args.benchmark == "scholargym" else None,
            "qasper_scope": args.qasper_scope if args.benchmark == "qasper" else None,
            "qasper_query_limit": args.qasper_query_limit if args.benchmark == "qasper" else None,
            "qasper_raw_dataset_id": (
                args.qasper_raw_dataset_id
                if args.benchmark == "qasper" and args.qasper_scope == "two-stage"
                else None
            ),
            "qasper_paper_top_k": (
                args.qasper_paper_top_k
                if args.benchmark == "qasper" and args.qasper_scope == "two-stage"
                else None
            ),
            "qasper_paper_collection": (
                args.qasper_paper_collection
                if uses_dense and args.benchmark == "qasper" and args.qasper_scope == "two-stage"
                else None
            ),
            "benchmark_version": "LMEB v4" if args.benchmark == "qasper" else None,
            "mode": args.mode,
            "embedding_model": args.embedding_model if uses_dense else None,
            "embedding_api_model": (args.embedding_api_model or args.embedding_model) if uses_dense else None,
            "query_prefix": args.query_prefix if uses_dense else None,
            "passage_prefix": args.passage_prefix if uses_dense else None,
            "collection": args.collection if uses_dense else None,
            "documents": (
                "title+abstract -> section+paragraph"
                if args.benchmark == "qasper" and args.qasper_scope == "two-stage"
                else _DOCUMENT_TEXT[args.benchmark]
            ),
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
            "query_sources": sorted(
                {
                    metadata["source"]
                    for metadata in (benchmark.query_metadata or {}).values()
                    if metadata.get("source")
                }
            ),
        }
    elif args.benchmark == "qasper":
        scoped = benchmark.candidate_ids is not None
        candidate_counts = [
            metadata["candidate_count"]
            for metadata in (benchmark.query_metadata or {}).values()
        ]
        output["qasper"] = {
            "label": (
                "QASPER two-stage paper-to-chunk retrieval (custom extension)"
                if args.qasper_scope == "two-stage"
                else (
                    "LMEB v4 QASPER paper-scoped"
                    if scoped
                    else "LMEB v4 QASPER-global (custom extension)"
                )
            ),
            "retrieval": (
                f"question -> top-{args.qasper_paper_top_k} title+abstract papers -> their chunks"
                if args.qasper_scope == "two-stage"
                else (
                    "question -> chunks from the known paper"
                    if scoped
                    else "question -> all QASPER chunks"
                )
            ),
            "query_context": "candidate scope only; paper text is not appended to the query",
            "official_metrics": {
                "ndcg@10": output["metrics"]["ndcg@10"],
                "capped_recall@10": evaluate_capped_recall(run, benchmark.qrels, 10),
            },
            "official_instruction": "Given a query, retrieve documents that answer the query",
            "average_candidates": (
                sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0
            ),
            "label_caveat": (
                None
                if args.qasper_scope == "paper"
                else "Gold evidence was annotated only inside the target paper; useful chunks in other papers are unlabeled."
            ),
        }
        if qasper_two_stage is not None:
            output["qasper"]["paper_retrieval"] = {
                "metrics": qasper_two_stage["paper_metrics"],
                "top_k": args.qasper_paper_top_k,
                "average_candidate_chunks": (
                    sum(candidate_counts) / len(candidate_counts)
                    if candidate_counts
                    else 0.0
                ),
            }
            output["qasper"]["conditional_evidence_metrics"] = qasper_two_stage[
                "conditional_evidence_metrics"
            ]
            output["qasper"]["paper_timings_ms"] = qasper_two_stage["paper_timings"]
    return output


def _run_qasper_two_stage(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    chunks: Benchmark,
) -> tuple[Benchmark, dict[str, list[str]], list[dict[str, float]], dict]:
    """Retrieve QASPER papers by title+abstract, then evidence within those papers."""
    if args.qasper_paper_top_k <= 0:
        parser.error("--qasper-paper-top-k must be positive")
    if args.mode in _DENSE_MODES and not args.qasper_paper_collection:
        parser.error("dense/hybrid two-stage QASPER requires --qasper-paper-collection")

    papers = load_qasper_paper_benchmark_hf(
        dataset_id=args.dataset_id or DEFAULT_QASPER_DATASET,
        raw_dataset_id=args.qasper_raw_dataset_id,
        split=args.split,
        cache_dir=args.cache_dir,
        query_limit=args.qasper_query_limit,
        chunk_benchmark=chunks,
    )
    print(
        f"[qasper/{args.mode}] Stage 1 loaded {len(papers.documents)} title+abstract papers",
        file=sys.stderr,
        flush=True,
    )
    paper_ranking_depth = max(50, args.qasper_paper_top_k)
    paper_retriever = _build_retriever(
        args,
        parser,
        papers,
        collection=args.qasper_paper_collection,
        top_n=max(args.top_n, paper_ranking_depth),
        top_k=paper_ranking_depth,
    )
    paper_run, paper_timings = _run_queries(args, papers, paper_retriever)
    candidate_paper_run = {
        query_id: ranking[: args.qasper_paper_top_k]
        for query_id, ranking in paper_run.items()
    }
    candidate_ids = qasper_chunk_candidates(chunks.documents, candidate_paper_run)
    chunk_metadata = {
        query_id: {
            "scope": "two-stage",
            "candidate_count": len(candidate_ids[query_id]),
            "retrieved_paper_count": len(candidate_paper_run[query_id]),
        }
        for query_id in chunks.queries
    }
    restricted_chunks = replace(
        chunks,
        candidate_ids=candidate_ids,
        query_metadata=chunk_metadata,
    )
    chunk_retriever = _build_retriever(args, parser, restricted_chunks)
    run, timings = _run_queries(args, restricted_chunks, chunk_retriever)

    successful_queries = {
        query_id
        for query_id, ranking in candidate_paper_run.items()
        if set(ranking) & papers.qrels[query_id]
    }
    conditional_run = {
        query_id: run[query_id]
        for query_id in successful_queries
    }
    conditional_qrels = {
        query_id: restricted_chunks.qrels[query_id]
        for query_id in successful_queries
    }
    details = {
        "paper_metrics": evaluate_run(
            paper_run,
            papers.qrels,
            ks=(1, 5, 10, 20, 50),
        ),
        "conditional_evidence_metrics": {
            **evaluate_run(
                conditional_run,
                conditional_qrels,
                ks=(5, 10, 20, 50, 100),
            ),
            "capped_recall@10": evaluate_capped_recall(
                conditional_run,
                conditional_qrels,
                10,
            ),
            "queries": len(successful_queries),
        },
        "paper_timings": paper_timings,
    }
    return restricted_chunks, run, timings, details


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    benchmark, dataset_id, paper_db_path, benchmark_path = _load_requested_benchmark(args, parser)
    print(
        f"[{args.benchmark}/{args.mode}] loaded {len(benchmark.documents)} documents "
        f"and {len(benchmark.queries)} queries",
        file=sys.stderr,
        flush=True,
    )
    qasper_two_stage = None
    if args.benchmark == "qasper" and args.qasper_scope == "two-stage":
        benchmark, run, timings, qasper_two_stage = _run_qasper_two_stage(
            args,
            parser,
            benchmark,
        )
    else:
        retriever = _build_retriever(args, parser, benchmark)
        run, timings = _run_queries(args, benchmark, retriever)
    output = _build_output(
        args,
        benchmark,
        dataset_id,
        paper_db_path,
        benchmark_path,
        run,
        timings,
        qasper_two_stage,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
