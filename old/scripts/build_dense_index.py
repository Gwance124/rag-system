#!/usr/bin/env python3
"""Build one model-specific Qdrant collection from a benchmark corpus."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from old.src.retrieval.benchmarks import (
    DEFAULT_MTEB_DATASET,
    DEFAULT_QASPER_DATASET,
    DEFAULT_QASPER_RAW_DATASET,
    load_bright_hf,
    load_jsonl_documents,
    load_litsearch_hf,
    load_mteb_hf,
    load_qasper_hf,
    load_qasper_paper_documents_hf,
    load_scholargym_benchmark,
    mteb_dataset_id,
    scholargym_paths,
)
from old.src.retrieval.dense import QdrantIndex, VllmEmbeddingClient
from old.src.retrieval.types import Document


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Qdrant dense index from the local HF cache or JSONL.")
    parser.add_argument(
        "--benchmark",
        choices=("bright", "litsearch", "mteb", "beir", "qasper", "scholargym", "jsonl"),
        default="litsearch",
        help="benchmark family; beir is a backward-compatible alias for mteb",
    )
    parser.add_argument("--domain", default="biology")
    parser.add_argument("--dataset-id", help="Hugging Face dataset ID override")
    parser.add_argument("--dataset", default=DEFAULT_MTEB_DATASET, help="MTEB retrieval dataset (default: scidocs)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--long-documents", action="store_true")
    parser.add_argument("--cache-dir", help="Hugging Face root containing hub/ and datasets/")
    parser.add_argument(
        "--qasper-corpus",
        choices=("chunks", "papers"),
        default="chunks",
        help="index LMEB chunks or raw-QASPER title+abstract papers",
    )
    parser.add_argument(
        "--qasper-raw-dataset-id",
        default=DEFAULT_QASPER_RAW_DATASET,
        help="raw QASPER dataset containing paper titles and abstracts",
    )
    parser.add_argument("--documents", help="JSONL documents for --benchmark jsonl")
    parser.add_argument("--scholargym-paper-db", help="ScholarGym scholargym_paper_db.json")
    parser.add_argument("--scholargym-benchmark", help="ScholarGym scholargym_bench.jsonl")
    parser.add_argument("--scholargym-dir", help="ScholarGym dataset directory; defaults to <cache-dir>/datasets/scholargym")
    parser.add_argument("--qdrant-url", required=True)
    parser.add_argument("--collection", required=True, help="Unique name for this corpus and embedding model")
    parser.add_argument("--embedding-url", default="http://192.168.3.4:8000/v1")
    parser.add_argument("--embedding-model", default="nvidia/llama-nv-embed-reasoning-3b")
    parser.add_argument("--embedding-api-model", help="vLLM served model name, if different from the checkpoint name")
    parser.add_argument("--query-prefix", default="query: ")
    parser.add_argument("--passage-prefix", default="passage: ")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def _load_documents(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[Document]:
    """Load only the corpus selected by the index-building arguments."""
    is_mteb = args.benchmark in ("mteb", "beir")
    if args.benchmark == "bright":
        return load_bright_hf(
            args.domain,
            dataset_id=args.dataset_id or "xlangai/BRIGHT",
            long_documents=args.long_documents,
            cache_dir=args.cache_dir,
        ).documents
    if args.benchmark == "litsearch":
        return load_litsearch_hf(
            dataset_id=args.dataset_id or "princeton-nlp/LitSearch",
            cache_dir=args.cache_dir,
        ).documents
    if is_mteb:
        dataset_id = args.dataset_id or mteb_dataset_id(args.dataset)
        return load_mteb_hf(dataset_id, split=args.split, cache_dir=args.cache_dir).documents
    if args.benchmark == "qasper":
        if args.qasper_corpus == "papers":
            return load_qasper_paper_documents_hf(
                dataset_id=args.qasper_raw_dataset_id,
                cache_dir=args.cache_dir,
            )
        return load_qasper_hf(
            dataset_id=args.dataset_id or DEFAULT_QASPER_DATASET,
            split=args.split,
            cache_dir=args.cache_dir,
        ).documents
    if args.benchmark == "scholargym":
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
        return load_scholargym_benchmark(
            paper_db_path,
            benchmark_path,
        ).documents
    if not args.documents:
        parser.error("jsonl indexing requires --documents")
    return load_jsonl_documents(args.documents)


def _progress_reporter(benchmark: str) -> Callable[[int, int], None]:
    started = last_report = time.monotonic()

    def report(done: int, total: int) -> None:
        nonlocal last_report
        now = time.monotonic()
        if done < total and now - last_report < 10:
            return
        elapsed = max(now - started, 0.001)
        rate = done / elapsed
        eta = (total - done) / rate if rate else 0
        print(
            f"[{benchmark}/dense-index] embedded+upserted {done}/{total} "
            f"({100 * done / total:.1f}%, {rate:.1f} docs/s, ETA {eta / 60:.1f} min)",
            file=sys.stderr,
            flush=True,
        )
        last_report = now

    return report


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    documents = _load_documents(args, parser)

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
    print(
        f"[{args.benchmark}/dense-index] loaded {len(documents)} documents",
        file=sys.stderr,
        flush=True,
    )
    index.create(documents, batch_size=args.batch_size, progress=_progress_reporter(args.benchmark))
    print(f"Indexed {len(documents)} documents into {args.collection}")


if __name__ == "__main__":
    main()
