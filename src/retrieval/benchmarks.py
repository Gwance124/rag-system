from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from retrieval.types import Document


@dataclass(frozen=True)
class Benchmark:
    documents: list[Document]
    queries: dict[str, str]
    qrels: dict[str, set[str]]
    excluded_ids: dict[str, set[str]]


def _load_hf_split(
    dataset_id: str,
    config: str,
    split: str,
    cache_dir: str | Path | None,
):
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from datasets import DownloadConfig, DownloadMode, load_dataset
    except ImportError as exc:
        raise RuntimeError("install the eval extra to load Hugging Face benchmarks: pip install -e '.[eval]'") from exc

    if cache_dir:
        cache_root = Path(cache_dir).expanduser()
        os.environ["HF_HOME"] = str(cache_root)
        os.environ["HF_DATASETS_CACHE"] = str(cache_root / "datasets")
        os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")
        dataset_cache = str(cache_root / "datasets")
    else:
        dataset_cache = None
    return load_dataset(
        dataset_id,
        config,
        split=split,
        cache_dir=dataset_cache,
        download_config=DownloadConfig(cache_dir=dataset_cache, local_files_only=True),
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
    )


def _records(path: str | Path):
    with open(path) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_jsonl_benchmark(
    documents_path: str | Path,
    queries_path: str | Path,
    qrels_path: str | Path,
) -> Benchmark:
    documents = [
        Document(
            doc_id=str(row["doc_id"]),
            text=row["text"],
            paper_id=row.get("paper_id"),
            metadata=row.get("metadata", {}),
        )
        for row in _records(documents_path)
    ]
    queries = {}
    excluded_ids: dict[str, set[str]] = {}
    for row in _records(queries_path):
        query_id = str(row["query_id"])
        queries[query_id] = row["query"]
        excluded_ids[query_id] = {str(doc_id) for doc_id in row.get("excluded_ids", [])}
    qrels = {}
    for row in _records(qrels_path):
        qrels.setdefault(str(row["query_id"]), set()).add(str(row["doc_id"]))
    return Benchmark(documents, queries, qrels, excluded_ids)


def load_bright_hf(
    domain: str,
    *,
    dataset_id: str = "xlangai/BRIGHT",
    long_documents: bool = False,
    cache_dir: str | Path | None = None,
) -> Benchmark:
    """Load one cached BRIGHT domain without any network fallback."""
    examples = _load_hf_split(dataset_id, "examples", domain, cache_dir)
    document_config = "long_documents" if long_documents else "documents"
    documents_table = _load_hf_split(dataset_id, document_config, domain, cache_dir)
    documents = [Document(str(row["id"]), row["content"]) for row in documents_table]
    queries = {str(row["id"]): row["query"] for row in examples}
    key = "gold_ids_long" if long_documents else "gold_ids"
    qrels = {str(row["id"]): {str(doc_id) for doc_id in row[key]} for row in examples}
    excluded = {
        str(row["id"]): {str(doc_id) for doc_id in row.get("excluded_ids", [])}
        for row in examples
    }
    return Benchmark(documents, queries, qrels, excluded)


def load_litsearch_hf(
    *,
    dataset_id: str = "princeton-nlp/LitSearch",
    cache_dir: str | Path | None = None,
) -> Benchmark:
    """Load LitSearch's citation-query benchmark from its HF cache."""
    query_table = _load_hf_split(dataset_id, "query", "full", cache_dir)
    corpus_table = _load_hf_split(dataset_id, "corpus_clean", "full", cache_dir)

    def document_id(value) -> str:
        value = str(value)
        return value if value.startswith("d") else f"d{value}"

    documents = [
        Document(
            document_id(row["corpusid"]),
            f"{row.get('title', '')} {row.get('abstract', '')}".strip(),
        )
        for row in corpus_table
    ]
    queries = {}
    qrels: dict[str, set[str]] = {}
    for index, row in enumerate(query_table, 1):
        query_id = f"q{index}"
        queries[query_id] = row["query"]
        qrels[query_id] = {document_id(doc_id) for doc_id in row["corpusids"]}
    return Benchmark(documents, queries, qrels, {query_id: set() for query_id in queries})


def load_mteb_hf(
    dataset_id: str,
    *,
    split: str = "test",
    cache_dir: str | Path | None = None,
) -> Benchmark:
    """Load an MTEB-format BEIR task, such as mteb/scifact."""
    corpus_table = _load_hf_split(dataset_id, "corpus", split, cache_dir)
    query_table = _load_hf_split(dataset_id, "queries", split, cache_dir)
    qrel_table = _load_hf_split(dataset_id, "default", split, cache_dir)

    documents = [
        Document(
            str(row["id"]),
            f"{row.get('title', '')} {row.get('text', '')}".strip(),
        )
        for row in corpus_table
    ]
    queries = {str(row["id"]): row.get("text", row.get("query", "")) for row in query_table}
    qrels: dict[str, set[str]] = {}
    for row in qrel_table:
        if int(row.get("score", 1)) > 0:
            qrels.setdefault(str(row["query-id"]), set()).add(str(row["corpus-id"]))
    return Benchmark(documents, queries, qrels, {query_id: set() for query_id in queries})
