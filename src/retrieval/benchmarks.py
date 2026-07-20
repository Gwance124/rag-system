"""Offline benchmark loaders normalized to a shared retrieval data model."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from retrieval.types import Document


DEFAULT_MTEB_DATASET = "scidocs"
DEFAULT_QASPER_DATASET = "mteb/QASPER"
QASPER_SCOPES = ("global", "paper")


def mteb_dataset_id(dataset: str) -> str:
    """Resolve a BEIR task name or preserve an explicit Hugging Face dataset ID."""
    return dataset if "/" in dataset else f"mteb/{dataset}"


def scholargym_paths(
    cache_dir: str | Path | None = None,
    dataset_dir: str | Path | None = None,
    paper_db_path: str | Path | None = None,
    benchmark_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve ScholarGym files under the shared Hugging Face dataset cache."""
    if dataset_dir is None:
        cache_root = Path(cache_dir or os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
        dataset_dir = cache_root / "datasets" / "datasets--shenhao--ScholarGym"
    dataset_dir = Path(dataset_dir).expanduser()

    # HF cache repositories use refs/main -> snapshots/<commit>; support the
    # extracted directory layout as a fallback for manually staged files.
    snapshot_dir = dataset_dir
    refs_main = dataset_dir / "refs" / "main"
    if refs_main.is_file():
        candidate = dataset_dir / "snapshots" / refs_main.read_text().strip()
        if candidate.is_dir():
            snapshot_dir = candidate
    elif (dataset_dir / "snapshots").is_dir():
        snapshots = sorted(path for path in (dataset_dir / "snapshots").iterdir() if path.is_dir())
        if snapshots:
            snapshot_dir = snapshots[-1]

    def cached_file(name: str, override: str | Path | None) -> Path:
        if override:
            return Path(override).expanduser()
        direct = snapshot_dir / name
        if direct.is_file():
            return direct
        matches = sorted(snapshot_dir.rglob(name))
        return matches[0] if matches else direct

    return (
        cached_file("scholargym_paper_db.json", paper_db_path),
        cached_file("scholargym_bench.jsonl", benchmark_path),
    )


@dataclass(frozen=True)
class Benchmark:
    documents: list[Document]
    queries: dict[str, str]
    qrels: dict[str, set[str]]
    excluded_ids: dict[str, set[str]]
    query_metadata: dict[str, dict] | None = None
    candidate_ids: dict[str, set[str]] | None = None


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


def load_jsonl_documents(path: str | Path) -> list[Document]:
    """Load the document schema shared by JSONL benchmarks and dense indexes."""
    return [
        Document(
            doc_id=str(row["doc_id"]),
            text=row["text"],
            paper_id=row.get("paper_id"),
            metadata=row.get("metadata", {}),
        )
        for row in _records(path)
    ]


def load_jsonl_benchmark(
    documents_path: str | Path,
    queries_path: str | Path,
    qrels_path: str | Path,
) -> Benchmark:
    documents = load_jsonl_documents(documents_path)
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


def _scholargym_id(value) -> str:
    """Use the version-free arXiv ID used by both ScholarGym files."""
    value = str(value).strip()
    if value.lower().startswith("arxiv:"):
        value = value[6:]
    value = value.rsplit("/", 1)[-1]
    return re.sub(r"v\d+$", "", value)


def _scholargym_papers(path: str | Path):
    with open(path) as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("papers", "documents", "data"):
            if isinstance(data.get(key), list):
                return data[key]
        # The released paper DB is a mapping from arXiv ID to metadata.
        return [dict(row, arxiv_id=row.get("arxiv_id", paper_id)) for paper_id, row in data.items() if isinstance(row, dict)]
    raise ValueError(f"unsupported ScholarGym paper DB format: {path}")


def _scholargym_positive_ids(row: dict) -> set[str]:
    if "gt_arxiv_ids" in row:
        return {_scholargym_id(value) for value in row["gt_arxiv_ids"]}
    cited = row.get("cited_paper", [])
    labels = row.get("gt_label")
    if labels is None:
        labels = [1] * len(cited)

    def positive(label) -> bool:
        return label.strip().lower() not in {"", "0", "false", "no"} if isinstance(label, str) else bool(label)

    return {
        _scholargym_id(paper["arxiv_id"] if isinstance(paper, dict) else paper)
        for paper, label in zip(cited, labels)
        if positive(label)
    }


def _scholargym_document(row: dict) -> Document | None:
    paper_id = row.get("arxiv_id", row.get("id", row.get("_id")))
    if paper_id is None:
        return None
    paper_id = _scholargym_id(paper_id)
    title = str(row.get("title") or "")
    abstract = str(row.get("abstract") or row.get("summary") or "")
    metadata = {
        key: row[key]
        for key in ("authors", "published", "year", "categories", "url")
        if key in row
    }
    return Document(paper_id, f"{title} {abstract}".strip(), paper_id, metadata)


def _scholargym_query(row: dict) -> tuple[str, str, set[str], dict] | None:
    if row.get("valid") is False:
        return None
    query_id = str(row.get("query_id", row.get("qid")))
    if query_id == "None":
        return None
    positives = _scholargym_positive_ids(row)
    if not positives:
        return None
    metadata = {
        key: row[key]
        for key in ("source", "split", "date", "date_constraint")
        if key in row
    }
    return query_id, str(row.get("query", "")), positives, metadata


def load_scholargym_benchmark(
    paper_db_path: str | Path,
    benchmark_path: str | Path,
    *,
    query_limit: int | None = None,
) -> Benchmark:
    """Load ScholarGym's released files for the ScholarGym-static extension.

    This is intentionally single-shot title+abstract retrieval. It does not
    implement ScholarGym's agent workflow or its iterative selection metrics.
    """
    documents = []
    for row in _scholargym_papers(paper_db_path):
        document = _scholargym_document(row)
        if document is not None:
            documents.append(document)

    queries = {}
    qrels: dict[str, set[str]] = {}
    query_metadata = {}
    for row in _records(benchmark_path):
        query = _scholargym_query(row)
        if query is None:
            continue
        query_id, text, positives, metadata = query
        queries[query_id] = text
        qrels[query_id] = positives
        query_metadata[query_id] = metadata
        if query_limit is not None and len(queries) >= query_limit:
            break
    return Benchmark(
        documents,
        queries,
        qrels,
        {query_id: set() for query_id in queries},
        query_metadata,
    )


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
    query_metadata = {}
    for index, row in enumerate(query_table, 1):
        query_id = f"q{index}"
        queries[query_id] = row["query"]
        qrels[query_id] = {document_id(doc_id) for doc_id in row["corpusids"]}
        query_metadata[query_id] = {
            key: row[key]
            for key in ("query_set", "specificity", "quality")
            if key in row
        }
    return Benchmark(
        documents,
        queries,
        qrels,
        {query_id: set() for query_id in queries},
        query_metadata,
    )


def load_mteb_hf(
    dataset_id: str,
    *,
    split: str = "test",
    cache_dir: str | Path | None = None,
) -> Benchmark:
    """Load an MTEB-format BEIR task, such as mteb/scifact."""
    corpus_table = _load_hf_split(dataset_id, "corpus", "corpus", cache_dir)
    query_table = _load_hf_split(dataset_id, "queries", "queries", cache_dir)
    qrel_table = _load_hf_split(dataset_id, "default", split, cache_dir)

    documents = [
        Document(
            str(row["id"] if "id" in row else row["_id"]),
            f"{row.get('title', '')} {row.get('text', '')}".strip(),
        )
        for row in corpus_table
    ]
    queries = {
        str(row["id"] if "id" in row else row["_id"]): row.get("text", row.get("query", ""))
        for row in query_table
    }
    qrels: dict[str, set[str]] = {}
    for row in qrel_table:
        if float(row.get("score", 1)) > 0:
            qrels.setdefault(str(row["query-id"]), set()).add(str(row["corpus-id"]))
    queries = {query_id: query for query_id, query in queries.items() if query_id in qrels}
    document_ids = {document.doc_id for document in documents}
    excluded_ids = {
        query_id: {query_id} if query_id in document_ids else set()
        for query_id in queries
    }
    return Benchmark(documents, queries, qrels, excluded_ids)


def _qasper_paper_id(chunk_id: str) -> str:
    """Return the arXiv paper ID from LMEB's ``<paper>_<chunk>`` ID."""
    paper_id, separator, chunk_number = chunk_id.rpartition("_")
    return paper_id if separator and chunk_number.isdigit() else chunk_id


def load_qasper_hf(
    *,
    scope: str = "global",
    dataset_id: str = DEFAULT_QASPER_DATASET,
    split: str = "test",
    cache_dir: str | Path | None = None,
    query_limit: int | None = None,
) -> Benchmark:
    """Load LMEB-QASPER for global or known-paper chunk retrieval.

    ``global`` searches the shared paragraph corpus and is a custom extension.
    ``paper`` uses LMEB's per-query ``top_ranked`` candidate IDs, which scope
    retrieval to the paper associated with the original QASPER question.
    """
    if scope not in QASPER_SCOPES:
        raise ValueError(f"unsupported QASPER scope {scope!r}; expected one of {QASPER_SCOPES}")

    corpus_table = _load_hf_split(dataset_id, "QASPER-corpus", split, cache_dir)
    query_table = _load_hf_split(dataset_id, "QASPER-queries", split, cache_dir)
    qrel_table = _load_hf_split(dataset_id, "QASPER-qrels", split, cache_dir)

    documents = []
    for row in corpus_table:
        chunk_id = str(row["id"])
        section = str(row.get("title") or "")
        paragraph = str(row.get("text") or "")
        documents.append(
            Document(
                chunk_id,
                f"{section} {paragraph}".strip(),
                _qasper_paper_id(chunk_id),
                {"section": section},
            )
        )

    qrels: dict[str, set[str]] = {}
    for row in qrel_table:
        if float(row.get("score", 1)) > 0:
            qrels.setdefault(str(row["query-id"]), set()).add(str(row["corpus-id"]))

    all_queries = {
        str(row["id"]): str(row.get("text", row.get("query", "")))
        for row in query_table
    }
    query_ids = [query_id for query_id in all_queries if query_id in qrels]

    candidate_ids = None
    if scope == "paper":
        candidate_table = _load_hf_split(
            dataset_id,
            "QASPER-top_ranked",
            split,
            cache_dir,
        )
        candidate_ids = {
            str(row["query-id"]): {str(chunk_id) for chunk_id in row["corpus-ids"]}
            for row in candidate_table
        }
        query_ids = [query_id for query_id in query_ids if query_id in candidate_ids]

    if query_limit is not None:
        query_ids = query_ids[:query_limit]
    queries = {query_id: all_queries[query_id] for query_id in query_ids}
    qrels = {query_id: qrels[query_id] for query_id in query_ids}
    if candidate_ids is not None:
        candidate_ids = {query_id: candidate_ids[query_id] for query_id in query_ids}
        missing_gold = {
            query_id: qrels[query_id] - candidate_ids[query_id]
            for query_id in query_ids
            if not qrels[query_id] <= candidate_ids[query_id]
        }
        if missing_gold:
            query_id = next(iter(missing_gold))
            raise ValueError(
                f"QASPER candidate pool for {query_id} omits gold chunks: "
                f"{sorted(missing_gold[query_id])}"
            )

    return Benchmark(
        documents,
        queries,
        qrels,
        {query_id: set() for query_id in queries},
        {
            query_id: {
                "scope": scope,
                "candidate_count": (
                    len(candidate_ids[query_id]) if candidate_ids is not None else len(documents)
                ),
            }
            for query_id in queries
        },
        candidate_ids,
    )
