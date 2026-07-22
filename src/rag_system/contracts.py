"""Immutable records shared by dataset preparation and later workflows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkQuery:
    query_id: str
    question: str
    reference_answer: str
    evidence_document_ids: tuple[str, ...]
    gold_document_ids: tuple[str, ...]


@dataclass(frozen=True)
class CorpusDocument:
    document_id: str
    text: str
    url: str


@dataclass(frozen=True)
class SearchCandidate:
    """Full-text document returned by the underlying document retriever."""

    document_id: str
    score: float
    text: str


@dataclass(frozen=True)
class StandardSearchHit:
    """One document exposed to the agent by the Standard search tool."""

    document_id: str
    score: float
    snippet: str
    snippet_token_count: int


@dataclass(frozen=True)
class StandardSearchTrace:
    """Validated output of one BrowseComp-Plus Standard search call."""

    query: str
    top_k: int
    snippet_max_tokens: int
    hits: tuple[StandardSearchHit, ...]


@dataclass(frozen=True)
class DatasetSplit:
    seed: str
    algorithm: str
    development_query_ids: tuple[str, ...]
    held_out_query_ids: tuple[str, ...]
    sha256: str


@dataclass(frozen=True)
class DatasetValidationReport:
    query_count: int
    corpus_document_count: int
    evidence_label_count: int
    gold_label_count: int
    queries_with_gold_outside_evidence: int
    duplicate_query_ids: tuple[str, ...]
    duplicate_document_ids: tuple[str, ...]
    missing_evidence_document_ids: tuple[str, ...]
    missing_gold_document_ids: tuple[str, ...]
    empty_corpus_document_ids: tuple[str, ...]
