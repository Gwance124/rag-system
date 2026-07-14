from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    paper_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def paper_key(self) -> str:
        return self.paper_id or self.doc_id


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    score: float
    paper_id: str | None = None


@dataclass(frozen=True)
class RetrievalConfig:
    top_n: int = 100
    top_k: int = 10
    rrf_k: int = 60
    sparse_weight: float = 0.5
    dense_weight: float = 0.5
    rewrite: bool = False


@dataclass
class RetrievalResult:
    hits: list[SearchHit]
    timings_ms: dict[str, float]
