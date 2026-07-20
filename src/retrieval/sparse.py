"""In-memory BM25 indexing backed by bm25s."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

import bm25s
from bm25s.tokenization import Tokenizer

from retrieval.types import Document, SearchHit

_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


class BM25Index:
    """A small in-memory BM25 index for abstracts or chunk collections."""

    def __init__(
        self,
        documents: Iterable[Document],
        k1: float = 1.2,
        b: float = 0.75,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        self.documents = list(documents)
        self._tokenizer = Tokenizer(splitter=_TOKEN_RE.pattern, stopwords=[])
        self._retriever = bm25s.BM25(k1=k1, b=b, method="lucene")
        if self.documents:
            show_progress = progress is not None
            corpus_tokens = self._tokenizer.tokenize(
                [document.text for document in self.documents],
                show_progress=show_progress,
                leave_progress=show_progress,
            )
            self._retriever.index(
                corpus_tokens,
                show_progress=show_progress,
                leave_progress=show_progress,
            )
            if progress:
                progress(len(self.documents), len(self.documents))

    def search(
        self,
        query: str,
        top_n: int = 100,
        allowed_ids: Iterable[str] | None = None,
    ) -> list[SearchHit]:
        if not self.documents or top_n <= 0:
            return []

        allowed = None if allowed_ids is None else set(allowed_ids)
        if allowed is not None and not allowed:
            return []

        query_tokens = self._tokenizer.tokenize(
            [query], update_vocab=False, show_progress=False
        )
        # Candidate-scoped retrieval must score beyond the global top_n;
        # otherwise a valid in-paper paragraph could be discarded before the
        # paper filter is applied. BM25 statistics remain fixed to the shared
        # corpus so global and paper-scoped results stay comparable.
        retrieve_count = len(self.documents) if allowed is not None else top_n
        indices, scores = self._retriever.retrieve(
            query_tokens,
            k=min(retrieve_count, len(self.documents)),
            show_progress=False,
        )
        hits = []
        for index, score in zip(indices[0], scores[0]):
            document = self.documents[index]
            if score <= 0 or (allowed is not None and document.doc_id not in allowed):
                continue
            hits.append(SearchHit(document.doc_id, float(score), document.paper_id))
            if len(hits) == top_n:
                break
        return hits
