from __future__ import annotations

import re
from collections.abc import Callable, Iterable

import bm25s
from bm25s.tokenization import Tokenizer

from retrieval.types import Document, SearchHit

_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


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

    def search(self, query: str, top_n: int = 100) -> list[SearchHit]:
        if not self.documents or top_n <= 0:
            return []

        query_tokens = self._tokenizer.tokenize(
            [query], update_vocab=False, show_progress=False
        )
        indices, scores = self._retriever.retrieve(
            query_tokens,
            k=min(top_n, len(self.documents)),
            show_progress=False,
        )
        return [
            SearchHit(
                doc_id=self.documents[index].doc_id,
                score=float(score),
                paper_id=self.documents[index].paper_id,
            )
            for index, score in zip(indices[0], scores[0])
            if score > 0
        ]
