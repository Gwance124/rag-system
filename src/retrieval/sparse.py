from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable

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
    ) -> None:
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self._postings: dict[str, dict[int, int]] = defaultdict(dict)
        self._lengths: list[int] = []

        for index, document in enumerate(self.documents):
            counts = Counter(tokenize(document.text))
            self._lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                self._postings[term][index] = frequency

        self._avgdl = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        size = len(self.documents)
        self._idf = {
            term: math.log(1 + (size - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in self._postings.items()
        }

    def search(self, query: str, top_n: int = 100) -> list[SearchHit]:
        if not self.documents or top_n <= 0:
            return []

        scores: dict[int, float] = defaultdict(float)
        for term in tokenize(query):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf[term]
            for index, frequency in postings.items():
                length = self._lengths[index]
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / self._avgdl
                )
                scores[index] += idf * frequency * (self.k1 + 1) / denominator

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [
            SearchHit(
                doc_id=self.documents[index].doc_id,
                score=score,
                paper_id=self.documents[index].paper_id,
            )
            for index, score in ranked[:top_n]
        ]
