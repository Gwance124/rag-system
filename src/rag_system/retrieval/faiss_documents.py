"""Document-level FAISS search over official Tevatron index shards."""

from __future__ import annotations

import glob
import pickle
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from rag_system.contracts import CorpusDocument, SearchCandidate


class QueryEncoder(Protocol):
    def encode(self, query: str) -> list[float]: ...


class FaissIndexError(RuntimeError):
    """Raised when official index shards violate the expected format."""


class FaissDocumentBackend:
    """Search a fixed document index using query vectors produced on g3."""

    def __init__(
        self,
        index_path_pattern: str,
        query_encoder: QueryEncoder,
        corpus_documents: Iterable[CorpusDocument],
    ) -> None:
        try:
            import faiss
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "FAISS document search requires numpy and faiss-cpu"
            ) from exc

        shard_paths = tuple(sorted(glob.glob(index_path_pattern)))
        if not shard_paths:
            raise FileNotFoundError(
                f"no index shards match pattern: {index_path_pattern}"
            )

        lookup: list[str] = []
        index = None
        dimension = None
        for shard_path in shard_paths:
            with Path(shard_path).open("rb") as handle:
                representations, shard_lookup = pickle.load(handle)
            matrix = np.asarray(representations, dtype="float32")
            if matrix.ndim != 2 or matrix.shape[0] != len(shard_lookup):
                raise FaissIndexError(f"invalid index shard: {shard_path}")
            if dimension is None:
                dimension = int(matrix.shape[1])
                index = faiss.IndexFlatIP(dimension)
            elif matrix.shape[1] != dimension:
                raise FaissIndexError(
                    f"embedding dimension changed in shard: {shard_path}"
                )
            index.add(matrix)
            lookup.extend(str(document_id) for document_id in shard_lookup)

        documents = {document.document_id: document.text for document in corpus_documents}
        missing = [document_id for document_id in lookup if document_id not in documents]
        if missing:
            raise FaissIndexError(
                f"{len(missing)} indexed document IDs are absent from the corpus"
            )

        self._faiss = faiss
        self._np = np
        self._index = index
        self._dimension = dimension
        self._lookup = tuple(lookup)
        self._documents = documents
        self._query_encoder = query_encoder

    def search(self, query: str, top_k: int) -> list[SearchCandidate]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        vector = self._np.asarray(self._query_encoder.encode(query), dtype="float32")
        if vector.ndim != 1 or vector.shape[0] != self._dimension:
            raise FaissIndexError(
                f"query dimension {vector.shape} does not match index dimension "
                f"{self._dimension}"
            )
        scores, indices = self._index.search(vector.reshape(1, -1), top_k)
        results = []
        for score, index_position in zip(scores[0], indices[0]):
            if index_position < 0:
                continue
            document_id = self._lookup[int(index_position)]
            results.append(
                SearchCandidate(
                    document_id=document_id,
                    score=float(score),
                    text=self._documents[document_id],
                )
            )
        return results
