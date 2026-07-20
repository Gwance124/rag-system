"""Minimal adapters for vLLM embeddings and Qdrant vector search."""

from __future__ import annotations

import json
from urllib.error import HTTPError
import urllib.request
import uuid
from collections.abc import Callable, Iterable, Sequence

from retrieval.types import Document, SearchHit


class VllmEmbeddingClient:
    """OpenAI-compatible embeddings client for the lab's vLLM endpoint."""

    def __init__(
        self,
        base_url: str = "http://192.168.3.4:8000/v1",
        model: str = "nvidia/llama-nv-embed-reasoning-3b",
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.timeout = timeout

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        body = json.dumps({"model": self.model, "input": list(texts)}).encode()
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.load(response)
        return [item["embedding"] for item in sorted(payload["data"], key=lambda item: item["index"])]


class QdrantIndex:
    """Minimal Qdrant REST adapter; keeps the dense dependency off the client."""

    def __init__(self, collection: str, embedder: VllmEmbeddingClient, base_url: str):
        self.collection = collection
        self.embedder = embedder
        self.base_url = base_url.rstrip("/")

    def _request(self, method: str, path: str, body: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        with urllib.request.urlopen(request, timeout=self.embedder.timeout) as response:
            return json.load(response)

    def create(
        self,
        documents: Iterable[Document],
        batch_size: int = 32,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        documents = list(documents)
        if not documents:
            return
        first_batch = documents[:batch_size]
        vectors = self.embedder.embed([self.embedder.passage_prefix + document.text for document in first_batch])
        try:
            self._request(
                "PUT",
                f"/collections/{self.collection}",
                {"vectors": {"size": len(vectors[0]), "distance": "Cosine"}},
            )
        except HTTPError as exc:
            if exc.code != 409:
                raise
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            batch_vectors = vectors if start == 0 else self.embedder.embed(
                [self.embedder.passage_prefix + document.text for document in batch]
            )
            self._request(
                "PUT",
                f"/collections/{self.collection}/points?wait=true",
                {
                    "points": [
                        {
                            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, document.doc_id)),
                            "vector": vector,
                            "payload": {
                                "doc_id": document.doc_id,
                                "paper_id": document.paper_id,
                            },
                        }
                        for document, vector in zip(batch, batch_vectors)
                    ]
                },
            )
            if progress:
                progress(min(start + len(batch), len(documents)), len(documents))

    def embed_query(self, query: str) -> list[float]:
        return self.embedder.embed([self.embedder.query_prefix + query])[0]

    def search_vector(
        self,
        vector: Sequence[float],
        top_n: int = 100,
        allowed_ids: Iterable[str] | None = None,
    ) -> list[SearchHit]:
        allowed = None if allowed_ids is None else sorted(set(allowed_ids))
        if top_n <= 0 or (allowed is not None and not allowed):
            return []
        body = {
            "vector": list(vector),
            "limit": min(top_n, len(allowed)) if allowed is not None else top_n,
            "with_payload": True,
        }
        if allowed is not None:
            body["filter"] = {
                "must": [{"key": "doc_id", "match": {"any": allowed}}]
            }
        response = self._request(
            "POST",
            f"/collections/{self.collection}/points/search",
            body,
        )
        return [
            SearchHit(
                doc_id=hit["payload"]["doc_id"],
                score=hit["score"],
                paper_id=hit["payload"].get("paper_id"),
            )
            for hit in response.get("result", [])
        ]
