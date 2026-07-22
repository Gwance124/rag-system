#!/usr/bin/env python3
"""Load the p7 corpus/index once and serve persistent Standard search."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from rag_system.datasets.browsecomp_plus import iter_corpus_repository
from rag_system.retrieval.faiss_documents import FaissDocumentBackend
from rag_system.retrieval.remote_encoder import RemoteQueryEncoder
from rag_system.retrieval.standard import StandardSearchTool
from rag_system.serving.standard_search import serve_standard_search


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-repo", type=Path, required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--encoder-url", required=True)
    parser.add_argument("--datasets-cache", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8012)
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        parser.error(f"Transformers is required: {exc}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer_path.expanduser().resolve()),
        local_files_only=True,
    )
    encoder = RemoteQueryEncoder(args.encoder_url)
    documents = iter_corpus_repository(
        args.corpus_repo,
        cache_dir=args.datasets_cache,
    )
    backend = FaissDocumentBackend(args.index_path, encoder, documents)
    search_tool = StandardSearchTool(backend, tokenizer)
    serve_standard_search(search_tool, args.host, args.port)


if __name__ == "__main__":
    main()
