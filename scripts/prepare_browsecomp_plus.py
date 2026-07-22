#!/usr/bin/env python3
"""Decrypt and validate transferred BrowseComp-Plus files on solab-p7."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rag_system.datasets.browsecomp_plus import (
    DEFAULT_DEVELOPMENT_COUNT,
    DEFAULT_SPLIT_SEED,
    EXPECTED_CORPUS_DOCUMENT_COUNT,
    EXPECTED_QUERY_COUNT,
    BrowseCompPlusLoader,
    DatasetValidationError,
    make_hash_split,
    validate_benchmark,
    write_prepared_artifacts,
)


def _path_default(environment_name: str) -> Path | None:
    value = os.environ.get(environment_name)
    return Path(value) if value else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries-repo",
        type=Path,
        default=_path_default("RAG_BROWSECOMP_QUERIES_DIR"),
        help="transferred Tevatron--browsecomp-plus directory",
    )
    parser.add_argument(
        "--corpus-repo",
        type=Path,
        default=_path_default("RAG_BROWSECOMP_CORPUS_DIR"),
        help="transferred Tevatron--browsecomp-plus-corpus directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            _path_default("RAG_ARTIFACT_ROOT") / "datasets" / "browsecomp-plus"
            if _path_default("RAG_ARTIFACT_ROOT")
            else None
        ),
        help="private artifact directory for decrypted queries/split/manifest",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="local datasets Arrow cache; no network access is attempted",
    )
    parser.add_argument("--development-count", type=int, default=DEFAULT_DEVELOPMENT_COUNT)
    parser.add_argument("--split-seed", default=DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="allow small fixtures or an explicitly reviewed upstream revision change",
    )
    args = parser.parse_args()

    missing = [
        name
        for name, value in (
            ("--queries-repo", args.queries_repo),
            ("--corpus-repo", args.corpus_repo),
            ("--output-dir", args.output_dir),
        )
        if value is None
    ]
    if missing:
        parser.error(f"required path(s) missing: {', '.join(missing)}")

    loader = BrowseCompPlusLoader(
        args.queries_repo,
        args.corpus_repo,
        cache_dir=args.cache_dir,
    )
    try:
        print("loading and selectively decrypting query labels", flush=True)
        queries = loader.load_queries()
        print("validating labeled IDs against the canonical corpus", flush=True)
        validation = validate_benchmark(
            queries,
            loader.iter_corpus(),
            expected_query_count=EXPECTED_QUERY_COUNT,
            expected_corpus_document_count=EXPECTED_CORPUS_DOCUMENT_COUNT,
            strict_counts=not args.allow_count_mismatch,
        )
        split = make_hash_split(
            (query.query_id for query in queries),
            development_count=args.development_count,
            seed=args.split_seed,
        )
        manifest = write_prepared_artifacts(
            args.output_dir,
            queries,
            split,
            validation,
            queries_repository=args.queries_repo,
            corpus_repository=args.corpus_repo,
        )
    except (DatasetValidationError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))

    summary = {
        "output_dir": str(args.output_dir.expanduser().resolve()),
        "query_count": validation.query_count,
        "corpus_document_count": validation.corpus_document_count,
        "development_query_count": len(split.development_query_ids),
        "held_out_query_count": len(split.held_out_query_ids),
        "split_sha256": split.sha256,
        "query_artifact_sha256": manifest["artifacts"]["queries"]["sha256"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
