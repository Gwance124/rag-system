#!/usr/bin/env python3
"""Download pinned Hugging Face assets on an internet-connected laptop."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_system.datasets.browsecomp_plus import (
    CORPUS_DATASET_ID,
    CORPUS_DATASET_REVISION,
    QUERY_DATASET_ID,
    QUERY_DATASET_REVISION,
)
from rag_system.offline_assets import (
    component_files,
    load_asset_manifest,
    write_asset_manifest,
)


MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
MODEL_RELATIVE_PATH = "models/Qwen--Qwen3.6-27B"
TOKENIZER_RELATIVE_PATH = "tokenizers/Qwen--Qwen3.6-27B"
QUERY_RELATIVE_PATH = "datasets/Tevatron--browsecomp-plus"
CORPUS_RELATIVE_PATH = "datasets/Tevatron--browsecomp-plus-corpus"
TOKENIZER_FILES = (
    "LICENSE",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)


def _snapshot_download(**kwargs):
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "install the staging extra first: python -m pip install -e '.[staging]'"
        ) from exc
    return snapshot_download(**kwargs)


def _download(
    output_root: Path,
    *,
    repo_id: str,
    repo_type: str,
    revision: str,
    relative_path: str,
    allow_patterns: tuple[str, ...] | None,
    token: str | None,
    checksums: bool,
) -> dict:
    destination = output_root / relative_path
    destination.mkdir(parents=True, exist_ok=True)
    _snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=destination,
        allow_patterns=list(allow_patterns) if allow_patterns else None,
        token=token,
    )
    return {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "revision": revision,
        "relative_path": relative_path,
        "checksums_recorded": checksums,
        "files": component_files(destination, checksums=checksums),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a checksummed, scp-ready BrowseComp-Plus/Qwen bundle."
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--component",
        action="append",
        choices=("datasets", "tokenizer", "model"),
        help="component to stage; repeat as needed (default: all)",
    )
    parser.add_argument(
        "--token",
        help="optional Hugging Face token; prefer HF_TOKEN instead of shell history",
    )
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="record sizes only; faster for the ~56 GB model but weaker verification",
    )
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    selected = set(args.component or ("datasets", "tokenizer", "model"))
    existing = load_asset_manifest(output_root)
    assets = dict(existing.get("assets", {}))
    checksums = not args.skip_checksums

    if "datasets" in selected:
        print(f"staging {QUERY_DATASET_ID}@{QUERY_DATASET_REVISION}", flush=True)
        assets["browsecomp_queries"] = _download(
            output_root,
            repo_id=QUERY_DATASET_ID,
            repo_type="dataset",
            revision=QUERY_DATASET_REVISION,
            relative_path=QUERY_RELATIVE_PATH,
            allow_patterns=("README.md", "data/*.parquet"),
            token=args.token,
            checksums=checksums,
        )
        print(f"staging {CORPUS_DATASET_ID}@{CORPUS_DATASET_REVISION}", flush=True)
        assets["browsecomp_corpus"] = _download(
            output_root,
            repo_id=CORPUS_DATASET_ID,
            repo_type="dataset",
            revision=CORPUS_DATASET_REVISION,
            relative_path=CORPUS_RELATIVE_PATH,
            allow_patterns=("README.md", "data/*.parquet"),
            token=args.token,
            checksums=checksums,
        )

    if "tokenizer" in selected:
        print(f"staging tokenizer {MODEL_ID}@{MODEL_REVISION}", flush=True)
        assets["qwen3_6_27b_tokenizer"] = _download(
            output_root,
            repo_id=MODEL_ID,
            repo_type="model",
            revision=MODEL_REVISION,
            relative_path=TOKENIZER_RELATIVE_PATH,
            allow_patterns=TOKENIZER_FILES,
            token=args.token,
            checksums=checksums,
        )

    if "model" in selected:
        print(f"staging model {MODEL_ID}@{MODEL_REVISION}", flush=True)
        assets["qwen3_6_27b_model"] = _download(
            output_root,
            repo_id=MODEL_ID,
            repo_type="model",
            revision=MODEL_REVISION,
            relative_path=MODEL_RELATIVE_PATH,
            allow_patterns=None,
            token=args.token,
            checksums=checksums,
        )

    path = write_asset_manifest(output_root, assets)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
