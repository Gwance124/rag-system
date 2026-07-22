#!/usr/bin/env python3
"""Verify selected offline assets after scp/rsync."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_system.offline_assets import OfflineAssetError, verify_asset_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--component",
        action="append",
        choices=(
            "browsecomp_queries",
            "browsecomp_corpus",
            "qwen3_5_27b_tokenizer",
            "qwen3_5_27b_model",
        ),
        help="verify only this component; repeat as needed (default: all in manifest)",
    )
    parser.add_argument(
        "--sizes-only",
        action="store_true",
        help="skip SHA-256 recomputation and verify only file presence/size",
    )
    args = parser.parse_args()
    try:
        summary = verify_asset_manifest(
            args.root,
            args.component,
            verify_checksums=not args.sizes_only,
        )
    except OfflineAssetError as exc:
        parser.error(str(exc))
    print(
        "verified "
        f"{summary['components']} component(s), {summary['files']} files, "
        f"{summary['bytes']} bytes"
    )


if __name__ == "__main__":
    main()
