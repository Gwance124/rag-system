"""Content-addressed manifests for laptop-to-server asset transfers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ASSET_MANIFEST_NAME = "OFFLINE_ASSET_MANIFEST.json"


class OfflineAssetError(RuntimeError):
    """Raised when an offline asset bundle is incomplete or corrupted."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def component_files(component_dir: str | Path, *, checksums: bool = True) -> list[dict[str, Any]]:
    root = Path(component_dir)
    if not root.is_dir():
        raise OfflineAssetError(f"asset component directory does not exist: {root}")
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        if relative.parts[:2] == (".cache", "huggingface"):
            continue
        files.append(
            {
                "path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path) if checksums else None,
            }
        )
    if not files:
        raise OfflineAssetError(f"asset component contains no files: {root}")
    return files


def load_asset_manifest(root: str | Path) -> dict[str, Any]:
    path = Path(root) / ASSET_MANIFEST_NAME
    if not path.is_file():
        return {"schema_version": "1.0", "assets": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OfflineAssetError(f"cannot read asset manifest: {path}") from exc
    if data.get("schema_version") != "1.0" or not isinstance(data.get("assets"), dict):
        raise OfflineAssetError(f"unsupported asset manifest schema: {path}")
    return data


def write_asset_manifest(root: str | Path, assets: Mapping[str, Mapping[str, Any]]) -> Path:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    path = root_path / ASSET_MANIFEST_NAME
    payload = json.dumps(
        {
            "schema_version": "1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "assets": dict(sorted(assets.items())),
        },
        indent=2,
        sort_keys=True,
    ) + "\n"

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=root_path,
            prefix=f".{ASSET_MANIFEST_NAME}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def verify_asset_manifest(
    root: str | Path,
    components: Iterable[str] | None = None,
    *,
    verify_checksums: bool = True,
) -> dict[str, int]:
    """Verify selected components after transfer and return file/byte totals."""
    root_path = Path(root)
    manifest = load_asset_manifest(root_path)
    assets = manifest["assets"]
    selected = tuple(components) if components is not None else tuple(sorted(assets))
    if not selected:
        raise OfflineAssetError("asset manifest contains no components")

    file_count = 0
    byte_count = 0
    errors = []
    for name in selected:
        asset = assets.get(name)
        if not isinstance(asset, Mapping):
            errors.append(f"component {name!r} is absent from the manifest")
            continue
        relative_root = asset.get("relative_path")
        files = asset.get("files")
        if not isinstance(relative_root, str) or not isinstance(files, list):
            errors.append(f"component {name!r} has invalid manifest metadata")
            continue
        component_root = root_path / relative_root
        for record in files:
            if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
                errors.append(f"component {name!r} contains an invalid file record")
                continue
            path = component_root / record["path"]
            if not path.is_file():
                errors.append(f"missing file: {path}")
                continue
            expected_size = record.get("size_bytes")
            actual_size = path.stat().st_size
            if actual_size != expected_size:
                errors.append(
                    f"size mismatch for {path}: expected {expected_size}, found {actual_size}"
                )
                continue
            expected_sha = record.get("sha256")
            if verify_checksums and expected_sha and sha256_file(path) != expected_sha:
                errors.append(f"checksum mismatch for {path}")
                continue
            file_count += 1
            byte_count += actual_size

    if errors:
        preview = "; ".join(errors[:10])
        suffix = f"; plus {len(errors) - 10} more" if len(errors) > 10 else ""
        raise OfflineAssetError(preview + suffix)
    return {"components": len(selected), "files": file_count, "bytes": byte_count}
