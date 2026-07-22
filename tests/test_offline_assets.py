from __future__ import annotations

import json

import pytest

from rag_system.offline_assets import (
    ASSET_MANIFEST_NAME,
    OfflineAssetError,
    component_files,
    verify_asset_manifest,
    write_asset_manifest,
)


def test_asset_manifest_round_trip_and_component_selection(tmp_path):
    dataset = tmp_path / "datasets" / "benchmark"
    model = tmp_path / "models" / "model"
    dataset.mkdir(parents=True)
    model.mkdir(parents=True)
    (dataset / "part.parquet").write_bytes(b"dataset")
    (model / "config.json").write_text("{}")

    write_asset_manifest(
        tmp_path,
        {
            "dataset": {
                "repo_id": "example/dataset",
                "repo_type": "dataset",
                "revision": "abc",
                "relative_path": "datasets/benchmark",
                "files": component_files(dataset),
            },
            "model": {
                "repo_id": "example/model",
                "repo_type": "model",
                "revision": "def",
                "relative_path": "models/model",
                "files": component_files(model),
            },
        },
    )

    summary = verify_asset_manifest(tmp_path, ["dataset"])
    assert summary == {"components": 1, "files": 1, "bytes": 7}
    manifest = json.loads((tmp_path / ASSET_MANIFEST_NAME).read_text())
    assert set(manifest["assets"]) == {"dataset", "model"}


def test_asset_manifest_detects_transfer_corruption(tmp_path):
    component = tmp_path / "datasets" / "benchmark"
    component.mkdir(parents=True)
    file_path = component / "part.parquet"
    file_path.write_bytes(b"before")
    write_asset_manifest(
        tmp_path,
        {
            "dataset": {
                "relative_path": "datasets/benchmark",
                "files": component_files(component),
            }
        },
    )
    file_path.write_bytes(b"after!")
    with pytest.raises(OfflineAssetError, match="checksum mismatch"):
        verify_asset_manifest(tmp_path)


def test_asset_manifest_reports_missing_selected_component(tmp_path):
    write_asset_manifest(tmp_path, {})
    with pytest.raises(OfflineAssetError, match="absent"):
        verify_asset_manifest(tmp_path, ["missing"])
