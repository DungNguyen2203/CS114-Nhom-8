import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config, resolve_project_paths
from src.data.download import download_dataset_shards
from src.features.phobert_embedder import load_embeddings_for_row_ids, save_embedding_cache
from src.pipelines.prepare_data import _merge_shard_parquets, run_prepare_data
from src.pipelines.provenance import check_cache, read_prepared
from tests.test_prepare_data_pipeline import _write_synthetic_shards


def test_offline_raw_lock_requires_authentic_bytes(tmp_path, monkeypatch):
    file = tmp_path / "shard_0000.parquet"
    pd.DataFrame({"v": [42]}).to_parquet(file)
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    lock = {"repo_id": "dataset", "revision": "f" * 40, "sha256": {file.name: digest}}
    (tmp_path / "dataset_lock.json").write_text(json.dumps(lock))
    monkeypatch.setattr("src.data.download.__file__", str(tmp_path / "download.py"))
    args = (tmp_path, lock["repo_id"], lock["revision"], "v", 1)
    assert download_dataset_shards(*args)["shards"][0]["sha256"] == digest
    manifest = (tmp_path / "manifest.json").read_bytes()
    pd.DataFrame({"v": [123]}).to_parquet(file)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        download_dataset_shards(*args)
    assert (tmp_path / "manifest.json").read_bytes() == manifest


def test_empty_merge_replaces_old_rows_and_partial_failure_preserves_destination(tmp_path):
    dest = tmp_path / "bucket.parquet"
    part = tmp_path / "part.parquet"
    df = pd.DataFrame({"price": [42.]})
    df.to_parquet(dest)
    df.iloc[:0].to_parquet(part)
    assert _merge_shard_parquets([part], dest) == 0
    assert pd.read_parquet(dest).equals(df.iloc[:0])
    before = dest.read_bytes()
    df.to_parquet(part)
    with pytest.raises(FileNotFoundError):
        _merge_shard_parquets([part, tmp_path / "missing"], dest)
    assert dest.read_bytes() == before


def test_prepared_and_embedding_checks_reject_mixed_runs_and_configs(tmp_path):
    cfg = load_config("configs/local_smoke.yaml")
    paths = resolve_project_paths(cfg, root=tmp_path)
    paths["aliases_path"] = Path("configs/location_aliases_v1.yaml")
    summary = run_prepare_data(cfg, paths, _write_synthetic_shards(paths["raw_dir"]))
    prepared = read_prepared(cfg, paths)
    assert prepared["prepare_id"] == summary["provenance"]["prepare_id"]
    assert "revision" not in prepared["source"]  # test shards supplied explicitly
    root = paths["processed_dir"] / "embeddings"
    save_embedding_cache(root, ["a"], np.ones((1, 2)), cfg["project"]["text_version"],
                         cfg["phobert"]["revision"], prepare_id=prepared["prepare_id"], encoder=cfg["phobert"])
    check_cache(cfg, paths, prepared)
    with pytest.raises(ValueError, match="cache/config/input"):
        check_cache(cfg, paths, {**prepared, "prepare_id": "other"})
    cfg["phobert"]["revision"] = "another-sha"
    with pytest.raises(ValueError, match="cache/config/input"):
        check_cache(cfg, paths, prepared)
    split = paths["processed_dir"] / "splits/scenario_a_manifest.parquet"
    pd.read_parquet(split).iloc[1:].to_parquet(split)
    with pytest.raises(ValueError, match="hash mismatch"):
        read_prepared(cfg, paths)


def test_embedding_corrupt_index_and_duplicate_ids(tmp_path):
    matrix = np.ones((2, 3))
    with pytest.raises(ValueError, match="unique"):
        save_embedding_cache(tmp_path, ["a", "a"], matrix, "v", "sha")
    save_embedding_cache(tmp_path, ["a", "b"], matrix, "v", "sha")
    index = tmp_path / "phobert_index.parquet"
    rows = pd.read_parquet(index)
    for offsets in ([1, 1], [-1, 1], [1, 2], [0.5, 1.0]):
        rows.assign(offset=offsets).to_parquet(index)
        with pytest.raises(ValueError, match="offsets"):
            load_embeddings_for_row_ids(["a"], tmp_path, "v")
