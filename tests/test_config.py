from pathlib import Path
import os

from src.config import load_config, resolve_project_paths, set_global_seed


def test_load_config_merges_base_and_local_smoke(tmp_path):
    cfg = load_config("configs/local_smoke.yaml")
    assert cfg["dataset"]["revision"] == "ca3fedcbb089bf65f7e4cfb03316cce4bd0d779f"
    assert cfg["phobert"]["revision"] == "01daacda68afe13d83023d16ec647239e344a1e6"
    assert cfg["phobert"]["max_length"] == 256
    assert cfg["phobert"]["batch_size"] == 16
    assert cfg["project"]["seeds"] == [42, 17, 2026]
    assert cfg["sampling"]["train_max_rows"] == 30000


def test_load_config_kaggle_profile():
    cfg = load_config("configs/kaggle_full.yaml")
    assert cfg["runtime"]["profile_name"] == "kaggle_full"
    assert cfg["phobert"]["batch_size"] == 64
    assert cfg["sampling"]["train_max_rows"] == 100000


def test_resolve_project_paths_creates_and_uses_root(tmp_path):
    cfg = load_config("configs/local_smoke.yaml")
    paths = resolve_project_paths(cfg, root=tmp_path)
    assert paths["root"] == tmp_path
    assert paths["raw_dir"] == tmp_path / "data" / "raw"
    assert paths["processed_dir"] == tmp_path / "data" / "processed"
    assert paths["outputs_dir"] == tmp_path / "outputs"
    assert paths["models_dir"] == tmp_path / "models"
    assert paths["reports_dir"] == tmp_path / "reports"
    assert paths["aliases_path"].name == "location_aliases_v1.yaml"


def test_set_global_seed_determinism():
    import random
    import numpy as np

    set_global_seed(42)
    a = (random.random(), float(np.random.rand()))
    set_global_seed(42)
    b = (random.random(), float(np.random.rand()))
    assert a == b
