"""Configuration loading, project path resolution, and reproducibility utilities."""

from __future__ import annotations

import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override dictionary into a copy of base."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file, resolving any single-level `inherits` base file."""
    path = Path(config_path)
    if not path.is_absolute():
        repo_root = Path(__file__).resolve().parent.parent
        candidate = repo_root / path
        if candidate.exists():
            path = candidate
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    inherits = cfg.pop("inherits", None)
    if inherits:
        base_path = (path.parent / inherits).resolve()
        base_cfg = load_config(base_path)
        cfg = _deep_merge(base_cfg, cfg)
    return cfg


def resolve_project_paths(cfg: dict[str, Any], root: Path | None = None) -> dict[str, Path]:
    """Resolve and return canonical project directories and ensure Hugging Face cache stays off C:."""
    if root is not None:
        base_root = Path(root)
    else:
        env_root = os.environ.get("VN_RE_PROJECT_ROOT")
        base_root = Path(env_root) if env_root else Path(__file__).resolve().parent.parent

    paths_cfg = cfg.get("paths", {})
    resolved = {
        "root": base_root,
        "raw_dir": base_root / paths_cfg.get("raw_dir", "data/raw"),
        "processed_dir": base_root / paths_cfg.get("processed_dir", "data/processed"),
        "models_dir": base_root / paths_cfg.get("models_dir", "models"),
        "outputs_dir": base_root / paths_cfg.get("outputs_dir", "outputs"),
        "reports_dir": base_root / paths_cfg.get("reports_dir", "reports"),
        "aliases_path": base_root / paths_cfg.get("aliases_path", "configs/location_aliases_v1.yaml"),
    }

    hf_home = cfg.get("runtime", {}).get("hf_home")
    if hf_home and "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = str(hf_home)
    return resolved


def set_global_seed(seed: int) -> None:
    """Set Python, NumPy, and (if installed) PyTorch deterministic random seeds."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
