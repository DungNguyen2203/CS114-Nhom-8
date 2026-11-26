"""Bind prepared data, embeddings and experiment outputs to their actual input files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.data.download import sha256_file


PREPARED_FILES = (
    "clean.parquet", "noisy_flagged.parquet", "rental_unknown.parquet",
    "splits/scenario_a_manifest.parquet", "splits/scenario_b_manifest.parquet",
)
CACHE_FILES = ("phobert_cache.fp16.npy", "phobert_index.parquet")


def fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prepare_settings(cfg: dict, paths: dict) -> dict:
    return {
        **{key: cfg[key] for key in ("dataset", "cleaning", "dedup", "splits", "sampling")},
        **{key: cfg["project"][key] for key in ("rules_version", "text_version", "mask_token", "primary_seed")},
        "aliases_sha256": sha256_file(paths["aliases_path"]),
    }


def file_hashes(root: Path, names) -> dict[str, str]:
    return {str(name): sha256_file(root / name) for name in names}


def read_prepared(cfg: dict, paths: dict) -> dict:
    root = paths["processed_dir"]
    summary = json.loads((root / "splits/split_manifest.json").read_text(encoding="utf-8"))
    prov = summary.get("provenance", {})
    if prov.get("settings") != prepare_settings(cfg, paths) or not prov.get("prepare_id"):
        raise ValueError("Prepared data/config mismatch. Re-run prepare_data.")
    if prov.get("files") != file_hashes(root, PREPARED_FILES):
        raise ValueError("Prepared parquet or split hash mismatch. Re-run prepare_data.")
    return prov


def check_cache(cfg: dict, paths: dict, prepared: dict) -> dict:
    root = paths["processed_dir"] / "embeddings"
    meta = json.loads((root / "cache_meta.json").read_text(encoding="utf-8"))
    if (
        meta.get("prepare_id") != prepared["prepare_id"]
        or meta.get("encoder") != cfg["phobert"]
        or meta.get("text_version") != cfg["project"]["text_version"]
        or meta.get("model_sha") != cfg["phobert"]["revision"]
        or meta.get("files") != file_hashes(root, CACHE_FILES)
    ):
        raise ValueError("Embedding cache/config/input mismatch. Re-run extract_embeddings.")
    return meta


def training_settings(cfg: dict) -> str:
    return fingerprint({k: cfg[k] for k in ("models", "features", "project", "conformal", "leakage", "audits")})
