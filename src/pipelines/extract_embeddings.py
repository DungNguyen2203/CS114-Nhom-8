"""TV3 CLI pipeline: extract and cache frozen PhoBERT embeddings across Scenario A & B union."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_config, resolve_project_paths, set_global_seed
from src.features.phobert_embedder import PhoBERTEmbedder


def run_extract_embeddings(cfg: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    """Compute and cache PhoBERT embeddings for the union of Scenario A & B row_ids."""
    set_global_seed(int(cfg["project"]["primary_seed"]))
    splits_dir = paths["processed_dir"] / "splits"
    cache_dir = paths["processed_dir"] / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)

    man_a = pd.read_parquet(splits_dir / "scenario_a_manifest.parquet", columns=["row_id"])
    man_b = pd.read_parquet(splits_dir / "scenario_b_manifest.parquet", columns=["row_id"])
    union_ids = set(man_a["row_id"].astype(str)).union(set(man_b["row_id"].astype(str)))

    import pyarrow.parquet as pq

    needed_cols = [
        "row_id",
        "property_type_name",
        "province_norm",
        "district_norm",
        "name_masked",
        "description_masked",
    ]
    parts = []
    pf = pq.ParquetFile(paths["processed_dir"] / "clean.parquet")
    for batch in pf.iter_batches(batch_size=100_000, columns=needed_cols):
        bdf = batch.to_pandas()
        sub = bdf[bdf["row_id"].astype(str).isin(union_ids)]
        if len(sub) > 0:
            parts.append(sub)
    target_df = pd.concat(parts, axis=0, ignore_index=True) if parts else pd.DataFrame(columns=needed_cols)
    target_df = target_df.sort_values("row_id").reset_index(drop=True)
    print(f"[TV3] Extracting PhoBERT embeddings for {len(target_df):,} unique union rows...", flush=True)

    ph_cfg = cfg["phobert"]
    embedder = PhoBERTEmbedder(
        model_name=ph_cfg["model_name"],
        revision=ph_cfg["revision"],
        max_length=int(ph_cfg["max_length"]),
        batch_size=int(ph_cfg["batch_size"]),
        fp16=bool(ph_cfg["fp16"]),
        normalize_l2=bool(ph_cfg["normalize_l2"]),
        chunk_pooling=bool(ph_cfg.get("chunk_pooling", False)),
        device=cfg.get("runtime", {}).get("device", "auto"),
    )
    stats = embedder.extract_and_cache(
        target_df,
        cache_dir=cache_dir,
        text_version=cfg["project"]["text_version"],
        mask_token=cfg["project"]["mask_token"],
    )
    (cache_dir / "embedding_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="TV3 PhoBERT embedding extraction pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = resolve_project_paths(cfg)
    stats = run_extract_embeddings(cfg, paths)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
