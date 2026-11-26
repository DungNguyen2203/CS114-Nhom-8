"""Pinned Hugging Face dataset shard downloader with SHA256 verification."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def download_dataset_shards(
    raw_dir: Path,
    repo_id: str,
    revision: str,
    rules_version: str,
    expected_shards: int = 10,
    max_shards: int | None = None,
) -> dict[str, Any]:
    """Download pinned parquet shards into `raw_dir` and record manifest.json."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    n_shards = min(expected_shards, max_shards) if max_shards is not None else expected_shards
    lock = json.loads(Path(__file__).with_name("dataset_lock.json").read_text(encoding="utf-8"))
    if (repo_id, revision) == (lock["repo_id"], lock["revision"]):
        expected = lock["sha256"]
    else:
        from huggingface_hub import HfApi

        info = HfApi().dataset_info(repo_id, revision=revision, files_metadata=True)
        if info.sha != revision:
            raise ValueError("Dataset revision must be an immutable commit SHA.")
        expected = {s.rfilename: s.lfs.sha256 for s in info.siblings if s.lfs}

    shards_meta: list[dict[str, Any]] = []
    total_rows = 0
    for idx in range(n_shards):
        fname = f"shard_{idx:04d}.parquet"
        if fname not in expected:
            raise ValueError(f"Missing upstream SHA256 for {fname} at {revision}")
        local_path = raw_dir / fname
        if not local_path.exists():
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                repo_type="dataset",
                revision=revision,
                local_dir=str(raw_dir),
            )
            local_path = Path(downloaded)

        digest = sha256_file(local_path)
        if digest != expected[fname]:
            raise ValueError(f"SHA256 mismatch for {fname} at {repo_id}@{revision}.")
        with pq.ParquetFile(local_path) as pf:
            num_rows = int(pf.metadata.num_rows)
        total_rows += num_rows
        shards_meta.append(
            {
                "shard_idx": idx,
                "filename": fname,
                "path": str(local_path),
                "size_bytes": local_path.stat().st_size,
                "sha256": digest,
                "num_rows": num_rows,
            }
        )

    manifest = {
        "repo_id": repo_id,
        "revision": revision,
        "rules_version": rules_version,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_rows": total_rows,
        "shards": shards_meta,
    }
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
