"""Exact and MinHash-LSH near-duplicate clustering without using price."""

from __future__ import annotations

import hashlib
import re

from datasketch import MinHash, MinHashLSH
import numpy as np
import pandas as pd


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _norm_dedup_text(text: str) -> str:
    tokens = _WORD_RE.findall((text or "").lower())
    return " ".join(tokens)


def _shingles(text: str, ngram_size: int = 3) -> list[str]:
    tokens = text.split()
    if len(tokens) <= ngram_size:
        return [text] if text else ["__EMPTY__"]
    return [" ".join(tokens[i : i + ngram_size]) for i in range(len(tokens) - ngram_size + 1)]


def _compute_minhash(shingles: list[str], num_perm: int) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    for sh in shingles:
        mh.update(sh.encode("utf-8"))
    return mh


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def assign_duplicate_groups(
    df: pd.DataFrame,
    threshold: float = 0.85,
    num_perm: int = 64,
    ngram_size: int = 3,
) -> pd.Series:
    """Assign duplicate cluster IDs from masked text + location + physical specs (never price)."""
    if len(df) == 0:
        return pd.Series([], index=df.index, dtype="object")

    text_col = "text_masked" if "text_masked" in df.columns else "name_masked"
    texts = [_norm_dedup_text(str(t or "")) for t in df[text_col]]
    provs = df["province_norm"].fillna("__MISSING__").astype(str).tolist()
    dists = df["district_norm"].fillna("__MISSING__").astype(str).tolist()
    ptypes = df["property_type_name"].fillna("__MISSING__").astype(str).tolist()
    areas = pd.to_numeric(df["area"], errors="coerce").fillna(-1.0).to_numpy()
    floors = pd.to_numeric(df.get("floor_count", pd.Series(np.nan, index=df.index)), errors="coerce").fillna(-1.0).to_numpy()
    beds = pd.to_numeric(df.get("bedroom_count", pd.Series(np.nan, index=df.index)), errors="coerce").fillna(-1.0).to_numpy()

    # Stage 1: Exact hash on masked text + location + rounded physical specs (price excluded)
    exact_sig_to_rep: dict[str, int] = {}
    row_to_rep: list[int] = [0] * len(df)
    blocks: dict[tuple[str, str, str, int], list[int]] = {}

    for i in range(len(df)):
        area_round = int(round(areas[i])) if areas[i] >= 0 else -1
        floor_round = int(round(floors[i])) if floors[i] >= 0 else -1
        bed_round = int(round(beds[i])) if beds[i] >= 0 else -1
        sig_raw = f"{provs[i]}|{dists[i]}|{ptypes[i]}|{area_round}|{floor_round}|{bed_round}|{texts[i]}"
        sig_hash = hashlib.blake2b(sig_raw.encode("utf-8"), digest_size=16).hexdigest()
        if sig_hash in exact_sig_to_rep:
            row_to_rep[i] = exact_sig_to_rep[sig_hash]
        else:
            exact_sig_to_rep[sig_hash] = i
            row_to_rep[i] = i
            area_bin = int(round(areas[i] / 5.0) * 5) if areas[i] >= 0 else -1
            blk = (provs[i], dists[i], ptypes[i], area_bin)
            blocks.setdefault(blk, []).append(i)

    # Stage 2: MinHash LSH within each (province, district, property_type, area_bin) block
    uf = _UnionFind(len(df))
    for i, rep in enumerate(row_to_rep):
        uf.union(i, rep)

    for rep_indices in blocks.values():
        if len(rep_indices) <= 1:
            continue
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        mh_map: dict[int, MinHash] = {}
        for idx in rep_indices:
            sh = _shingles(texts[idx], ngram_size=ngram_size)
            mh = _compute_minhash(sh, num_perm=num_perm)
            mh_map[idx] = mh
            matches = lsh.query(mh)
            for match_key in matches:
                uf.union(idx, int(match_key))
            lsh.insert(str(idx), mh)

    group_ids = [f"dup_{uf.find(i):08d}" for i in range(len(df))]
    return pd.Series(group_ids, index=df.index, dtype="object")
