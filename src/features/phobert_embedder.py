"""Vietnamese word segmentation, PhoBERT embedding extraction, and row_id-keyed caching (TV3)."""

from __future__ import annotations

import json
from pathlib import Path
import re
import time
from typing import Any, Sequence
import warnings

import numpy as np
import pandas as pd


_MASK_SEG_RE = re.compile(r"MASK\s*_?\s*GIA", re.IGNORECASE)


def format_phobert_input(
    property_type: Any,
    province: Any,
    district: Any,
    name_masked: Any,
    description_masked: Any,
) -> str:
    """Format listing fields in priority order so truncation at 256 tokens preserves core signal."""
    pt = str(property_type or "").strip()
    pr = str(province or "").strip()
    di = str(district or "").strip()
    nm = str(name_masked or "").strip()
    ds = str(description_masked or "").strip()
    loc = ", ".join([x for x in (di, pr) if x and x != "__MISSING__"])
    parts = [p for p in [f"{pt}." if pt else "", f"Khu vực: {loc}." if loc else "", f"{nm}." if nm else "", ds] if p]
    return " ".join(parts).strip()


def segment_vietnamese_texts(
    texts: Sequence[str],
    mask_token: str = "MASK_GIA",
) -> tuple[list[str], float]:
    """Word-segment Vietnamese texts with pyvi and map mask token to `<mask>` for PhoBERT."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        from pyvi import ViTokenizer

    t0 = time.perf_counter()
    segmented: list[str] = []
    pat = re.compile(re.escape(mask_token).replace("_", r"\s*_?\s*"), re.IGNORECASE) if mask_token else _MASK_SEG_RE
    for txt in texts:
        s = str(txt or "").strip()
        if not s:
            segmented.append("")
            continue
        seg = ViTokenizer.tokenize(s)
        seg = pat.sub(" <mask> ", seg)
        seg = re.sub(r"\s+", " ", seg).strip()
        segmented.append(seg)
    elapsed = time.perf_counter() - t0
    return segmented, elapsed


def save_embedding_cache(
    cache_dir: Path,
    row_ids: Sequence[str],
    embeddings: np.ndarray,
    text_version: str,
    model_sha: str,
    stats: dict[str, Any] | None = None,
) -> None:
    """Persist fp16 embeddings and an explicit `row_id -> offset` index."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(embeddings, dtype=np.float16)
    if len(row_ids) != len(arr):
        raise ValueError(f"row_ids length ({len(row_ids)}) != embeddings length ({len(arr)})")

    np.save(cache_dir / "phobert_cache.fp16.npy", arr)
    idx_df = pd.DataFrame({"row_id": [str(r) for r in row_ids], "offset": np.arange(len(row_ids), dtype=np.int64)})
    idx_df.to_parquet(cache_dir / "phobert_index.parquet", index=False)

    meta = {
        "n_rows": len(row_ids),
        "dim": int(arr.shape[1]) if arr.ndim == 2 else 0,
        "text_version": text_version,
        "model_sha": model_sha,
        "stats": stats or {},
    }
    (cache_dir / "cache_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def load_embeddings_for_row_ids(
    row_ids: Sequence[str],
    cache_dir: Path,
    expected_text_version: str | None = None,
) -> np.ndarray:
    """Load embeddings strictly aligned to `row_ids` via `phobert_index.parquet`.

    With `expected_text_version`, refuse a cache built from another masking/text version (stale embeddings).
    """
    cache_dir = Path(cache_dir)
    if expected_text_version is not None:
        meta_path = cache_dir / "cache_meta.json"
        cached = json.loads(meta_path.read_text(encoding="utf-8")).get("text_version") if meta_path.exists() else None
        if cached != expected_text_version:
            raise ValueError(
                f"PhoBERT cache text_version={cached!r} != expected {expected_text_version!r}; "
                "re-run extract_embeddings after changing masking/text rules."
            )
    idx_df = pd.read_parquet(cache_dir / "phobert_index.parquet")
    id_to_off = dict(zip(idx_df["row_id"].astype(str), idx_df["offset"].astype(int)))

    offsets: list[int] = []
    missing: list[str] = []
    for rid in row_ids:
        key = str(rid)
        off = id_to_off.get(key)
        if off is None:
            missing.append(key)
        else:
            offsets.append(off)
    if missing:
        raise KeyError(f"{len(missing)} row_ids missing from PhoBERT cache, e.g. {missing[:5]}")

    arr = np.load(cache_dir / "phobert_cache.fp16.npy", mmap_mode="r")
    return np.asarray(arr[offsets], dtype=np.float32)


class PhoBERTEmbedder:
    """Frozen PhoBERT masked-mean-pooling embedder with explicit 256-token truncation."""

    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        revision: str = "01daacda68afe13d83023d16ec647239e344a1e6",
        max_length: int = 256,
        batch_size: int = 16,
        fp16: bool = True,
        normalize_l2: bool = True,
        chunk_pooling: bool = False,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.fp16 = bool(fp16)
        self.normalize_l2 = bool(normalize_l2)
        self.chunk_pooling = bool(chunk_pooling)
        self.device_pref = device
        self._tokenizer = None
        self._model = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self.device_pref == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(self.device_pref)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision)
        self._model = AutoModel.from_pretrained(self.model_name, revision=self.revision)
        self._model.eval()
        self._model.to(self._device)

    def _encode_batch_ids(self, seg_batch: list[str]) -> tuple[np.ndarray, int]:
        import torch

        assert self._tokenizer is not None and self._model is not None and self._device is not None
        enc = self._tokenizer(
            seg_batch,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        attn = enc["attention_mask"]
        # PhoBERT adds <s> and </s>; hitting max_length implies likely truncation
        n_truncated = int((attn.sum(dim=1) >= self.max_length).sum().item())

        input_ids = enc["input_ids"].to(self._device)
        attention_mask = attn.to(self._device)

        use_amp = self.fp16 and self._device.type == "cuda"
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                out = self._model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = out.last_hidden_state
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
                if self.normalize_l2:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.detach().cpu().numpy().astype(np.float16), n_truncated

    def embed_texts(
        self,
        texts: Sequence[str],
        mask_token: str = "MASK_GIA",
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Segment and embed texts, returning `(embeddings_fp16, stats_dict)`."""
        import torch

        seg_texts, seg_time = segment_vietnamese_texts(texts, mask_token=mask_token)
        self._ensure_loaded()
        assert self._device is not None

        if self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self._device)

        t0 = time.perf_counter()
        chunks: list[np.ndarray] = []
        truncated_total = 0
        for start in range(0, len(seg_texts), self.batch_size):
            batch = seg_texts[start : start + self.batch_size]
            if not self.chunk_pooling:
                emb_b, n_tr = self._encode_batch_ids(batch)
                chunks.append(emb_b)
                truncated_total += n_tr
            else:
                # Head + tail window averaging for long descriptions
                emb_head, n_tr = self._encode_batch_ids(batch)
                tail_batch = [" ".join(s.split()[-180:]) if len(s.split()) > 180 else s for s in batch]
                emb_tail, _ = self._encode_batch_ids(tail_batch)
                comb = ((emb_head.astype(np.float32) + emb_tail.astype(np.float32)) * 0.5)
                if self.normalize_l2:
                    norms = np.linalg.norm(comb, axis=1, keepdims=True)
                    comb = comb / np.clip(norms, 1e-6, None)
                chunks.append(comb.astype(np.float16))
                truncated_total += n_tr

        embed_time = time.perf_counter() - t0
        embs = np.vstack(chunks) if chunks else np.zeros((0, 768), dtype=np.float16)
        peak_vram_mb = (
            float(torch.cuda.max_memory_allocated(self._device) / (1024 * 1024))
            if self._device.type == "cuda"
            else 0.0
        )
        n = max(1, len(seg_texts))
        stats = {
            "n_texts": len(seg_texts),
            "device": str(self._device),
            "fp16": self.fp16,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "chunk_pooling": self.chunk_pooling,
            "segmentation_sec": seg_time,
            "segmentation_sec_per_1k": seg_time * 1000.0 / n,
            "embed_sec": embed_time,
            "embed_texts_per_sec": len(seg_texts) / max(1e-6, embed_time),
            "truncated_count": truncated_total,
            "truncated_ratio": truncated_total / n,
            "peak_vram_mb": peak_vram_mb,
        }
        return embs, stats

    def extract_and_cache(
        self,
        df: pd.DataFrame,
        cache_dir: Path,
        text_version: str = "v1.0.0",
        mask_token: str = "MASK_GIA",
    ) -> dict[str, Any]:
        """Format listing rows, compute embeddings, and save `row_id`-indexed cache."""
        formatted = [
            format_phobert_input(pt, pr, di, nm, ds)
            for pt, pr, di, nm, ds in zip(
                df["property_type_name"],
                df["province_norm"],
                df["district_norm"],
                df["name_masked"],
                df["description_masked"],
            )
        ]
        embs, stats = self.embed_texts(formatted, mask_token=mask_token)
        save_embedding_cache(
            cache_dir=cache_dir,
            row_ids=df["row_id"].astype(str).tolist(),
            embeddings=embs,
            text_version=text_version,
            model_sha=self.revision,
            stats=stats,
        )
        return stats
