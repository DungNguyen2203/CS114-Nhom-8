"""Versioned location, direction, and post-07/2025 administrative drift normalizer."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import pandas as pd
import yaml

from src.data.text_normalizer import normalize_text


_NUMERIC_DISTRICT_RE = re.compile(r"^(?:quận|q\.?|huyện|h\.?)?\s*0*(\d{1,2})$", re.IGNORECASE)
_DISTRICT_PREFIX_RE = re.compile(r"^(?:quận|q\.|huyện|h\.|thị\s*xã|tx\.|thành\s*phố|tp\.?)\s+", re.IGNORECASE)
_NUMERIC_WARD_RE = re.compile(r"^(?:phường|p\.?|xã|x\.?)?\s*0*(\d{1,2})$", re.IGNORECASE)
_WARD_PREFIX_RE = re.compile(r"^(?:phường|p\.|xã|x\.|thị\s*trấn|tt\.?)\s+", re.IGNORECASE)
_MULTI_DIR_SEP_RE = re.compile(r"[,;/]|(?:\b(?:đông|tây|nam|bắc)\b.*?\b(?:đông|tây|nam|bắc)\b)", re.IGNORECASE)


class LocationNormalizer:
    """Normalize provinces, districts, wards, streets, projects, and compass directions."""

    def __init__(self, alias_path: str | Path) -> None:
        self.alias_path = Path(alias_path)
        with self.alias_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        self.version: str = str(cfg.get("version", "v1.0.0"))
        self.province_aliases: dict[str, str] = {
            normalize_text(k).lower(): v for k, v in (cfg.get("province_aliases") or {}).items()
        }
        self.merged_cues: dict[str, list[str]] = {
            k: [normalize_text(c).lower() for c in cues]
            for k, cues in (cfg.get("post_2025_07_merged_cues") or {}).items()
        }
        self.direction_aliases: dict[str, str] = {
            normalize_text(k).lower(): v for k, v in (cfg.get("direction_aliases") or {}).items()
        }

    def normalize_province(self, val: Any) -> str:
        s = normalize_text(val)
        if not s:
            return "__MISSING__"
        return self.province_aliases.get(s.lower(), s)

    def normalize_district(self, val: Any) -> str:
        s = normalize_text(val)
        if not s:
            return "__MISSING__"
        m = _NUMERIC_DISTRICT_RE.match(s)
        if m:
            return f"Quận {int(m.group(1))}"
        cleaned = _DISTRICT_PREFIX_RE.sub("", s).strip()
        return cleaned or s

    def normalize_ward(self, val: Any) -> str:
        s = normalize_text(val)
        if not s:
            return "__MISSING__"
        m = _NUMERIC_WARD_RE.match(s)
        if m:
            return f"Phường {int(m.group(1))}"
        cleaned = _WARD_PREFIX_RE.sub("", s).strip()
        return cleaned or s

    def normalize_free_name(self, val: Any) -> str:
        s = normalize_text(val)
        return s if s else "__MISSING__"

    def normalize_direction(self, val: Any) -> str:
        s = normalize_text(val)
        if not s:
            return "Thiếu"
        low = re.sub(r"\s*-\s*", " ", s.lower())
        low = re.sub(r"\s+", " ", low).strip()
        if low in self.direction_aliases:
            return self.direction_aliases[low]
        if "," in s or ";" in s or "/" in s or "-" in s:
            return "Đa hướng"
        # Check if multiple cardinal words remain unmatched
        tokens = re.findall(r"(?:đông|tây|nam|bắc)", low)
        if len(tokens) > 2:
            return "Đa hướng"
        return s

    def normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["province_norm"] = [self.normalize_province(v) for v in out.get("province_name", pd.Series([None] * len(out)))]
        out["district_norm"] = [self.normalize_district(v) for v in out.get("district_name", pd.Series([None] * len(out)))]
        out["ward_norm"] = [self.normalize_ward(v) for v in out.get("ward_name", pd.Series([None] * len(out)))]
        out["street_norm"] = [self.normalize_free_name(v) for v in out.get("street_name", pd.Series([None] * len(out)))]
        out["project_norm"] = [self.normalize_free_name(v) for v in out.get("project_name", pd.Series([None] * len(out)))]
        out["district_key"] = out["province_norm"].astype(str) + "|" + out["district_norm"].astype(str)

        out["house_direction_norm"] = [
            self.normalize_direction(v) for v in out.get("house_direction", pd.Series([None] * len(out)))
        ]
        out["balcony_direction_norm"] = [
            self.normalize_direction(v) for v in out.get("balcony_direction", pd.Series([None] * len(out)))
        ]

        desc_series = out.get("description", pd.Series([""] * len(out))).fillna("").astype(str).str.lower()
        drift_flags: list[bool] = []
        for prov, desc in zip(out["province_norm"], desc_series):
            cues = self.merged_cues.get(prov)
            if not cues:
                drift_flags.append(False)
                continue
            desc_norm = re.sub(r"[.\s]+", " ", desc)
            drift_flags.append(any((c in desc) or (re.sub(r"[.\s]+", " ", c) in desc_norm) for c in cues))
        out["admin_drift_flag"] = drift_flags
        return out
