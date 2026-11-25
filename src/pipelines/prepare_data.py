"""TV1 streaming data preparation, parsing, masking, dedup, split manifests, audits, and data reports."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import load_config, resolve_project_paths, set_global_seed
from src.data.dedup import assign_duplicate_groups
from src.data.download import download_dataset_shards
from src.data.location_normalizer import LocationNormalizer
from src.data.price_area_parser import parse_listing_text, reconcile_labels
from src.data.splits import apply_stratified_budget, build_scenario_a_split, build_scenario_b_split
from src.data.tabular_cleaner import clean_tabular_domain
from src.data.text_masker import find_leaked_money, mask_money_text
from src.data.text_normalizer import normalize_text, redact_pii


NEEDED_RAW_COLUMNS = [
    "name",
    "description",
    "property_type_name",
    "province_name",
    "district_name",
    "ward_name",
    "street_name",
    "project_name",
    "price",
    "area",
    "floor_count",
    "frontage_width",
    "house_depth",
    "road_width",
    "bedroom_count",
    "bathroom_count",
    "house_direction",
    "balcony_direction",
    "published_at",
]


def _process_batch(
    batch_df: pd.DataFrame,
    shard_idx: int,
    row_offset: int,
    cfg: dict[str, Any],
    loc_norm: LocationNormalizer,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    domain_cfg = cfg["cleaning"]
    mask_token = cfg["project"]["mask_token"]

    n = len(batch_df)
    batch_df = batch_df.reset_index(drop=True)
    batch_df["row_id"] = [f"{shard_idx:02d}_{row_offset + i:07d}" for i in range(n)]
    batch_df["month"] = batch_df["published_at"].astype(str).str.slice(0, 7)

    batch_df = loc_norm.normalize_df(batch_df)
    batch_df = clean_tabular_domain(batch_df, domain_cfg)

    names = [normalize_text(v) for v in batch_df["name"]]
    descs = [normalize_text(v) for v in batch_df["description"]]
    # Drop heavy unnormalized text columns right after normalization
    batch_df = batch_df.drop(columns=[c for c in ["name", "description"] if c in batch_df.columns])

    prices = pd.to_numeric(batch_df["price"], errors="coerce").tolist()
    areas = pd.to_numeric(batch_df["area"], errors="coerce").tolist()

    buckets: list[str] = []
    statuses: list[str] = []
    tx_types: list[str] = []
    text_prices: list[float] = []
    text_unit_prices: list[float] = []
    text_areas: list[float] = []
    area_mismatch_flags: list[bool] = []
    price_ratios: list[float] = []
    is_approx_list: list[bool] = []
    confidences: list[str] = []
    name_masked: list[str] = []
    desc_masked: list[str] = []

    for nm, ds, p, a in zip(names, descs, prices, areas):
        parsed = parse_listing_text(nm, ds, tabular_area=(None if (a is None or a != a) else float(a)))
        rec = reconcile_labels(p, a, parsed, domain_cfg)
        buckets.append(rec.bucket)
        statuses.append(rec.status)
        tx_types.append(parsed.tx_type)
        text_prices.append(float(parsed.sale_price_vnd) if parsed.sale_price_vnd is not None else np.nan)
        text_unit_prices.append(float(parsed.unit_price_per_m2_vnd) if parsed.unit_price_per_m2_vnd is not None else np.nan)
        text_areas.append(float(parsed.area_m2) if parsed.area_m2 is not None else np.nan)
        area_mismatch_flags.append(bool(rec.area_mismatch_flag))
        price_ratios.append(float(rec.price_ratio) if rec.price_ratio is not None else np.nan)
        is_approx_list.append(bool(parsed.is_approx))
        confidences.append(parsed.confidence)
        nm_m = mask_money_text(nm, token=mask_token)
        ds_m = mask_money_text(ds, token=mask_token)
        name_masked.append(nm_m)
        desc_masked.append(ds_m)

    batch_df["name_norm"] = names
    batch_df["description_norm"] = descs
    batch_df["name_masked"] = name_masked
    batch_df["description_masked"] = desc_masked
    batch_df["text_masked"] = [f"{a}. {b}".strip(". ").strip() for a, b in zip(name_masked, desc_masked)]
    batch_df["tx_type"] = tx_types
    batch_df["label_bucket"] = buckets
    batch_df["text_status"] = statuses
    batch_df["text_price_vnd"] = text_prices
    batch_df["text_unit_price_per_m2"] = text_unit_prices
    batch_df["text_area_m2"] = text_areas
    batch_df["area_mismatch_flag"] = area_mismatch_flags
    batch_df["price_ratio"] = price_ratios
    batch_df["is_approx"] = is_approx_list
    batch_df["parser_confidence"] = confidences
    batch_df["rules_version"] = cfg["project"]["rules_version"]

    clean_df = batch_df[batch_df["label_bucket"] == "clean"].copy()
    clean_df["y_true"] = np.log(pd.to_numeric(clean_df["price"], errors="coerce"))
    clean_df["price_per_m2"] = pd.to_numeric(clean_df["price"], errors="coerce") / pd.to_numeric(clean_df["area"], errors="coerce")

    compact_cols = [
        c for c in [
            "row_id", "property_type_name", "province_norm", "district_norm",
            "price", "area", "published_at", "tx_type", "label_bucket", "text_status",
            "text_price_vnd", "text_area_m2", "name_norm", "description_norm",
        ] if c in batch_df.columns
    ]
    noisy_df = batch_df[batch_df["label_bucket"] == "noisy"][compact_cols].copy()
    rental_unknown_df = batch_df[
        batch_df["label_bucket"].isin(["rental", "missing_or_negotiable", "multi_or_unknown"])
    ][compact_cols].copy()

    counts = {
        "total_raw": n,
        "clean": len(clean_df),
        "noisy": len(noisy_df),
        "rental": int((batch_df["label_bucket"] == "rental").sum()),
        "missing_or_negotiable": int((batch_df["label_bucket"] == "missing_or_negotiable").sum()),
        "multi_or_unknown": int((batch_df["label_bucket"] == "multi_or_unknown").sum()),
        "admin_drift_count": int(batch_df["admin_drift_flag"].sum()),
    }
    return clean_df, noisy_df, rental_unknown_df, counts


class _ParquetStreamWriter:
    """Append pandas DataFrames to a single Parquet file with unified schema and minimal RAM."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.writer: pq.ParquetWriter | None = None
        self.schema: pa.Schema | None = None
        self.rows_written: int = 0

    def write(self, df: pd.DataFrame) -> None:
        if len(df) == 0:
            return
        # Ensure consistent object/string columns across batches
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
                df[col] = df[col].fillna("").astype(str)
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self.writer is None:
            self.schema = table.schema
            self.writer = pq.ParquetWriter(self.path, self.schema, compression="snappy")
        else:
            table = table.cast(self.schema)
        self.writer.write_table(table)
        self.rows_written += len(df)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
        elif not self.path.exists():
            pd.DataFrame().to_parquet(self.path, index=False)


def _compute_train_ppm2_diagnostics(
    clean_meta_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    quantiles: list[float],
) -> list[dict[str, Any]]:
    merged = manifest_df.merge(clean_meta_df[["row_id", "price_per_m2"]], on="row_id", how="left")
    train_part = merged[merged["split"] == "train"]
    lo_q, hi_q = float(quantiles[0]), float(quantiles[1])
    bounds = (
        train_part.groupby("property_type_name")["price_per_m2"]
        .quantile([lo_q, hi_q])
        .unstack()
        .rename(columns={lo_q: "q_low", hi_q: "q_high"})
    )
    rows: list[dict[str, Any]] = []
    for ptype, b in bounds.iterrows():
        sub = merged[merged["property_type_name"] == ptype]
        for sp in ["train", "val", "cal", "test"]:
            sp_sub = sub[sub["split"] == sp]
            if len(sp_sub) == 0:
                continue
            out_cnt = int(((sp_sub["price_per_m2"] < b["q_low"]) | (sp_sub["price_per_m2"] > b["q_high"])).sum())
            rows.append(
                {
                    "property_type_name": ptype,
                    "split": sp,
                    "n_rows": len(sp_sub),
                    "train_q_low": float(b["q_low"]),
                    "train_q_high": float(b["q_high"]),
                    "out_of_train_range": out_cnt,
                    "out_of_train_pct": 100.0 * out_cnt / len(sp_sub),
                }
            )
    return rows


def _sample_parquet_rows(path: Path, n_target: int, columns: list[str], seed: int) -> pd.DataFrame:
    if not path.exists() or n_target <= 0:
        return pd.DataFrame()
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    if total == 0:
        return pd.DataFrame()
    avail_cols = [c for c in columns if c in pf.schema.names]
    frac = min(1.0, (n_target * 2.5) / max(1, total))
    rng = np.random.default_rng(seed)
    parts = []
    for batch in pf.iter_batches(batch_size=50_000, columns=avail_cols):
        bdf = batch.to_pandas()
        if frac < 1.0:
            keep = rng.random(len(bdf)) < frac
            bdf = bdf[keep]
        if len(bdf) > 0:
            parts.append(bdf)
    if not parts:
        # fallback first batch
        first = next(pf.iter_batches(batch_size=min(n_target, total), columns=avail_cols)).to_pandas()
        return first.head(n_target)
    df = pd.concat(parts, axis=0, ignore_index=True)
    if len(df) > n_target:
        df = df.sample(n=n_target, random_state=seed).reset_index(drop=True)
    return df


def _export_audits_from_disk(
    clean_path: Path,
    noisy_path: Path,
    rental_path: Path,
    audits_dir: Path,
    label_n: int,
    mask_n: int,
    seed: int,
) -> dict[str, Any]:
    audits_dir.mkdir(parents=True, exist_ok=True)
    lbl_cols = [
        "row_id", "property_type_name", "province_norm", "district_norm",
        "price", "area", "tx_type", "label_bucket", "text_status",
        "text_price_vnd", "text_area_m2", "name_norm", "description_norm",
    ]
    pool_parts = []
    for name, p in [("clean", clean_path), ("noisy", noisy_path), ("rental_unknown", rental_path)]:
        sub = _sample_parquet_rows(p, n_target=max(label_n, 200), columns=lbl_cols, seed=seed)
        if len(sub) > 0:
            sub["source_pool"] = name
            pool_parts.append(sub)

    all_pool = pd.concat(pool_parts, axis=0, ignore_index=True)
    n_lbl = min(label_n, len(all_pool))
    sample_lbl = (
        all_pool.groupby(["property_type_name", "label_bucket"], group_keys=False)
        .apply(lambda g: g.sample(n=max(1, int(round(len(g) / len(all_pool) * n_lbl))), random_state=seed))
        .head(n_lbl)
        .reset_index(drop=True)
    )
    if len(sample_lbl) < n_lbl:
        rem = all_pool[~all_pool["row_id"].isin(sample_lbl["row_id"])]
        sample_lbl = pd.concat(
            [sample_lbl, rem.sample(n=min(len(rem), n_lbl - len(sample_lbl)), random_state=seed)],
            axis=0,
            ignore_index=True,
        )

    sample_lbl["name_redacted"] = [redact_pii(v) for v in sample_lbl["name_norm"]]
    sample_lbl["description_redacted"] = [redact_pii(v)[:500] for v in sample_lbl["description_norm"]]
    sample_lbl = sample_lbl.drop(columns=["name_norm", "description_norm"])
    sample_lbl["human_tx_type_ok"] = ""
    sample_lbl["human_price_match_ok"] = ""
    sample_lbl["human_notes"] = ""
    sample_lbl.to_csv(audits_dir / "audit_labels_300.csv", index=False, encoding="utf-8-sig")

    sample_msk = _sample_parquet_rows(
        clean_path,
        n_target=mask_n,
        columns=["row_id", "property_type_name", "name_masked", "description_masked", "text_masked"],
        seed=seed,
    )
    leaks = [find_leaked_money(t) for t in sample_msk["text_masked"]] if len(sample_msk) > 0 else []
    mask_audit_df = pd.DataFrame(
        {
            "row_id": sample_msk["row_id"] if len(sample_msk) else [],
            "property_type_name": sample_msk["property_type_name"] if len(sample_msk) else [],
            "name_masked_redacted": [redact_pii(t) for t in sample_msk["name_masked"]] if len(sample_msk) else [],
            "description_masked_redacted": [redact_pii(t)[:500] for t in sample_msk["description_masked"]] if len(sample_msk) else [],
            "leaked_tokens": ["; ".join(lk) for lk in leaks],
            "leaked_count": [len(lk) for lk in leaks],
            "human_no_leak_ok": "",
        }
    )
    mask_audit_df.to_csv(audits_dir / "audit_masking_100.csv", index=False, encoding="utf-8-sig")

    return {
        "label_audit_rows": len(sample_lbl),
        "mask_audit_rows": len(mask_audit_df),
        "mask_audit_leaks_detected": int((mask_audit_df["leaked_count"] > 0).sum()) if len(mask_audit_df) else 0,
    }


def _write_data_reports(
    reports_dir: Path,
    summary: dict[str, Any],
    clean_meta_df: pd.DataFrame,
    diag_a: list[dict[str, Any]],
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    funnel = summary["funnel"]
    lines = [
        "# Báo cáo Dữ liệu & Kiểm định Nhãn (TV1)",
        "",
        f"- **Dataset:** `{summary['dataset_repo']}` (revision `{summary['dataset_revision']}`)",
        f"- **Rules version:** `{summary['rules_version']}`",
        f"- **Số shard đã xử lý:** {summary['shards_processed']}",
        "",
        "## 1. Bảng Chuyển đổi Phễu lọc Dữ liệu",
        "",
        "| Bước / Nhóm | Số dòng | Tỷ lệ (%) |",
        "|---|---:|---:|",
        f"| Tổng dòng thô | {funnel['total_raw']:,} | 100.00% |",
        f"| Tập sạch dùng huấn luyện/đánh giá (`clean`) | {funnel['clean']:,} | {100*funnel['clean']/max(1,funnel['total_raw']):.2f}% |",
        f"| Tập mâu thuẫn nhãn (`noisy_flagged`) | {funnel['noisy']:,} | {100*funnel['noisy']/max(1,funnel['total_raw']):.2f}% |",
        f"| Tin cho thuê (`rental`) | {funnel['rental']:,} | {100*funnel['rental']/max(1,funnel['total_raw']):.2f}% |",
        f"| Thiếu giá / thỏa thuận (`missing_or_negotiable`) | {funnel['missing_or_negotiable']:,} | {100*funnel['missing_or_negotiable']/max(1,funnel['total_raw']):.2f}% |",
        f"| Gộp nhiều BĐS / không rõ (`multi_or_unknown`) | {funnel['multi_or_unknown']:,} | {100*funnel['multi_or_unknown']/max(1,funnel['total_raw']):.2f}% |",
        f"| Tin có dấu hiệu lệch địa giới cũ/mới (07/2025) | {funnel['admin_drift_count']:,} | {100*funnel['admin_drift_count']/max(1,funnel['total_raw']):.2f}% |",
        "",
        "## 2. Phân bố Tập Sạch theo Tháng & Loại hình",
        "",
        "### Theo tháng (`published_at`)",
        "| Tháng | Số mẫu sạch |",
        "|---|---:|",
    ]
    for m, cnt in clean_meta_df["month"].value_counts().sort_index().items():
        lines.append(f"| {m} | {cnt:,} |")

    lines.extend(["", "### Theo loại hình (`property_type_name`)", "| Loại hình | Số mẫu sạch | Tỷ lệ (%) |", "|---|---:|---:|"])
    for ptype, cnt in clean_meta_df["property_type_name"].value_counts().items():
        lines.append(f"| {ptype} | {cnt:,} | {100*cnt/max(1,len(clean_meta_df)):.2f}% |")

    lines.extend(["", "### Top 10 Tỉnh/Thành (tập sạch)", "| Tỉnh/Thành | Số mẫu sạch | Tỷ lệ (%) |", "|---|---:|---:|"])
    for prov, cnt in clean_meta_df["province_norm"].value_counts().head(10).items():
        lines.append(f"| {prov} | {cnt:,} | {100*cnt/max(1,len(clean_meta_df)):.2f}% |")

    lines.extend(
        [
            "",
            "## 3. Khử trùng & Splits",
            f"- Số nhóm trùng duy nhất trên `clean`: **{summary['dedup']['unique_groups']:,}** / {len(clean_meta_df):,} dòng.",
            f"- Nhóm trùng có thay đổi giá giữa các lần đăng lại: **{summary['dedup']['groups_with_price_change']:,}**.",
            f"- Kịch bản A (mẫu cuối sau budget): {summary['splits']['scenario_a_counts']}",
            f"- Kịch bản B (mẫu cuối sau budget): {summary['splits']['scenario_b_counts']}",
            "",
            "## 4. Ngưỡng phân vị giá/m² học từ Train (Kịch bản A) & Tỷ lệ ngoài miền",
            "| Loại hình | Split | N | Train Q0.5% (VND/m²) | Train Q99.5% (VND/m²) | Ngoài miền Train |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in diag_a:
        lines.append(
            f"| {row['property_type_name']} | {row['split']} | {row['n_rows']:,} | "
            f"{row['train_q_low']:,.0f} | {row['train_q_high']:,.0f} | "
            f"{row['out_of_train_range']:,} ({row['out_of_train_pct']:.2f}%) |"
        )

    lines.extend(
        [
            "",
            "## 5. Kiểm tra Tự động Ẩn số tiền & Xuất mẫu Audit",
            f"- `audit_labels_300.csv`: {summary['audits']['label_audit_rows']} dòng (đã che PII).",
            f"- `audit_masking_100.csv`: {summary['audits']['mask_audit_rows']} dòng, số dòng còn sót ký hiệu tiền: **{summary['audits']['mask_audit_leaks_detected']}**.",
        ]
    )
    (reports_dir / "data_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    dict_lines = [
        "# Data Dictionary — Vietnam Real Estate Valuation",
        "",
        "| Cột | Kiểu | Mô tả |",
        "|---|---|---|",
        "| `row_id` | `string` | Định danh duy nhất `SS_RRRRRRR` từ chỉ số shard và số dòng gốc |",
        "| `price` | `float64` | Giá chào bán gốc (VND), chỉ dùng tạo nhãn `y_true = log(price)` |",
        "| `y_true` | `float64` | Biến mục tiêu `log(price)` trên tập sạch |",
        "| `area` | `float64` | Diện tích (m²) |",
        "| `property_type_name` | `string` | 5 loại hình thực tế (`Nhà`, `Đất`, `Căn hộ chung cư`, `Biệt thự/Nhà liền kề`, `Shophouse`) |",
        "| `province_norm`, `district_norm`, `ward_norm` | `string` | Địa giới chuẩn hóa theo `location_aliases_v1.yaml` |",
        "| `district_key` | `string` | Khóa nhóm `province_norm|district_norm` cho Kịch bản B |",
        "| `house_direction_norm`, `balcony_direction_norm` | `string` | Hướng đã chuẩn hóa (`Đông`, `Tây`, ..., `Đa hướng`, `Thiếu`) |",
        "| `*_was_outlier`, `*_is_missing` | `int8` | Cờ ngoại lệ miền và khuyết thiếu dữ liệu bảng |",
        "| `name_masked`, `description_masked`, `text_masked` | `string` | Văn bản đã thay mọi số tiền bằng `MASK_GIA` |",
        "| `label_bucket`, `text_status` | `string` | Kết quả phân loại nhãn và đối chiếu giá/diện tích độc lập |",
        "| `dup_group_id` | `string` | Mã cụm trùng từ text đã ẩn giá + vị trí + thông số vật lý (không dùng `price`) |",
    ]
    (reports_dir / "data_dictionary.md").write_text("\n".join(dict_lines) + "\n", encoding="utf-8")


def run_prepare_data(
    cfg: dict[str, Any],
    paths: dict[str, Path],
    raw_shards: list[Path] | None = None,
) -> dict[str, Any]:
    """Execute TV1 streaming ingestion, parsing, normalization, dedup, splits, and reports."""
    seed = int(cfg["project"]["primary_seed"])
    set_global_seed(seed)

    paths["raw_dir"].mkdir(parents=True, exist_ok=True)
    paths["processed_dir"].mkdir(parents=True, exist_ok=True)
    splits_dir = paths["processed_dir"] / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    audits_dir = paths["outputs_dir"] / "audits"

    if raw_shards is None:
        max_shards = cfg.get("sampling", {}).get("max_shards")
        manifest = download_dataset_shards(
            raw_dir=paths["raw_dir"],
            repo_id=cfg["dataset"]["repo_id"],
            revision=cfg["dataset"]["revision"],
            rules_version=cfg["project"]["rules_version"],
            expected_shards=int(cfg["dataset"]["expected_shards"]),
            max_shards=max_shards,
        )
        shard_paths = [Path(s["path"]) for s in manifest["shards"]]
    else:
        shard_paths = list(raw_shards)

    loc_norm = LocationNormalizer(paths["aliases_path"])

    clean_path = paths["processed_dir"] / "clean.parquet"
    noisy_path = paths["processed_dir"] / "noisy_flagged.parquet"
    rental_path = paths["processed_dir"] / "rental_unknown.parquet"

    clean_writer = _ParquetStreamWriter(clean_path)
    noisy_writer = _ParquetStreamWriter(noisy_path)
    rental_writer = _ParquetStreamWriter(rental_path)

    funnel = {
        "total_raw": 0,
        "clean": 0,
        "noisy": 0,
        "rental": 0,
        "missing_or_negotiable": 0,
        "multi_or_unknown": 0,
        "admin_drift_count": 0,
    }

    for shard_idx, spath in enumerate(shard_paths):
        pf = pq.ParquetFile(spath)
        cols_to_read = [c for c in NEEDED_RAW_COLUMNS if c in pf.schema.names]
        row_offset = 0
        for batch in pf.iter_batches(batch_size=50_000, columns=cols_to_read):
            bdf = batch.to_pandas()
            c_df, n_df, r_df, cnts = _process_batch(bdf, shard_idx, row_offset, cfg, loc_norm)
            row_offset += len(bdf)
            for k, v in cnts.items():
                funnel[k] += v
            clean_writer.write(c_df)
            noisy_writer.write(n_df)
            rental_writer.write(r_df)
            del bdf, c_df, n_df, r_df
        gc.collect()

    clean_writer.close()
    noisy_writer.close()
    rental_writer.close()

    # Pass 2: Read lightweight columns ONLY from clean.parquet for dedup & split creation
    dedup_cols = [
        "row_id",
        "published_at",
        "month",
        "province_norm",
        "district_norm",
        "district_key",
        "property_type_name",
        "area",
        "floor_count",
        "bedroom_count",
        "price",
        "price_per_m2",
        "y_true",
        "text_masked",
    ]
    if clean_writer.rows_written > 0:
        pf_clean = pq.ParquetFile(clean_path)
        meta_parts = []
        for batch in pf_clean.iter_batches(batch_size=100_000, columns=dedup_cols):
            bdf = batch.to_pandas()
            # Keep only leading 300 chars of text_masked in RAM for dedup
            bdf["text_masked"] = bdf["text_masked"].astype(str).str.slice(0, 300)
            meta_parts.append(bdf)
        clean_meta = pd.concat(meta_parts, axis=0, ignore_index=True)
        clean_meta["dup_group_id"] = assign_duplicate_groups(
            clean_meta,
            threshold=float(cfg["dedup"]["minhash_threshold"]),
            num_perm=int(cfg["dedup"]["minhash_num_perm"]),
            ngram_size=int(cfg["dedup"]["ngram_size"]),
        )
        clean_meta = clean_meta.drop(columns=["text_masked"])
    else:
        clean_meta = pd.DataFrame(columns=dedup_cols + ["dup_group_id"])

    grp_price_nunique = clean_meta.groupby("dup_group_id")["price"].nunique() if len(clean_meta) > 0 else pd.Series(dtype=int)
    groups_with_price_change = int((grp_price_nunique > 1).sum())

    # Build Scenario A & Scenario B manifests and apply stratified budgets
    man_a_full = build_scenario_a_split(clean_meta, cfg["splits"]["scenario_a"])
    man_a = apply_stratified_budget(man_a_full, cfg["sampling"], seed=seed)

    man_b_full = build_scenario_b_split(clean_meta, cfg["splits"]["scenario_b"], seed=seed)
    man_b = apply_stratified_budget(man_b_full, cfg["sampling"], seed=seed)

    man_a.to_parquet(splits_dir / "scenario_a_manifest.parquet", index=False)
    man_b.to_parquet(splits_dir / "scenario_b_manifest.parquet", index=False)

    diag_a = _compute_train_ppm2_diagnostics(clean_meta, man_a, cfg["cleaning"]["train_ppm2_quantiles"]) if len(man_a) > 0 else []

    audits_meta = _export_audits_from_disk(
        clean_path=clean_path,
        noisy_path=noisy_path,
        rental_path=rental_path,
        audits_dir=audits_dir,
        label_n=int(cfg["audits"]["label_audit_size"]),
        mask_n=int(cfg["audits"]["mask_audit_size"]),
        seed=seed,
    )

    summary = {
        "dataset_repo": cfg["dataset"]["repo_id"],
        "dataset_revision": cfg["dataset"]["revision"],
        "rules_version": cfg["project"]["rules_version"],
        "shards_processed": len(shard_paths),
        "funnel": funnel,
        "dedup": {
            "unique_groups": int(clean_meta["dup_group_id"].nunique()) if len(clean_meta) > 0 else 0,
            "groups_with_price_change": groups_with_price_change,
        },
        "splits": {
            "scenario_a_full_counts": man_a_full["split"].value_counts().to_dict() if len(man_a_full) else {},
            "scenario_a_counts": man_a["split"].value_counts().to_dict() if len(man_a) else {},
            "scenario_b_full_counts": man_b_full["split"].value_counts().to_dict() if len(man_b_full) else {},
            "scenario_b_counts": man_b["split"].value_counts().to_dict() if len(man_b) else {},
            "train_ppm2_diagnostics_a": diag_a,
        },
        "audits": audits_meta,
    }
    (splits_dir / "split_manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_data_reports(paths["reports_dir"], summary, clean_meta, diag_a)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="TV1 data preparation pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = resolve_project_paths(cfg)
    summary = run_prepare_data(cfg, paths)
    print(json.dumps({"funnel": summary["funnel"], "splits": summary["splits"], "audits": summary["audits"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
