"""Deployable inference service used by Streamlit demo ensuring train/inference parity."""

from __future__ import annotations

import json
import os
from pathlib import Path, PureWindowsPath
import time
from typing import Any

from catboost import CatBoostRegressor
import joblib
import numpy as np
import pandas as pd

from src.data.location_normalizer import LocationNormalizer
from src.data.download import sha256_file
from src.eval.flagger import classify_g2
from src.data.price_area_parser import parse_listing_text, reconcile_labels
from src.data.tabular_cleaner import clean_tabular_domain
from src.data.text_masker import mask_money_text
from src.data.text_normalizer import normalize_text, redact_pii
from src.features.phobert_embedder import PhoBERTEmbedder, format_phobert_input
from src.models.catboost_utils import concat_dense_to_tabular


SAMPLE_DEMO_LISTINGS: list[dict[str, Any]] = [
    {
        "label": "Nhà riêng Cầu Giấy (Hà Nội) — 45m², 4 tầng",
        "property_type_name": "Nhà",
        "province_name": "Hà Nội",
        "district_name": "Cầu Giấy",
        "ward_name": "Dịch Vọng",
        "street_name": "Xuân Thủy",
        "project_name": "",
        "area": 45.0,
        "floor_count": 4.0,
        "frontage_width": 4.2,
        "house_depth": 10.5,
        "road_width": 4.5,
        "bedroom_count": 4.0,
        "bathroom_count": 4.0,
        "house_direction": "Đông Nam",
        "balcony_direction": "",
        "name": "Bán nhà phân lô Dịch Vọng Cầu Giấy 45m2 4 tầng ô tô tránh",
        "description": "Nhà dân xây chắc chắn, sổ đỏ chính chủ, ngõ thông ô tô vào nhà. Giá chào 8.5 tỷ thương lượng.",
        "asking_price_vnd": 8_500_000_000.0,
    },
    {
        "label": "Căn hộ Quận 7 (TP.HCM) — 72m², 2PN",
        "property_type_name": "Căn hộ chung cư",
        "province_name": "Hồ Chí Minh",
        "district_name": "7",
        "ward_name": "Tân Phong",
        "street_name": "Phạm Văn Nghị",
        "project_name": "Sky Garden 3",
        "area": 72.0,
        "floor_count": None,
        "frontage_width": None,
        "house_depth": None,
        "road_width": None,
        "bedroom_count": 2.0,
        "bathroom_count": 2.0,
        "house_direction": "Tây",
        "balcony_direction": "Nam",
        "name": "Bán căn hộ Sky Garden 3 Phú Mỹ Hưng 72m2 2PN full nội thất",
        "description": "Căn hộ tầng trung thoáng mát, sổ hồng sẵn, đang có HDT 18tr/th. Giá bán 5.4 tỷ.",
        "asking_price_vnd": 5_400_000_000.0,
    },
    {
        "label": "Đất nền Ngũ Hành Sơn (Đà Nẵng) — 100m²",
        "property_type_name": "Đất",
        "province_name": "Đà Nẵng",
        "district_name": "Ngũ Hành Sơn",
        "ward_name": "Hòa Hải",
        "street_name": "Trần Đại Nghĩa",
        "project_name": "",
        "area": 100.0,
        "floor_count": None,
        "frontage_width": 5.0,
        "house_depth": 20.0,
        "road_width": 7.5,
        "bedroom_count": None,
        "bathroom_count": None,
        "house_direction": "Đông",
        "balcony_direction": "",
        "name": "Bán lô đất sạch đẹp 100m2 đường 7m5 Ngũ Hành Sơn",
        "description": "Lô đất vuông vức 5x20m, sổ đỏ riêng, khu dân cư hiện hữu, giá 3.2 tỷ.",
        "asking_price_vnd": 3_200_000_000.0,
    },
    {
        "label": "Tin nghi lỗi bậc độ lớn — 2.050 tỷ vs 20.5 tỷ",
        "property_type_name": "Nhà",
        "province_name": "Hà Nội",
        "district_name": "Hoàng Mai",
        "ward_name": "Định Công",
        "street_name": "Định Công Thượng",
        "project_name": "",
        "area": 50.0,
        "floor_count": 4.0,
        "frontage_width": 4.0,
        "house_depth": 12.5,
        "road_width": 3.5,
        "bedroom_count": 3.0,
        "bathroom_count": 3.0,
        "house_direction": "Nam",
        "balcony_direction": "",
        "name": "Bán nhà Định Công 50m2 4 tầng giá 2.050 tỷ",
        "description": "Nhà đẹp ở ngay ngõ thoáng gần đường lớn. Giá 2 tỷ 050 triệu.",
        "asking_price_vnd": 20_500_000_000.0,
    },
    {
        "label": "Biệt thự / Liền kề Hà Đông (Hà Nội) — 90m², 4 tầng",
        "property_type_name": "Biệt thự/Nhà liền kề",
        "province_name": "Hà Nội",
        "district_name": "Hà Đông",
        "ward_name": "Phú Lương",
        "street_name": "Văn Khê",
        "project_name": "",
        "area": 90.0,
        "floor_count": 4.0,
        "frontage_width": 7.2,
        "house_depth": 12.5,
        "road_width": 12.0,
        "bedroom_count": 4.0,
        "bathroom_count": 4.0,
        "house_direction": "Đông Bắc",
        "balcony_direction": "",
        "name": "Liền kề khu đô thị Phú Lương 90m2 mặt tiền 7.2m kinh doanh",
        "description": "Đường trước nhà 12m vỉa hè rộng, tiện ở và làm văn phòng, giá chào 19 tỷ.",
        "asking_price_vnd": 19_000_000_000.0,
    },
    {
        "label": "Shophouse Nha Trang (Khánh Hòa) — 110m², 4 tầng",
        "property_type_name": "Shophouse",
        "province_name": "Khánh Hòa",
        "district_name": "Nha Trang",
        "ward_name": "Lộc Thọ",
        "street_name": "Trần Phú",
        "project_name": "",
        "area": 110.0,
        "floor_count": 4.0,
        "frontage_width": 5.5,
        "house_depth": 20.0,
        "road_width": 15.0,
        "bedroom_count": 4.0,
        "bathroom_count": 4.0,
        "house_direction": "Đông",
        "balcony_direction": "",
        "name": "Bán shophouse trục thương mại Nha Trang 110m2",
        "description": "Khai thác kinh doanh ngay, sổ hồng lâu dài, giá 12.5 tỷ.",
        "asking_price_vnd": 12_500_000_000.0,
    },
]


# Fallback for bundles saved before `reconcile_cfg` existed; keep equal to configs/base.yaml `cleaning`.
_DEFAULT_RECONCILE_CFG = {
    "min_area_m2": 5.0,
    "max_area_m2": 50000.0,
    "min_price_vnd": 50_000_000.0,
    "max_price_vnd": 2_000_000_000_000.0,
    "min_price_per_m2_vnd": 500_000.0,
    "max_price_per_m2_vnd": 3_000_000_000.0,
    "price_mismatch_ratio_strict": 1.35,
    "price_mismatch_ratio_approx": 1.60,
    "order_of_magnitude_ratio": 5.0,
    "area_mismatch_ratio": 1.30,
}


class ValuationInferenceService:
    """Single-listing inference matching train-time preprocessing, masking, and C2 CQR intervals."""

    def __init__(self, bundle_dir: Path) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.meta: dict[str, Any] = json.loads((self.bundle_dir / "bundle_meta.json").read_text(encoding="utf-8"))
        for name, digest in self.meta.get("files", {}).items():
            if sha256_file(self.bundle_dir / name) != digest:
                raise ValueError(f"Bundle file differs from training manifest: {name}")
        self.tab_ext = joblib.load(self.bundle_dir / "tabular_extractor.joblib")
        self.point_model = CatBoostRegressor().load_model(str(self.bundle_dir / "point_model.cbm"))
        self.q_lo_model = CatBoostRegressor().load_model(str(self.bundle_dir / "q_lo_model.cbm"))
        self.q_hi_model = CatBoostRegressor().load_model(str(self.bundle_dir / "q_hi_model.cbm"))

        if "aliases_file" in self.meta:
            aliases_path = self.bundle_dir / self.meta["aliases_file"]
            if sha256_file(aliases_path) != self.meta["aliases_sha256"]:
                raise ValueError("Location alias file does not match the trained bundle")
        else:
            aliases_path = Path(self.meta["aliases_path"])
            if not aliases_path.exists():
                aliases_path = Path(__file__).resolve().parent.parent / "configs" / "location_aliases_v1.yaml"
        self.loc_norm = LocationNormalizer(aliases_path)
        self.train_comps = pd.read_parquet(self.bundle_dir / "train_comps.parquet")

        dist_counts_p = self.bundle_dir / "district_train_counts.json"
        self.district_counts: dict[str, int] = json.loads(dist_counts_p.read_text(encoding="utf-8")) if dist_counts_p.exists() else {}

        self.emb_scaler = joblib.load(self.bundle_dir / "emb_scaler.joblib") if self.meta.get("uses_phobert") else None
        self.emb_pca = joblib.load(self.bundle_dir / "emb_pca.joblib") if self.meta.get("uses_phobert") else None
        self.tfidf_svd = joblib.load(self.bundle_dir / "tfidf_svd.joblib") if self.meta.get("uses_svd") else None
        self._phobert: PhoBERTEmbedder | None = None

    @classmethod
    def load(cls, bundle_dir: str | Path) -> "ValuationInferenceService":
        return cls(Path(bundle_dir))

    def _get_phobert(self) -> PhoBERTEmbedder:
        if self._phobert is None:
            if "HF_HOME" not in os.environ:
                cfg_hf = str(self.meta.get("hf_home") or "")
                # Use the saved path only if it fits this OS: a drive letter on Windows, none on POSIX
                # (a Windows "E:/hf_cache" on Linux would become a junk relative directory).
                if cfg_hf and (os.name == "nt") == bool(PureWindowsPath(cfg_hf).drive):
                    os.environ["HF_HOME"] = cfg_hf
                elif os.name == "nt" and Path("E:/hf_cache").exists():
                    os.environ["HF_HOME"] = "E:/hf_cache"
            ph_cfg = self.meta.get("phobert", {})
            self._phobert = PhoBERTEmbedder(
                model_name=ph_cfg.get("model_name", "vinai/phobert-base"),
                revision=ph_cfg.get("revision", "01daacda68afe13d83023d16ec647239e344a1e6"),
                max_length=int(ph_cfg.get("max_length", 256)),
                batch_size=1,
                fp16=bool(ph_cfg.get("fp16", True)),
                normalize_l2=bool(ph_cfg.get("normalize_l2", True)),
                chunk_pooling=bool(ph_cfg.get("chunk_pooling", False)),
            )
        return self._phobert

    def _find_similar_comps(self, row_norm: pd.Series, k: int = 5) -> list[dict[str, Any]]:
        sub = self.train_comps[self.train_comps["property_type_name"] == row_norm["property_type_name"]].copy()
        same_dist = sub[
            (sub["province_norm"] == row_norm["province_norm"]) & (sub["district_norm"] == row_norm["district_norm"])
        ]
        if len(same_dist) >= 3:
            pool = same_dist.copy()
        else:
            same_prov = sub[sub["province_norm"] == row_norm["province_norm"]]
            pool = same_prov.copy() if len(same_prov) >= 1 else sub.copy()

        if len(pool) == 0:
            return []
        target_area = float(row_norm["area"]) if pd.notna(row_norm["area"]) else 60.0
        pool["_dist"] = np.abs(np.log(np.clip(pool["area"].to_numpy(dtype=float), 1.0, None)) - np.log(max(1.0, target_area)))
        # Break area-distance ties using available physical specs, including on older bundles.
        pool["_spec_distance"] = 0.0
        for feature in ("bedroom_count", "floor_count"):
            target = row_norm.get(feature)
            if feature in pool and pd.notna(target):
                difference = (pd.to_numeric(pool[feature], errors="coerce") - float(target)).abs()
                pool["_spec_distance"] += difference.fillna(difference.max() + 1).fillna(0)
        top = pool.sort_values(["_dist", "_spec_distance", "row_id"]).head(k)
        results = []
        for _, r in top.iterrows():
            results.append(
                {
                    "row_id": str(r["row_id"]),
                    "property_type_name": str(r["property_type_name"]),
                    "province": str(r["province_norm"]),
                    "district": str(r["district_norm"]),
                    "area_m2": float(r["area"]),
                    "price_vnd": float(r["price"]),
                    "title_redacted": redact_pii(str(r.get("name_masked", ""))),
                }
            )
        return results

    def predict_single(self, listing: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        mask_token = self.meta.get("mask_token", "MASK_GIA")
        name_norm = normalize_text(listing.get("name", ""))
        desc_norm = normalize_text(listing.get("description", ""))
        name_masked = mask_money_text(name_norm, token=mask_token)
        desc_masked = mask_money_text(desc_norm, token=mask_token)
        text_masked = f"{name_masked}. {desc_masked}".strip(". ").strip()

        raw_df = pd.DataFrame(
            [
                {
                    "property_type_name": listing.get("property_type_name", "Nhà"),
                    "province_name": listing.get("province_name", ""),
                    "district_name": listing.get("district_name", ""),
                    "ward_name": listing.get("ward_name", ""),
                    "street_name": listing.get("street_name", ""),
                    "project_name": listing.get("project_name", ""),
                    "area": listing.get("area", 50.0),
                    "floor_count": listing.get("floor_count"),
                    "frontage_width": listing.get("frontage_width"),
                    "house_depth": listing.get("house_depth"),
                    "road_width": listing.get("road_width"),
                    "bedroom_count": listing.get("bedroom_count"),
                    "bathroom_count": listing.get("bathroom_count"),
                    "house_direction": listing.get("house_direction"),
                    "balcony_direction": listing.get("balcony_direction"),
                    "description": desc_norm,
                    "name_masked": name_masked,
                    "description_masked": desc_masked,
                    "text_masked": text_masked,
                }
            ]
        )
        norm_df = self.loc_norm.normalize_df(raw_df)
        clean_df = clean_tabular_domain(norm_df, {"tabular_bounds": self.meta.get("tabular_bounds", {})})
        X_tab, _ = self.tab_ext.transform(clean_df)

        if self.meta.get("uses_phobert") and self.emb_scaler is not None and self.emb_pca is not None:
            fmt = format_phobert_input(
                clean_df.loc[0, "property_type_name"],
                clean_df.loc[0, "province_norm"],
                clean_df.loc[0, "district_norm"],
                name_masked,
                desc_masked,
            )
            emb, _ = self._get_phobert().embed_texts([fmt], mask_token=mask_token)
            z = self.emb_pca.transform(self.emb_scaler.transform(emb.astype(np.float32))).astype(np.float32)
            X_in = concat_dense_to_tabular(X_tab, z, "pca")
        elif self.meta.get("uses_svd") and self.tfidf_svd is not None:
            z = self.tfidf_svd.transform_svd([text_masked])
            X_in = concat_dense_to_tabular(X_tab, z, "svd")
        else:
            X_in = X_tab

        y_pred = float(self.point_model.predict(X_in)[0])
        q_lo_raw = float(self.q_lo_model.predict(X_in)[0])
        q_hi_raw = float(self.q_hi_model.predict(X_in)[0])
        lo_s, hi_s = min(q_lo_raw, q_hi_raw), max(q_lo_raw, q_hi_raw)
        c2_q = max(float(self.meta.get("c2_q", 0.0)), 0.0)
        q_lo_c2 = lo_s - c2_q
        q_hi_c2 = hi_s + c2_q

        pred_vnd = float(np.exp(y_pred))
        q_lo_vnd = float(np.exp(q_lo_c2))
        q_hi_vnd = float(np.exp(q_hi_c2))

        asking = listing.get("asking_price_vnd")
        price_position = None
        parsed = parse_listing_text(name_norm, desc_norm, tabular_area=float(clean_df.loc[0, "area"]))
        text_consistency = "not_checked"
        flag_g2 = "none"
        if asking is not None and float(asking) > 0:
            ask_log = float(np.log(float(asking)))
            if ask_log < q_lo_c2:
                price_position = "below_interval"
            elif ask_log > q_hi_c2:
                price_position = "above_interval"
            else:
                price_position = "in_interval"
            rec = reconcile_labels(
                float(asking),
                float(clean_df.loc[0, "area"]),
                parsed,
                self.meta.get("reconcile_cfg", _DEFAULT_RECONCILE_CFG),  # thresholds saved with the bundle at train time
            )
            text_consistency = rec.status
            flag_g2 = classify_g2(rec.status, rec.area_mismatch_flag)

        d_key = str(clean_df.loc[0, "district_key"])
        n_train_dist = int(self.district_counts.get(d_key, 0))
        low_data_warning = n_train_dist < 30

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "pred_price_vnd": pred_vnd,
            "q_lo_vnd": q_lo_vnd,
            "q_hi_vnd": q_hi_vnd,
            "model_version": self.meta.get("model_version", "unknown"),
            "inference_ms": elapsed_ms,
            "masked_text": text_masked,
            "price_position": price_position,
            "text_consistency": text_consistency,
            "flag_g2": flag_g2,
            "district_train_count": n_train_dist,
            "low_data_warning": low_data_warning,
            "similar_listings": self._find_similar_comps(clean_df.loc[0], k=5),
        }
