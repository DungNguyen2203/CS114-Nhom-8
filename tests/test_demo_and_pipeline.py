import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from src.config import load_config
from app.inference import ValuationInferenceService
from src.features.tabular_features import TabularFeatureExtractor
from src.models.m1_catboost import M1CatBoostModel
from tests.test_tabular_and_baselines import _make_tabular_splits


def test_valuation_inference_service_end_to_end(tmp_path):
    train_df, val_df = _make_tabular_splits(n_train=100, n_val=20)
    train_df["province_norm"] = train_df["province_norm"]
    train_df["district_norm"] = train_df["district_norm"]
    train_df["name_masked"] = train_df["text_masked"]
    train_df["description_masked"] = "Sổ hồng chính chủ ngõ ô tô"

    tab = TabularFeatureExtractor(min_freq=5)
    X_tr, cat_cols = tab.fit_transform(train_df)
    X_va, _ = tab.transform(val_df)

    m1 = M1CatBoostModel(iterations=40, early_stopping_rounds=10, loss_candidates=["RMSE"], depth_candidates=[4], lr_candidates=[0.1])
    m1.tune_and_fit(X_tr, train_df["y_true"].to_numpy(), X_va, val_df["y_true"].to_numpy(), cat_cols, seed=42)
    m1.fit_quantiles(X_tr, train_df["y_true"].to_numpy(), X_va, val_df["y_true"].to_numpy(), cat_cols, seed=42)

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(tab, bundle_dir / "tabular_extractor.joblib")
    m1.point_model.save_model(str(bundle_dir / "point_model.cbm"))
    m1.q_lo_model.save_model(str(bundle_dir / "q_lo_model.cbm"))
    m1.q_hi_model.save_model(str(bundle_dir / "q_hi_model.cbm"))
    meta = {
        "model_id": "M1",
        "model_version": "test_v1",
        "uses_phobert": False,
        "uses_svd": False,
        "c2_q": 0.15,
        "aliases_path": str(Path("configs/location_aliases_v1.yaml").resolve()),
        "tabular_bounds": load_config("configs/local_smoke.yaml")["cleaning"]["tabular_bounds"],
    }
    (bundle_dir / "bundle_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    # Retrieval index strictly from clean train
    comps = train_df[
        [
            "row_id",
            "property_type_name",
            "province_norm",
            "district_norm",
            "ward_norm",
            "area",
            "floor_count",
            "bedroom_count",
            "price",
            "name_masked",
        ]
    ].copy()
    comps.to_parquet(bundle_dir / "train_comps.parquet", index=False)

    svc = ValuationInferenceService.load(bundle_dir)
    res = svc.predict_single(
        {
            "property_type_name": "Nhà",
            "province_name": "Hà Nội",
            "district_name": "Cầu Giấy",
            "ward_name": "Dịch Vọng",
            "area": 55.0,
            "floor_count": 4.0,
            "bedroom_count": 3.0,
            "name": "Bán nhà Cầu Giấy 7.5 tỷ",
            "description": "Nhà đẹp 55m2 LH anh Nam 0912345678 giá 7.5 tỷ",
            "asking_price_vnd": 7_500_000_000.0,
        }
    )
    assert res["pred_price_vnd"] > 0
    assert res["q_lo_vnd"] <= res["q_hi_vnd"]
    assert "7.5" not in res["masked_text"]
    assert "MASK_GIA" in res["masked_text"]
    assert res["price_position"] in {"below_interval", "in_interval", "above_interval"}
    assert len(res["similar_listings"]) >= 1
    assert "0912345678" not in str(res["similar_listings"])


def test_kaggle_notebook_valid_nbformat():
    import nbformat

    nb = nbformat.read("notebooks/kaggle_runner.ipynb", as_version=4)
    assert len(nb.cells) >= 4
    joined = "\n".join(c.source for c in nb.cells)
    assert "prepare_data" in joined
    assert "extract_embeddings" in joined
    assert "train_eval" in joined
    assert "kaggle_full.yaml" in joined
