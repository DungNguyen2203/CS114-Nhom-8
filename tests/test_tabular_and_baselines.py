import numpy as np
import pandas as pd
import pytest

from src.models.contract import PREDICTION_COLUMNS, make_prediction_df, validate_prediction_df
from src.features.tabular_features import TabularFeatureExtractor
from src.models.m0_median import M0GroupMedianModel
from src.models.m1_catboost import M1CatBoostModel


def _make_tabular_splits(n_train=120, n_val=30):
    rng = np.random.default_rng(42)
    rows = []
    provs = ["Hà Nội", "Hồ Chí Minh", "Đà Nẵng"]
    dists = {"Hà Nội": ["Cầu Giấy", "Hoàng Mai"], "Hồ Chí Minh": ["Quận 7", "Quận 8"], "Đà Nẵng": ["Hải Châu"]}
    ptypes = ["Nhà", "Căn hộ chung cư", "Đất"]
    for i in range(n_train + n_val):
        split = "train" if i < n_train else "val"
        prov = provs[i % len(provs)]
        dist = dists[prov][i % len(dists[prov])]
        ptype = ptypes[i % len(ptypes)]
        # Make an unseen ward & street in val to test rare category handling
        ward = f"Phường {i % 4}" if split == "train" else ("Phường Mới Lạ" if i % 3 == 0 else "Phường 0")
        area = float(rng.uniform(35.0, 120.0))
        floors = float(rng.integers(1, 6)) if ptype == "Nhà" else np.nan
        beds = float(rng.integers(1, 5)) if ptype != "Đất" else np.nan
        base_ppm2 = 120e6 if prov == "Hà Nội" else (100e6 if prov == "Hồ Chí Minh" else 50e6)
        noise = float(rng.normal(0.0, 0.08))
        price = area * base_ppm2 * (1.0 + 0.05 * (0 if np.isnan(floors) else floors)) * np.exp(noise)
        rows.append(
            {
                "row_id": f"00_{i:07d}",
                "split": split,
                "property_type_name": ptype,
                "province_norm": prov,
                "district_norm": dist,
                "ward_norm": ward,
                "street_norm": "Đường A" if i % 2 == 0 else "Đường Hiếm",
                "project_norm": "Dự án X" if i % 5 == 0 else "__MISSING__",
                "house_direction_norm": "Đông Nam" if i % 2 == 0 else "Thiếu",
                "balcony_direction_norm": "Nam" if i % 3 == 0 else "Thiếu",
                "area": area,
                "floor_count": floors,
                "frontage_width": 4.5,
                "house_depth": np.nan,
                "road_width": 6.0,
                "bedroom_count": beds,
                "bathroom_count": 2.0,
                "floor_count_is_missing": int(np.isnan(floors)),
                "frontage_width_is_missing": 0,
                "house_depth_is_missing": 1,
                "road_width_is_missing": 0,
                "bedroom_count_is_missing": int(np.isnan(beds)),
                "bathroom_count_is_missing": 0,
                "area_was_outlier": 0,
                "frontage_width_was_outlier": 0,
                "house_depth_was_outlier": 0,
                "road_width_was_outlier": 0,
                "floor_count_was_outlier": 0,
                "bedroom_count_was_outlier": 0,
                "bathroom_count_was_outlier": 0,
                "text_masked": "bán nhà chính chủ sổ hồng hẻm xe hơi MASK_GIA" if i % 2 == 0 else "đất đẹp mặt tiền MASK_GIA",
                "y_true": float(np.log(price)),
                "price": float(price),
            }
        )
    df = pd.DataFrame(rows)
    return df[df["split"] == "train"].reset_index(drop=True), df[df["split"] == "val"].reset_index(drop=True)


def test_prediction_contract_validation():
    df = make_prediction_df(
        row_ids=["00_0000001", "00_0000002"],
        split="val",
        scenario="A",
        model_id="M0",
        seed=42,
        y_true=np.array([22.0, 23.0]),
        y_pred=np.array([22.1, 22.9]),
    )
    assert list(df.columns) == PREDICTION_COLUMNS
    validate_prediction_df(df)
    bad = df.drop(columns=["q_hi"])
    with pytest.raises(ValueError):
        validate_prediction_df(bad)


def test_tabular_extractor_fits_rare_categories_strictly_on_train():
    train_df, val_df = _make_tabular_splits()
    ext = TabularFeatureExtractor(min_freq=10)
    X_tr, cat_cols = ext.fit_transform(train_df)
    X_va, _ = ext.transform(val_df)
    assert "ward_norm" in cat_cols
    assert "Phường Mới Lạ" not in set(X_va["ward_norm"].astype(str))
    assert "__RARE__" in set(X_va["ward_norm"].astype(str))
    assert "log_area" in X_tr.columns
    assert "has_so_hong" in X_tr.columns


def test_m0_group_median_fallback_and_prediction_shape():
    train_df, val_df = _make_tabular_splits()
    m0 = M0GroupMedianModel(min_group_samples=30)
    m0.fit(train_df)
    preds = m0.predict(val_df)
    assert len(preds) == len(val_df)
    assert np.isfinite(preds).all()


def test_m1_catboost_tunes_and_trains_quantiles():
    train_df, val_df = _make_tabular_splits(n_train=140, n_val=40)
    ext = TabularFeatureExtractor(min_freq=5)
    X_tr, cat_cols = ext.fit_transform(train_df)
    X_va, _ = ext.transform(val_df)
    m1 = M1CatBoostModel(
        iterations=60,
        early_stopping_rounds=20,
        loss_candidates=["RMSE"],
        depth_candidates=[4],
        lr_candidates=[0.1],
    )
    best_cfg, tuning_rows = m1.tune_and_fit(X_tr, train_df["y_true"].to_numpy(), X_va, val_df["y_true"].to_numpy(), cat_cols, seed=42)
    assert best_cfg["loss_function"] == "RMSE"
    assert len(tuning_rows) == 1
    y_pred = m1.predict(X_va)
    assert y_pred.shape == (len(val_df),)
    m1.fit_quantiles(X_tr, train_df["y_true"].to_numpy(), X_va, val_df["y_true"].to_numpy(), cat_cols, seed=42)
    q_lo, q_hi = m1.predict_quantiles(X_va)
    assert q_lo.shape == (len(val_df),)
    assert q_hi.shape == (len(val_df),)
    assert float(np.mean(q_hi >= q_lo)) > 0.7
