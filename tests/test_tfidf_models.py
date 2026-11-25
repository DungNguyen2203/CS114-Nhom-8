import numpy as np
import pandas as pd

from src.features.tabular_features import TabularFeatureExtractor
from src.features.tfidf_features import TfidfSvdExtractor
from src.models.m2_tfidf_ridge import M2TfidfRidgeModel
from src.models.m4_tfidf_catboost import M4TfidfCatBoostModel
from tests.test_tabular_and_baselines import _make_tabular_splits


def test_tfidf_svd_and_m2_m4_fit_predict():
    train_df, val_df = _make_tabular_splits(n_train=120, n_val=30)
    ext = TfidfSvdExtractor(max_features=500, min_df=2, ngram_range=(1, 2), svd_components=8, seed=42)
    X_tr_sp, X_tr_svd = ext.fit_transform(train_df["text_masked"].tolist())
    X_va_sp, X_va_svd = ext.transform(val_df["text_masked"].tolist())
    assert X_tr_svd.shape == (len(train_df), 8)
    assert X_va_svd.shape == (len(val_df), 8)

    m2 = M2TfidfRidgeModel(alphas=[1.0, 10.0])
    best_alpha, rows = m2.tune_and_fit(X_tr_sp, train_df["y_true"].to_numpy(), X_va_sp, val_df["y_true"].to_numpy())
    assert best_alpha in (1.0, 10.0)
    assert m2.predict(X_va_sp).shape == (len(val_df),)

    tab = TabularFeatureExtractor(min_freq=5)
    X_tr_tab, cat_cols = tab.fit_transform(train_df)
    X_va_tab, _ = tab.transform(val_df)

    m4 = M4TfidfCatBoostModel(iterations=50, early_stopping_rounds=15, depth=4, learning_rate=0.1, loss_function="RMSE")
    m4.fit(X_tr_tab, X_tr_svd, train_df["y_true"].to_numpy(), X_va_tab, X_va_svd, val_df["y_true"].to_numpy(), cat_cols, seed=42)
    preds = m4.predict(X_va_tab, X_va_svd)
    assert preds.shape == (len(val_df),)
    assert np.isfinite(preds).all()
