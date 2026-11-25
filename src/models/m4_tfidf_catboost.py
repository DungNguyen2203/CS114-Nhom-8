"""M4 TruncatedSVD(TF-IDF) + tabular CatBoost multimodal baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.catboost_utils import concat_dense_to_tabular, fit_quantile_pair, train_catboost_single


class M4TfidfCatBoostModel:
    """CatBoost combining tabular features with train-fitted TF-IDF TruncatedSVD components."""

    def __init__(
        self,
        iterations: int = 2000,
        early_stopping_rounds: int = 100,
        depth: int = 6,
        learning_rate: float = 0.05,
        loss_function: str = "RMSE",
    ) -> None:
        self.iterations = int(iterations)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.depth = int(depth)
        self.learning_rate = float(learning_rate)
        self.loss_function = str(loss_function)
        self.point_model = None
        self.q_lo_model = None
        self.q_hi_model = None

    def fit(
        self,
        X_tr_tab: pd.DataFrame,
        X_tr_svd: np.ndarray,
        y_tr: np.ndarray,
        X_va_tab: pd.DataFrame,
        X_va_svd: np.ndarray,
        y_va: np.ndarray,
        cat_cols: list[str],
        seed: int = 42,
    ) -> float:
        X_tr = concat_dense_to_tabular(X_tr_tab, X_tr_svd, "svd")
        X_va = concat_dense_to_tabular(X_va_tab, X_va_svd, "svd")
        model, elapsed = train_catboost_single(
            X_tr,
            y_tr,
            X_va,
            y_va,
            cat_cols,
            loss_function=self.loss_function,
            depth=self.depth,
            learning_rate=self.learning_rate,
            iterations=self.iterations,
            early_stopping_rounds=self.early_stopping_rounds,
            seed=seed,
        )
        self.point_model = model
        return elapsed

    def predict(self, X_tab: pd.DataFrame, X_svd: np.ndarray) -> np.ndarray:
        if self.point_model is None:
            raise RuntimeError("M4 model not fitted.")
        X = concat_dense_to_tabular(X_tab, X_svd, "svd")
        return np.asarray(self.point_model.predict(X), dtype=np.float64)

    def fit_quantiles(
        self,
        X_tr_tab: pd.DataFrame,
        X_tr_svd: np.ndarray,
        y_tr: np.ndarray,
        X_va_tab: pd.DataFrame,
        X_va_svd: np.ndarray,
        y_va: np.ndarray,
        cat_cols: list[str],
        quantiles: tuple[float, float] = (0.05, 0.95),
        seed: int = 42,
    ) -> float:
        X_tr = concat_dense_to_tabular(X_tr_tab, X_tr_svd, "svd")
        X_va = concat_dense_to_tabular(X_va_tab, X_va_svd, "svd")
        self.q_lo_model, self.q_hi_model, elapsed = fit_quantile_pair(
            X_tr,
            y_tr,
            X_va,
            y_va,
            cat_cols,
            depth=self.depth,
            learning_rate=self.learning_rate,
            iterations=self.iterations,
            early_stopping_rounds=self.early_stopping_rounds,
            quantiles=quantiles,
            seed=seed,
        )
        return elapsed

    def predict_quantiles(self, X_tab: pd.DataFrame, X_svd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.q_lo_model is None or self.q_hi_model is None:
            raise RuntimeError("M4 quantile models not fitted.")
        X = concat_dense_to_tabular(X_tab, X_svd, "svd")
        return (
            np.asarray(self.q_lo_model.predict(X), dtype=np.float64),
            np.asarray(self.q_hi_model.predict(X), dtype=np.float64),
        )
