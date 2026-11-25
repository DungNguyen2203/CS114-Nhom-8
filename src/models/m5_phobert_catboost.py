"""M5 train-fitted PCA(PhoBERT) + tabular CatBoost multimodal model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.models.catboost_utils import concat_dense_to_tabular, fit_quantile_pair, train_catboost_single


class M5PhoBERTCatBoostModel:
    """CatBoost combining tabular features with PCA-reduced PhoBERT embeddings fit strictly on train."""

    def __init__(
        self,
        pca_components: int = 64,
        iterations: int = 2000,
        early_stopping_rounds: int = 100,
        depth: int = 6,
        learning_rate: float = 0.05,
        loss_function: str = "RMSE",
    ) -> None:
        self.pca_components = int(pca_components)
        self.iterations = int(iterations)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.depth = int(depth)
        self.learning_rate = float(learning_rate)
        self.loss_function = str(loss_function)
        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None
        self.point_model = None
        self.q_lo_model = None
        self.q_hi_model = None

    def _fit_transform_pca(self, emb_tr: np.ndarray, seed: int) -> np.ndarray:
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(np.asarray(emb_tr, dtype=np.float32))
        k = min(self.pca_components, scaled.shape[0] - 1, scaled.shape[1])
        self.pca = PCA(n_components=max(1, k), random_state=seed)
        return self.pca.fit_transform(scaled).astype(np.float32)

    def _transform_pca(self, emb: np.ndarray) -> np.ndarray:
        if self.scaler is None or self.pca is None:
            raise RuntimeError("PCA not fitted.")
        scaled = self.scaler.transform(np.asarray(emb, dtype=np.float32))
        return self.pca.transform(scaled).astype(np.float32)

    def fit(
        self,
        X_tr_tab: pd.DataFrame,
        emb_tr: np.ndarray,
        y_tr: np.ndarray,
        X_va_tab: pd.DataFrame,
        emb_va: np.ndarray,
        y_va: np.ndarray,
        cat_cols: list[str],
        seed: int = 42,
    ) -> float:
        z_tr = self._fit_transform_pca(emb_tr, seed=seed)
        z_va = self._transform_pca(emb_va)
        X_tr = concat_dense_to_tabular(X_tr_tab, z_tr, "pca")
        X_va = concat_dense_to_tabular(X_va_tab, z_va, "pca")
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

    def predict(self, X_tab: pd.DataFrame, emb: np.ndarray) -> np.ndarray:
        if self.point_model is None:
            raise RuntimeError("M5 model not fitted.")
        z = self._transform_pca(emb)
        X = concat_dense_to_tabular(X_tab, z, "pca")
        return np.asarray(self.point_model.predict(X), dtype=np.float64)

    def fit_quantiles(
        self,
        X_tr_tab: pd.DataFrame,
        emb_tr: np.ndarray,
        y_tr: np.ndarray,
        X_va_tab: pd.DataFrame,
        emb_va: np.ndarray,
        y_va: np.ndarray,
        cat_cols: list[str],
        quantiles: tuple[float, float] = (0.05, 0.95),
        seed: int = 42,
    ) -> float:
        z_tr = self._transform_pca(emb_tr)
        z_va = self._transform_pca(emb_va)
        X_tr = concat_dense_to_tabular(X_tr_tab, z_tr, "pca")
        X_va = concat_dense_to_tabular(X_va_tab, z_va, "pca")
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

    def predict_quantiles(self, X_tab: pd.DataFrame, emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.q_lo_model is None or self.q_hi_model is None:
            raise RuntimeError("M5 quantile models not fitted.")
        z = self._transform_pca(emb)
        X = concat_dense_to_tabular(X_tab, z, "pca")
        return (
            np.asarray(self.q_lo_model.predict(X), dtype=np.float64),
            np.asarray(self.q_hi_model.predict(X), dtype=np.float64),
        )
