"""M1 tabular-only CatBoost point and quantile regression model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.catboost_utils import fit_quantile_pair, train_catboost_single, tune_catboost_grid


class M1CatBoostModel:
    """CatBoost model on clean tabular features (plus missing indicators and keyword flags)."""

    def __init__(
        self,
        iterations: int = 2000,
        early_stopping_rounds: int = 100,
        loss_candidates: list[str] | None = None,
        depth_candidates: list[int] | None = None,
        lr_candidates: list[float] | None = None,
    ) -> None:
        self.iterations = int(iterations)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.loss_candidates = loss_candidates or ["RMSE", "MAE"]
        self.depth_candidates = depth_candidates or [6, 8]
        self.lr_candidates = lr_candidates or [0.05, 0.1]
        self.point_model = None
        self.best_cfg: dict[str, Any] = {}
        self.q_lo_model = None
        self.q_hi_model = None

    def tune_and_fit(
        self,
        X_tr: pd.DataFrame,
        y_tr: np.ndarray,
        X_va: pd.DataFrame,
        y_va: np.ndarray,
        cat_cols: list[str],
        seed: int = 42,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        model, best_cfg, records = tune_catboost_grid(
            X_tr,
            y_tr,
            X_va,
            y_va,
            cat_cols,
            loss_candidates=self.loss_candidates,
            depth_candidates=self.depth_candidates,
            lr_candidates=self.lr_candidates,
            iterations=self.iterations,
            early_stopping_rounds=self.early_stopping_rounds,
            seed=seed,
        )
        self.point_model = model
        self.best_cfg = best_cfg
        return best_cfg, records

    def fit_with_config(
        self,
        X_tr: pd.DataFrame,
        y_tr: np.ndarray,
        X_va: pd.DataFrame,
        y_va: np.ndarray,
        cat_cols: list[str],
        cfg: dict[str, Any],
        seed: int = 42,
    ) -> float:
        model, elapsed = train_catboost_single(
            X_tr,
            y_tr,
            X_va,
            y_va,
            cat_cols,
            loss_function=cfg.get("loss_function", "RMSE"),
            depth=int(cfg.get("depth", 6)),
            learning_rate=float(cfg.get("learning_rate", 0.05)),
            iterations=self.iterations,
            early_stopping_rounds=self.early_stopping_rounds,
            seed=seed,
        )
        self.point_model = model
        self.best_cfg = dict(cfg)
        return elapsed

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.point_model is None:
            raise RuntimeError("Point model not fitted.")
        return np.asarray(self.point_model.predict(X), dtype=np.float64)

    def fit_quantiles(
        self,
        X_tr: pd.DataFrame,
        y_tr: np.ndarray,
        X_va: pd.DataFrame,
        y_va: np.ndarray,
        cat_cols: list[str],
        quantiles: tuple[float, float] = (0.05, 0.95),
        seed: int = 42,
    ) -> float:
        depth = int(self.best_cfg.get("depth", 6))
        lr = float(self.best_cfg.get("learning_rate", 0.05))
        self.q_lo_model, self.q_hi_model, elapsed = fit_quantile_pair(
            X_tr,
            y_tr,
            X_va,
            y_va,
            cat_cols,
            depth=depth,
            learning_rate=lr,
            iterations=self.iterations,
            early_stopping_rounds=self.early_stopping_rounds,
            quantiles=quantiles,
            seed=seed,
        )
        return elapsed

    def predict_quantiles(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.q_lo_model is None or self.q_hi_model is None:
            raise RuntimeError("Quantile models not fitted.")
        return (
            np.asarray(self.q_lo_model.predict(X), dtype=np.float64),
            np.asarray(self.q_hi_model.predict(X), dtype=np.float64),
        )
