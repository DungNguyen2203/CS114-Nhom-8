"""Shared CatBoostRegressor training, tuning, quantile regression, and importance helpers."""

from __future__ import annotations

import time
from typing import Any

from catboost import CatBoostRegressor, Pool
import numpy as np
import pandas as pd


def train_catboost_single(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    cat_cols: list[str],
    loss_function: str = "RMSE",
    depth: int = 6,
    learning_rate: float = 0.05,
    iterations: int = 2000,
    early_stopping_rounds: int = 100,
    seed: int = 42,
    eval_metric: str | None = "MAE",
) -> tuple[CatBoostRegressor, float]:
    """Fit a single CatBoostRegressor on CPU and return `(model, elapsed_sec)`.

    `eval_metric=None` lets CatBoost evaluate (and early-stop on) the training loss itself, e.g. the Quantile loss.
    """
    t0 = time.perf_counter()
    tr_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
    va_pool = Pool(X_va, y_va, cat_features=cat_cols)
    model = CatBoostRegressor(
        loss_function=loss_function,
        **({"eval_metric": eval_metric} if eval_metric else {}),
        iterations=int(iterations),
        learning_rate=float(learning_rate),
        depth=int(depth),
        random_seed=int(seed),
        early_stopping_rounds=int(early_stopping_rounds),
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
    return model, time.perf_counter() - t0


def tune_catboost_grid(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    cat_cols: list[str],
    loss_candidates: list[str],
    depth_candidates: list[int],
    lr_candidates: list[float],
    iterations: int = 2000,
    early_stopping_rounds: int = 100,
    seed: int = 42,
    max_configs: int = 8,
) -> tuple[CatBoostRegressor, dict[str, Any], list[dict[str, Any]]]:
    """Grid-search up to `max_configs` CatBoost configs by validation MAE."""
    configs: list[tuple[str, int, float]] = []
    for loss in loss_candidates:
        for depth in depth_candidates:
            for lr in lr_candidates:
                configs.append((str(loss), int(depth), float(lr)))
    configs = configs[:max_configs]

    best_model: CatBoostRegressor | None = None
    best_cfg: dict[str, Any] = {}
    best_mae = float("inf")
    records: list[dict[str, Any]] = []

    for loss, depth, lr in configs:
        model, elapsed = train_catboost_single(
            X_tr,
            y_tr,
            X_va,
            y_va,
            cat_cols,
            loss_function=loss,
            depth=depth,
            learning_rate=lr,
            iterations=iterations,
            early_stopping_rounds=early_stopping_rounds,
            seed=seed,
        )
        preds = model.predict(X_va)
        val_mae = float(np.mean(np.abs(preds - y_va)))
        rec = {
            "loss_function": loss,
            "depth": depth,
            "learning_rate": lr,
            "best_iteration": model.get_best_iteration(),
            "iterations": iterations,
            "early_stopping_rounds": early_stopping_rounds,
            "seed": seed,
            "val_mae": val_mae,
            "train_time_sec": elapsed,
        }
        records.append(rec)
        if val_mae < best_mae:
            best_mae = val_mae
            best_model = model
            best_cfg = rec

    assert best_model is not None
    return best_model, best_cfg, records


def fit_quantile_pair(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    cat_cols: list[str],
    depth: int = 6,
    learning_rate: float = 0.05,
    iterations: int = 2000,
    early_stopping_rounds: int = 100,
    quantiles: tuple[float, float] = (0.05, 0.95),
    seed: int = 42,
) -> tuple[CatBoostRegressor, CatBoostRegressor, float]:
    """Train lower and upper CatBoost Quantile regressors."""
    q_lo_val, q_hi_val = float(quantiles[0]), float(quantiles[1])
    m_lo, t1 = train_catboost_single(
        X_tr,
        y_tr,
        X_va,
        y_va,
        cat_cols,
        loss_function=f"Quantile:alpha={q_lo_val}",
        eval_metric=None,
        depth=depth,
        learning_rate=learning_rate,
        iterations=iterations,
        early_stopping_rounds=early_stopping_rounds,
        seed=seed,
    )
    m_hi, t2 = train_catboost_single(
        X_tr,
        y_tr,
        X_va,
        y_va,
        cat_cols,
        loss_function=f"Quantile:alpha={q_hi_val}",
        eval_metric=None,
        depth=depth,
        learning_rate=learning_rate,
        iterations=iterations,
        early_stopping_rounds=early_stopping_rounds,
        seed=seed,
    )
    return m_lo, m_hi, t1 + t2


def concat_dense_to_tabular(X_tab: pd.DataFrame, dense: np.ndarray, prefix: str) -> pd.DataFrame:
    """Append a 2D float matrix to a tabular DataFrame preserving index and column names."""
    dense_cols = {f"{prefix}_{i}": dense[:, i].astype(np.float32) for i in range(dense.shape[1])}
    dense_df = pd.DataFrame(dense_cols, index=X_tab.index)
    return pd.concat([X_tab, dense_df], axis=1)


def compute_catboost_importance(
    model: CatBoostRegressor,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_cols: list[str],
    max_rows: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute PredictionValuesChange and SHAP mean(|value|) importance on a validation slice."""
    if len(X_val) > max_rows:
        sample_idx = np.random.default_rng(seed).choice(len(X_val), size=max_rows, replace=False)
        X_sub = X_val.iloc[sample_idx]
        y_sub = y_val[sample_idx]
    else:
        X_sub = X_val
        y_sub = y_val

    pool = Pool(X_sub, y_sub, cat_features=cat_cols)
    pvc = model.get_feature_importance(pool, type="PredictionValuesChange")
    shap_vals = model.get_feature_importance(pool, type="ShapValues")[:, :-1]
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    imp_df = pd.DataFrame(
        {
            "feature": list(X_sub.columns),
            "importance_pvc": pvc,
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return imp_df
