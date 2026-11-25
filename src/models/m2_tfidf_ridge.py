"""M2 text-only TF-IDF + Ridge baseline."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.sparse import spmatrix
from sklearn.linear_model import Ridge


class M2TfidfRidgeModel:
    """Sparse TF-IDF + L2 Ridge regression tuned on validation MAE."""

    def __init__(self, alphas: list[float] | None = None) -> None:
        self.alphas = [float(a) for a in (alphas or [1.0, 10.0, 100.0])]
        self.model: Ridge | None = None
        self.best_alpha: float | None = None

    def tune_and_fit(
        self,
        X_tr: spmatrix,
        y_tr: np.ndarray,
        X_va: spmatrix,
        y_va: np.ndarray,
    ) -> tuple[float, list[dict[str, Any]]]:
        best_alpha = self.alphas[0]
        best_model: Ridge | None = None
        best_mae = float("inf")
        rows: list[dict[str, Any]] = []

        for alpha in self.alphas:
            t0 = time.perf_counter()
            reg = Ridge(alpha=alpha, solver="lsqr")
            reg.fit(X_tr, y_tr)
            elapsed = time.perf_counter() - t0
            preds = reg.predict(X_va)
            val_mae = float(np.mean(np.abs(preds - y_va)))
            rows.append({"alpha": alpha, "val_mae": val_mae, "train_time_sec": elapsed})
            if val_mae < best_mae:
                best_mae = val_mae
                best_alpha = alpha
                best_model = reg

        self.model = best_model
        self.best_alpha = best_alpha
        return best_alpha, rows

    def predict(self, X: spmatrix) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("M2 model not fitted.")
        return np.asarray(self.model.predict(X), dtype=np.float64)
