"""M3 frozen PhoBERT embedding + L2 Ridge text-only model."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize


class M3PhoBERTRidgeModel:
    """L2-normalized PhoBERT embedding + Ridge regression tuned on validation MAE."""

    def __init__(self, alphas: list[float] | None = None) -> None:
        self.alphas = [float(a) for a in (alphas or [1.0, 10.0, 100.0])]
        self.model: Ridge | None = None
        self.best_alpha: float | None = None

    def tune_and_fit(
        self,
        emb_tr: np.ndarray,
        y_tr: np.ndarray,
        emb_va: np.ndarray,
        y_va: np.ndarray,
    ) -> tuple[float, list[dict[str, Any]]]:
        X_tr = normalize(np.asarray(emb_tr, dtype=np.float32), norm="l2")
        X_va = normalize(np.asarray(emb_va, dtype=np.float32), norm="l2")

        best_alpha = self.alphas[0]
        best_model: Ridge | None = None
        best_mae = float("inf")
        rows: list[dict[str, Any]] = []

        for alpha in self.alphas:
            t0 = time.perf_counter()
            reg = Ridge(alpha=alpha)
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

    def predict(self, emb: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("M3 model not fitted.")
        X = normalize(np.asarray(emb, dtype=np.float32), norm="l2")
        return np.asarray(self.model.predict(X), dtype=np.float64)
