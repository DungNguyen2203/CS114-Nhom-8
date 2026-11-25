"""M0 hierarchical group median price-per-m2 baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd


class M0GroupMedianModel:
    """Predict log(price) from train median price/m2 with hierarchical fallback."""

    def __init__(self, min_group_samples: int = 30) -> None:
        self.min_group_samples = int(min_group_samples)
        self.lvl1_map: dict[tuple[str, str, str], float] = {}
        self.lvl2_map: dict[tuple[str, str], float] = {}
        self.lvl3_map: dict[str, float] = {}
        self.global_median_ppm2: float = 50_000_000.0

    def fit(self, train_df: pd.DataFrame) -> "M0GroupMedianModel":
        area = pd.to_numeric(train_df["area"], errors="coerce").to_numpy(dtype=np.float64)
        price = np.exp(pd.to_numeric(train_df["y_true"], errors="coerce").to_numpy(dtype=np.float64))
        ppm2 = price / np.clip(area, 1e-3, None)

        work = pd.DataFrame(
            {
                "prov": train_df["province_norm"].fillna("__MISSING__").astype(str),
                "dist": train_df["district_norm"].fillna("__MISSING__").astype(str),
                "ptype": train_df["property_type_name"].fillna("__MISSING__").astype(str),
                "ppm2": ppm2,
            }
        )
        self.global_median_ppm2 = float(np.nanmedian(ppm2))

        g1 = work.groupby(["prov", "dist", "ptype"])["ppm2"].agg(["median", "count"])
        self.lvl1_map = {
            idx: float(row["median"])
            for idx, row in g1[g1["count"] >= self.min_group_samples].iterrows()
        }

        g2 = work.groupby(["prov", "ptype"])["ppm2"].agg(["median", "count"])
        self.lvl2_map = {
            idx: float(row["median"])
            for idx, row in g2[g2["count"] >= self.min_group_samples].iterrows()
        }

        g3 = work.groupby("ptype")["ppm2"].median()
        self.lvl3_map = {str(idx): float(val) for idx, val in g3.items()}
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        provs = df["province_norm"].fillna("__MISSING__").astype(str).tolist()
        dists = df["district_norm"].fillna("__MISSING__").astype(str).tolist()
        ptypes = df["property_type_name"].fillna("__MISSING__").astype(str).tolist()
        areas = np.clip(pd.to_numeric(df["area"], errors="coerce").fillna(50.0).to_numpy(dtype=np.float64), 1e-3, None)

        preds_ppm2 = np.empty(len(df), dtype=np.float64)
        for i, (pr, di, pt) in enumerate(zip(provs, dists, ptypes)):
            val = self.lvl1_map.get((pr, di, pt))
            if val is None:
                val = self.lvl2_map.get((pr, pt))
            if val is None:
                val = self.lvl3_map.get(pt, self.global_median_ppm2)
            preds_ppm2[i] = val

        return np.log(np.clip(preds_ppm2 * areas, 1.0, None))
