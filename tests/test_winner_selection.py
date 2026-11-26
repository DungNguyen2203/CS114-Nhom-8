import pytest
import numpy as np
import pandas as pd

from src.config import load_config
from src.features.phobert_embedder import save_embedding_cache
from src.pipelines.train_eval import _pick_conformal_model, _run_scenario_b
from tests.test_tabular_and_baselines import _make_tabular_splits


def test_conformal_model_is_best_among_quantile_capable_not_silent_m5():
    # M0 has the lowest validation MAE but no quantile heads: must not be picked, nor silently replaced by M5.
    val = {"M0": 0.55, "M1": 0.60, "M4": 0.566, "M5": 0.5658}
    assert min(val, key=val.get) == "M0"
    assert _pick_conformal_model(val) == "M5"
    assert _pick_conformal_model({"M0": 0.5, "M1": 0.6, "M4": 0.55, "M5": 0.58}) == "M4"
    with pytest.raises(RuntimeError):
        _pick_conformal_model({"M0": 0.5, "M2": 0.6})


def test_scenario_b_tunes_all_models_without_holdout_labels(tmp_path):
    cfg = load_config("configs/local_smoke.yaml")
    cfg["features"].update(tfidf_min_df=1, svd_components_candidates=[2, 3], pca_components_candidates=[2, 3])
    cfg["models"]["catboost"].update(iterations=3, depth_candidates=[2], lr_candidates=[0.1], loss_candidates=["RMSE"])
    train, rest = _make_tabular_splits(n_train=60, n_val=36)
    clean = pd.concat([train, rest], ignore_index=True)
    manifest = clean[["row_id"]].assign(split=["train"] * 60 + ["val"] * 12 + ["cal"] * 12 + ["test"] * 12,
                                        dup_group_id=clean.row_id, month="2025-06")
    manifest["district_key"] = manifest.split + "|" + clean.district_norm
    paths = {"processed_dir": tmp_path, "outputs_dir": tmp_path}
    embeddings = np.random.default_rng(42).normal(size=(len(clean), 6))
    save_embedding_cache(tmp_path / "embeddings", clean.row_id, embeddings, cfg["project"]["text_version"], "synthetic")

    def run(rows):
        result = _run_scenario_b(cfg, paths, rows, manifest)
        point = pd.DataFrame(result["tuning_logs"]).dropna(subset=["val_mae"])
        assert set(point.model_id) == {"M1", "M4", "M5"}
        assert set(point.svd_k.dropna()) == {2, 3}
        assert set(point.pca_k.dropna()) == {2, 3}
        return result["winner_id"], point.val_mae.to_numpy()

    winner, maes = run(clean)
    clean.loc[manifest.split.isin(["cal", "test"]), "y_true"] += 1.
    second_winner, second_maes = run(clean)
    assert winner == second_winner
    np.testing.assert_allclose(maes, second_maes, rtol=1e-10)
