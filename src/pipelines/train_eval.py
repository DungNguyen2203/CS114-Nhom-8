"""TV2/TV3/TV4 unified training, tuning, E1-E6 evaluation, conformal calibration, and bundle saving."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import load_config, resolve_project_paths, set_global_seed
from src.data.splits import stratified_sample_df
from src.data.text_normalizer import redact_pii
from src.eval.conformal import fit_apply_conformal
from src.eval.error_analysis import analyze_residual_keywords, sample_error_buckets
from src.eval.flagger import export_flag_audit_samples, flag_mispriced_listings
from src.eval.leakage_checks import verify_split_leakage
from src.eval.metrics import cluster_bootstrap_mae_diff, evaluate_predictions_df
from src.features.phobert_embedder import PhoBERTEmbedder, format_phobert_input, load_embeddings_for_row_ids
from src.features.tabular_features import TabularFeatureExtractor
from src.features.tfidf_features import TfidfSvdExtractor
from src.models.catboost_utils import compute_catboost_importance, concat_dense_to_tabular
from src.models.contract import make_prediction_df, save_predictions
from src.models.m0_median import M0GroupMedianModel
from src.models.m1_catboost import M1CatBoostModel
from src.models.m2_tfidf_ridge import M2TfidfRidgeModel
from src.models.m3_phobert_ridge import M3PhoBERTRidgeModel
from src.models.m4_tfidf_catboost import M4TfidfCatBoostModel
from src.models.m5_phobert_catboost import M5PhoBERTCatBoostModel
from src.models.m6_finetune import M6FineTuneRegressor


def _slice_by_split(clean_df: pd.DataFrame, man_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    merged = man_df[["row_id", "split", "dup_group_id", "district_key", "month"]].merge(
        clean_df,
        on="row_id",
        how="inner",
        suffixes=("", "_clean"),
    )
    return {
        sp: merged[merged["split"] == sp].reset_index(drop=True)
        for sp in ["train", "val", "cal", "test"]
    }


# Only models with quantile heads can produce conformal intervals and a deployable bundle.
_CONFORMAL_CAPABLE = ("M1", "M4", "M5")


def _pick_conformal_model(val_mae_by_model: dict[str, float]) -> str:
    """Best validation-MAE model among those that have quantile heads (never silently M5)."""
    cand = {k: v for k, v in val_mae_by_model.items() if k in _CONFORMAL_CAPABLE}
    if not cand:
        raise RuntimeError(f"No conformal-capable model (M1/M4/M5) in {sorted(val_mae_by_model)}")
    return min(cand, key=lambda k: cand[k])


def _measure_latency(predict_fn, n_rows: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    _ = predict_fn()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    per_row_ms = elapsed_ms / max(1, n_rows)
    return {"batch_total_ms": elapsed_ms, "per_row_mean_ms": per_row_ms}


_LISTING_COLS = {
    "property_type_name": "property_type_name", "area": "area", "floor_count": "floor_count",
    "frontage_width": "frontage_width", "house_depth": "house_depth", "road_width": "road_width",
    "bedroom_count": "bedroom_count", "bathroom_count": "bathroom_count",
    "province_name": "province_norm", "district_name": "district_norm", "ward_name": "ward_norm",
    "street_name": "street_norm", "project_name": "project_norm",
    "house_direction": "house_direction_norm", "balcony_direction": "balcony_direction_norm",
    "name": "name_norm", "description": "description_norm",
}


def _measure_single_row_latency(bundle_dir: Path, te_df: pd.DataFrame, n_rows: int = 50, seed: int = 42) -> dict[str, float]:
    """End-to-end latency (mask + features + model + conformal) via the demo service: median and p95 per listing."""
    from app.inference import ValuationInferenceService

    svc = ValuationInferenceService.load(bundle_dir)
    rows = te_df.sample(n=min(n_rows, len(te_df)), random_state=seed)
    listings = [
        {k: (None if pd.isna(r[c]) else r[c]) for k, c in _LISTING_COLS.items() if c in rows.columns}
        for _, r in rows.iterrows()
    ]
    svc.predict_single(listings[0])  # warm-up (lazy model loads)
    times_ms = []
    for lst in listings:
        t0 = time.perf_counter()
        svc.predict_single(lst)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return {
        "single_row_median_ms": float(np.median(times_ms)),
        "single_row_p95_ms": float(np.percentile(times_ms, 95)),
        "single_row_n": len(times_ms),
    }


def _run_scenario_a(
    cfg: dict[str, Any],
    paths: dict[str, Path],
    clean_df: pd.DataFrame,
    man_a: pd.DataFrame,
    noisy_df: pd.DataFrame,
) -> dict[str, Any]:
    preds_dir = paths["outputs_dir"] / "predictions"
    metrics_dir = paths["outputs_dir"] / "metrics"
    audits_dir = paths["outputs_dir"] / "audits"
    preds_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    splits = _slice_by_split(clean_df, man_a)
    tr, va, ca, te = splits["train"], splits["val"], splits["cal"], splits["test"]
    y_tr, y_va, y_ca, y_te = tr["y_true"].to_numpy(), va["y_true"].to_numpy(), ca["y_true"].to_numpy(), te["y_true"].to_numpy()

    seeds = [int(s) for s in cfg["project"]["seeds"]]
    primary_seed = int(cfg["project"]["primary_seed"])
    cb_cfg = cfg["models"]["catboost"]
    feat_cfg = cfg["features"]
    cache_dir = paths["processed_dir"] / "embeddings"

    # Extract tabular features strictly fit on train
    tab_ext = TabularFeatureExtractor(min_freq=int(feat_cfg["rare_category_min_freq"]))
    X_tr_tab, cat_cols = tab_ext.fit_transform(tr)
    X_va_tab, _ = tab_ext.transform(va)
    X_ca_tab, _ = tab_ext.transform(ca)
    X_te_tab, _ = tab_ext.transform(te)

    # TF-IDF & SVD strictly fit on train
    tfidf_ext = TfidfSvdExtractor(
        max_features=int(feat_cfg["tfidf_max_features"]),
        min_df=int(feat_cfg["tfidf_min_df"]),
        ngram_range=(1, 2),
        svd_components=int(feat_cfg["svd_components_candidates"][0]),
        seed=primary_seed,
    )
    X_tr_sp, X_tr_svd = tfidf_ext.fit_transform(tr["text_masked"].tolist(), fit_svd=True)
    X_va_sp, X_va_svd = tfidf_ext.transform(va["text_masked"].tolist())
    X_ca_sp, X_ca_svd = tfidf_ext.transform(ca["text_masked"].tolist())
    X_te_sp, X_te_svd = tfidf_ext.transform(te["text_masked"].tolist())

    # Load PhoBERT embeddings aligned strictly by row_id
    emb_tr = load_embeddings_for_row_ids(tr["row_id"].tolist(), cache_dir, cfg["project"]["text_version"])
    emb_va = load_embeddings_for_row_ids(va["row_id"].tolist(), cache_dir, cfg["project"]["text_version"])
    emb_ca = load_embeddings_for_row_ids(ca["row_id"].tolist(), cache_dir, cfg["project"]["text_version"])
    emb_te = load_embeddings_for_row_ids(te["row_id"].tolist(), cache_dir, cfg["project"]["text_version"])

    all_metrics: list[dict[str, Any]] = []
    tuning_logs: list[dict[str, Any]] = []
    val_mae_by_model: dict[str, float] = {}
    pred_dfs_test: dict[str, pd.DataFrame] = {}

    def _record_model(model_id: str, seed: int, sp_name: str, df_sp: pd.DataFrame, y_pred: np.ndarray, q_lo=None, q_hi=None, train_sec: float = 0.0) -> pd.DataFrame:
        p_df = make_prediction_df(
            row_ids=df_sp["row_id"].tolist(),
            split=sp_name,
            scenario="A",
            model_id=model_id,
            seed=seed,
            y_true=df_sp["y_true"].to_numpy(),
            y_pred=y_pred,
            q_lo=q_lo,
            q_hi=q_hi,
        )
        save_predictions(p_df, preds_dir / f"A_{model_id}_{sp_name}_seed{seed}.parquet")
        ev = evaluate_predictions_df(p_df, clean_df, train_df=tr, alpha=float(cfg["conformal"]["alpha_primary"]), min_eval_group_size=int(cfg["conformal"]["min_eval_group_size"]))
        row = {"scenario": "A", "model_id": model_id, "seed": seed, "split": sp_name, "train_time_sec": train_sec, **ev["overall"]}
        all_metrics.append(row)
        return p_df

    # --- M0 ---
    print(f"[Scenario A] Training M0/M1/M2/M3/M4/M5 (train={len(tr):,}, val={len(va):,}, cal={len(ca):,}, test={len(te):,})...", flush=True)
    t0 = time.perf_counter()
    m0 = M0GroupMedianModel(min_group_samples=int(cfg["models"]["m0_min_group_samples"])).fit(tr)
    t_m0 = time.perf_counter() - t0
    for sp_name, sp_df in [("val", va), ("cal", ca), ("test", te)]:
        p = m0.predict(sp_df)
        pdf = _record_model("M0", primary_seed, sp_name, sp_df, p, train_sec=t_m0)
        if sp_name == "val":
            val_mae_by_model["M0"] = float(np.mean(np.abs(p - y_va)))
        if sp_name == "test":
            pred_dfs_test["M0"] = pdf

    # --- M1 (tune + multi-seed + quantiles) ---
    print("[Scenario A] Tuning & training M1 (Tabular CatBoost)...", flush=True)
    m1 = M1CatBoostModel(
        iterations=int(cb_cfg["iterations"]),
        early_stopping_rounds=int(cb_cfg["early_stopping_rounds"]),
        loss_candidates=cb_cfg["loss_candidates"],
        depth_candidates=cb_cfg["depth_candidates"],
        lr_candidates=cb_cfg["lr_candidates"],
    )
    m1_best_cfg, m1_tune = m1.tune_and_fit(X_tr_tab, y_tr, X_va_tab, y_va, cat_cols, seed=primary_seed)
    for r in m1_tune:
        tuning_logs.append({"model_id": "M1", **r})
    t_m1_q = m1.fit_quantiles(X_tr_tab, y_tr, X_va_tab, y_va, cat_cols, quantiles=tuple(cb_cfg["quantiles"]), seed=primary_seed)
    m1_imp = compute_catboost_importance(m1.point_model, X_va_tab, y_va, cat_cols, seed=primary_seed)
    m1_imp.to_csv(metrics_dir / "feature_importance_val.csv", index=False)

    m1_cal_df = None
    m1_test_df = None
    for s in seeds:
        m1_s = m1 if s == primary_seed else M1CatBoostModel(iterations=int(cb_cfg["iterations"]), early_stopping_rounds=int(cb_cfg["early_stopping_rounds"]))
        t_s = m1_best_cfg["train_time_sec"] if s == primary_seed else m1_s.fit_with_config(X_tr_tab, y_tr, X_va_tab, y_va, cat_cols, m1_best_cfg, seed=s)
        for sp_name, sp_df, X_sp in [("val", va, X_va_tab), ("cal", ca, X_ca_tab), ("test", te, X_te_tab)]:
            yp = m1_s.predict(X_sp)
            q_lo, q_hi = m1.predict_quantiles(X_sp) if s == primary_seed else (None, None)
            pdf = _record_model("M1", s, sp_name, sp_df, yp, q_lo=q_lo, q_hi=q_hi, train_sec=t_s)
            if s == primary_seed and sp_name == "val":
                val_mae_by_model["M1"] = float(np.mean(np.abs(yp - y_va)))
            if s == primary_seed and sp_name == "cal":
                m1_cal_df = pdf
            if s == primary_seed and sp_name == "test":
                m1_test_df = pdf
                pred_dfs_test["M1"] = pdf

    # --- M2 (TF-IDF + Ridge) ---
    m2 = M2TfidfRidgeModel(alphas=cfg["models"]["ridge_alphas"])
    m2_alpha, m2_tune = m2.tune_and_fit(X_tr_sp, y_tr, X_va_sp, y_va)
    m2_best_sec = next((r["train_time_sec"] for r in m2_tune if float(r["alpha"]) == float(m2_alpha)), m2_tune[0]["train_time_sec"])
    for r in m2_tune:
        tuning_logs.append({"model_id": "M2", **r})
    for sp_name, sp_df, X_sp in [("val", va, X_va_sp), ("cal", ca, X_ca_sp), ("test", te, X_te_sp)]:
        yp = m2.predict(X_sp)
        pdf = _record_model("M2", primary_seed, sp_name, sp_df, yp, train_sec=m2_best_sec)
        if sp_name == "val":
            val_mae_by_model["M2"] = float(np.mean(np.abs(yp - y_va)))
        if sp_name == "test":
            pred_dfs_test["M2"] = pdf

    # --- M3 (PhoBERT + Ridge) ---
    m3 = M3PhoBERTRidgeModel(alphas=cfg["models"]["ridge_alphas"])
    m3_alpha, m3_tune = m3.tune_and_fit(emb_tr, y_tr, emb_va, y_va)
    m3_best_sec = next((r["train_time_sec"] for r in m3_tune if float(r["alpha"]) == float(m3_alpha)), m3_tune[0]["train_time_sec"])
    for r in m3_tune:
        tuning_logs.append({"model_id": "M3", **r})
    for sp_name, sp_df, emb_sp in [("val", va, emb_va), ("cal", ca, emb_ca), ("test", te, emb_te)]:
        yp = m3.predict(emb_sp)
        pdf = _record_model("M3", primary_seed, sp_name, sp_df, yp, train_sec=m3_best_sec)
        if sp_name == "val":
            val_mae_by_model["M3"] = float(np.mean(np.abs(yp - y_va)))
        if sp_name == "test":
            pred_dfs_test["M3"] = pdf

    # --- M4 (TF-IDF SVD + Tabular CatBoost) ---
    print("[Scenario A] Tuning & training M2, M3, M4...", flush=True)
    best_svd_k = int(feat_cfg["svd_components_candidates"][0])
    best_m4_mae = float("inf")
    best_m4 = None
    best_m4_ext = tfidf_ext
    best_m4_sec = 0.0
    for k in feat_cfg["svd_components_candidates"]:
        ext_k = tfidf_ext if int(k) == best_svd_k else TfidfSvdExtractor(
            max_features=int(feat_cfg["tfidf_max_features"]),
            min_df=int(feat_cfg["tfidf_min_df"]),
            ngram_range=(1, 2),
            svd_components=int(k),
            seed=primary_seed,
        ).fit(tr["text_masked"].tolist(), fit_svd=True)
        tr_svd_k = X_tr_svd if int(k) == best_svd_k else ext_k.transform_svd(tr["text_masked"].tolist())
        va_svd_k = X_va_svd if int(k) == best_svd_k else ext_k.transform_svd(va["text_masked"].tolist())
        m4_k = M4TfidfCatBoostModel(
            iterations=int(cb_cfg["iterations"]),
            early_stopping_rounds=int(cb_cfg["early_stopping_rounds"]),
            depth=int(m1_best_cfg["depth"]),
            learning_rate=float(m1_best_cfg["learning_rate"]),
            loss_function=str(m1_best_cfg["loss_function"]),
        )
        t_k = m4_k.fit(X_tr_tab, tr_svd_k, y_tr, X_va_tab, va_svd_k, y_va, cat_cols, seed=primary_seed)
        mae_k = float(np.mean(np.abs(m4_k.predict(X_va_tab, va_svd_k) - y_va)))
        tuning_logs.append({"model_id": "M4", "svd_k": int(k), "val_mae": mae_k, "train_time_sec": t_k})
        if mae_k < best_m4_mae:
            best_m4_mae = mae_k
            best_m4 = m4_k
            best_svd_k = int(k)
            best_m4_ext = ext_k
            best_m4_sec = t_k
    assert best_m4 is not None
    tr_svd_b = best_m4_ext.transform_svd(tr["text_masked"].tolist())
    va_svd_b = best_m4_ext.transform_svd(va["text_masked"].tolist())
    ca_svd_b = best_m4_ext.transform_svd(ca["text_masked"].tolist())
    te_svd_b = best_m4_ext.transform_svd(te["text_masked"].tolist())
    best_m4.fit_quantiles(X_tr_tab, tr_svd_b, y_tr, X_va_tab, va_svd_b, y_va, cat_cols, quantiles=tuple(cb_cfg["quantiles"]), seed=primary_seed)
    for s in seeds:
        m4_s = best_m4 if s == primary_seed else M4TfidfCatBoostModel(
            iterations=int(cb_cfg["iterations"]),
            early_stopping_rounds=int(cb_cfg["early_stopping_rounds"]),
            depth=int(m1_best_cfg["depth"]),
            learning_rate=float(m1_best_cfg["learning_rate"]),
            loss_function=str(m1_best_cfg["loss_function"]),
        )
        t_s = best_m4_sec if s == primary_seed else m4_s.fit(X_tr_tab, tr_svd_b, y_tr, X_va_tab, va_svd_b, y_va, cat_cols, seed=s)
        for sp_name, sp_df, X_sp, svd_sp in [("val", va, X_va_tab, va_svd_b), ("cal", ca, X_ca_tab, ca_svd_b), ("test", te, X_te_tab, te_svd_b)]:
            yp = m4_s.predict(X_sp, svd_sp)
            q_lo, q_hi = best_m4.predict_quantiles(X_sp, svd_sp) if s == primary_seed else (None, None)
            pdf = _record_model("M4", s, sp_name, sp_df, yp, q_lo=q_lo, q_hi=q_hi, train_sec=t_s)
            if s == primary_seed and sp_name == "val":
                val_mae_by_model["M4"] = float(np.mean(np.abs(yp - y_va)))
            if s == primary_seed and sp_name == "test":
                pred_dfs_test["M4"] = pdf

    # --- M5 (PhoBERT PCA + Tabular CatBoost) ---
    print("[Scenario A] Tuning & training M5 (PhoBERT + Tabular CatBoost)...", flush=True)
    best_pca_k = int(feat_cfg["pca_components_candidates"][0])
    best_m5_mae = float("inf")
    best_m5 = None
    best_m5_sec = 0.0
    for k in feat_cfg["pca_components_candidates"]:
        m5_k = M5PhoBERTCatBoostModel(
            pca_components=int(k),
            iterations=int(cb_cfg["iterations"]),
            early_stopping_rounds=int(cb_cfg["early_stopping_rounds"]),
            depth=int(m1_best_cfg["depth"]),
            learning_rate=float(m1_best_cfg["learning_rate"]),
            loss_function=str(m1_best_cfg["loss_function"]),
        )
        t_k = m5_k.fit(X_tr_tab, emb_tr, y_tr, X_va_tab, emb_va, y_va, cat_cols, seed=primary_seed)
        mae_k = float(np.mean(np.abs(m5_k.predict(X_va_tab, emb_va) - y_va)))
        tuning_logs.append({"model_id": "M5", "pca_k": int(k), "val_mae": mae_k, "train_time_sec": t_k})
        if mae_k < best_m5_mae:
            best_m5_mae = mae_k
            best_m5 = m5_k
            best_pca_k = int(k)
            best_m5_sec = t_k
    assert best_m5 is not None
    best_m5.fit_quantiles(X_tr_tab, emb_tr, y_tr, X_va_tab, emb_va, y_va, cat_cols, quantiles=tuple(cb_cfg["quantiles"]), seed=primary_seed)
    m5_cal_df = None
    m5_test_df = None
    for s in seeds:
        m5_s = best_m5 if s == primary_seed else M5PhoBERTCatBoostModel(
            pca_components=best_pca_k,
            iterations=int(cb_cfg["iterations"]),
            early_stopping_rounds=int(cb_cfg["early_stopping_rounds"]),
            depth=int(m1_best_cfg["depth"]),
            learning_rate=float(m1_best_cfg["learning_rate"]),
            loss_function=str(m1_best_cfg["loss_function"]),
        )
        t_s = best_m5_sec if s == primary_seed else m5_s.fit(X_tr_tab, emb_tr, y_tr, X_va_tab, emb_va, y_va, cat_cols, seed=s)
        for sp_name, sp_df, X_sp, emb_sp in [("val", va, X_va_tab, emb_va), ("cal", ca, X_ca_tab, emb_ca), ("test", te, X_te_tab, emb_te)]:
            yp = m5_s.predict(X_sp, emb_sp)
            q_lo, q_hi = best_m5.predict_quantiles(X_sp, emb_sp) if s == primary_seed else (None, None)
            pdf = _record_model("M5", s, sp_name, sp_df, yp, q_lo=q_lo, q_hi=q_hi, train_sec=t_s)
            if s == primary_seed and sp_name == "val":
                val_mae_by_model["M5"] = float(np.mean(np.abs(yp - y_va)))
            if s == primary_seed and sp_name == "cal":
                m5_cal_df = pdf
            if s == primary_seed and sp_name == "test":
                m5_test_df = pdf
                pred_dfs_test["M5"] = pdf

    # --- Optional M6 (End-to-End PhoBERT Fine-Tune) ---
    m6_cfg = cfg["models"].get("m6_finetune", {})
    if bool(m6_cfg.get("enabled", False)):
        print("[Scenario A] Training optional M6 (End-to-End PhoBERT fine-tune)...", flush=True)
        ph_cfg = cfg["phobert"]
        m6 = M6FineTuneRegressor(
            model_name=ph_cfg["model_name"],
            revision=ph_cfg["revision"],
            max_length=int(ph_cfg["max_length"]),
            lr=float(m6_cfg.get("lr", 1.5e-5)),
            epochs=int(m6_cfg.get("epochs", 2)),
            batch_size=int(m6_cfg.get("batch_size", 16)),
        )
        def _fmt_df(d: pd.DataFrame) -> list[str]:
            return [
                format_phobert_input(pt, pr, dt, nm, dm)
                for pt, pr, dt, nm, dm in zip(
                    d["property_type_name"], d["province_norm"], d["district_norm"], d["name_masked"], d["description_masked"]
                )
            ]
        t0_m6 = time.perf_counter()
        m6_res = m6.fit(_fmt_df(tr), y_tr, _fmt_df(va), y_va, checkpoint_dir=paths["models_dir"] / "m6_a", seed=primary_seed)
        t_m6 = time.perf_counter() - t0_m6
        tuning_logs.append({"model_id": "M6", "val_mae": m6_res["best_val_mae"], "train_time_sec": t_m6})
        for sp_name, sp_df in [("val", va), ("cal", ca), ("test", te)]:
            yp = m6.predict(_fmt_df(sp_df))
            pdf = _record_model("M6", primary_seed, sp_name, sp_df, yp, train_sec=t_m6)
            if sp_name == "val":
                val_mae_by_model["M6"] = float(np.mean(np.abs(yp - y_va)))
            if sp_name == "test":
                pred_dfs_test["M6"] = pdf

    # Validation winner over ALL models (reported as-is), and the best model that has quantile heads
    # (M1/M4/M5): only those can feed conformal intervals and the deployable bundle.
    val_winner_id = min(val_mae_by_model, key=lambda k: val_mae_by_model[k])
    winner_id = _pick_conformal_model(val_mae_by_model)
    print(
        f"[Scenario A] Validation winner: {val_winner_id}; conformal/bundle model: {winner_id} ({val_mae_by_model}). Running E3 ablation...",
        flush=True,
    )

    # E3: Fair apple-to-apple stratified ablation & unmasked leakage control on Scenario A
    e3_rows: list[dict[str, Any]] = [
        {"variant": "M1_tabular_only_full", "MAE_log": float(np.mean(np.abs(pred_dfs_test["M1"]["y_pred"] - y_te))), "n_train": len(tr), "n_test": len(te)},
        {"variant": "M4_masked_full", "MAE_log": float(np.mean(np.abs(pred_dfs_test["M4"]["y_pred"] - y_te))), "n_train": len(tr), "n_test": len(te)},
        {"variant": "M5_masked_full", "MAE_log": float(np.mean(np.abs(pred_dfs_test["M5"]["y_pred"] - y_te))), "n_train": len(tr), "n_test": len(te)},
    ]
    ph_cfg = cfg["phobert"]
    e3_tr_n = min(len(tr), 5000)
    e3_va_n = min(len(va), 1500)
    e3_te_n = min(len(te), 1500)
    if e3_tr_n >= 50 and e3_va_n >= 10 and e3_te_n >= 10:
        def _e3_stratified_idx(df_sp: pd.DataFrame, n_target: int) -> tuple[pd.DataFrame, np.ndarray]:
            strata_cols = [c for c in ["province_norm", "property_type_name"] if c in df_sp.columns] or ["row_id"]
            sub_idx = stratified_sample_df(df_sp.assign(_pos=np.arange(len(df_sp))), strata_cols, n_total=n_target, seed=primary_seed, id_col="row_id")
            pos = sub_idx["_pos"].to_numpy(dtype=int)
            return df_sp.iloc[pos].reset_index(drop=True), pos

        tr_e3, tr_pos = _e3_stratified_idx(tr, e3_tr_n)
        va_e3, va_pos = _e3_stratified_idx(va, e3_va_n)
        te_e3, te_pos = _e3_stratified_idx(te, e3_te_n)

        # Same-slice M1 tabular baseline
        m1_e3 = M1CatBoostModel(iterations=int(cb_cfg["iterations"]), early_stopping_rounds=int(cb_cfg["early_stopping_rounds"]))
        m1_e3.fit_with_config(X_tr_tab.iloc[tr_pos], y_tr[tr_pos], X_va_tab.iloc[va_pos], y_va[va_pos], cat_cols, m1_best_cfg, seed=primary_seed)
        e3_rows.append({"variant": "M1_tabular_only_slice", "MAE_log": float(np.mean(np.abs(m1_e3.predict(X_te_tab.iloc[te_pos]) - y_te[te_pos]))), "n_train": e3_tr_n, "n_test": e3_te_n})

        # Same-slice M4 masked vs unmasked control
        m4_ext_m = TfidfSvdExtractor(max_features=int(feat_cfg["tfidf_max_features"]), min_df=int(feat_cfg["tfidf_min_df"]), ngram_range=(1, 2), svd_components=best_svd_k, seed=primary_seed)
        _, svd_m_tr = m4_ext_m.fit_transform(tr_e3["text_masked"].tolist(), fit_svd=True)
        svd_m_va = m4_ext_m.transform_svd(va_e3["text_masked"].tolist())
        svd_m_te = m4_ext_m.transform_svd(te_e3["text_masked"].tolist())
        m4_e3_m = M4TfidfCatBoostModel(iterations=int(cb_cfg["iterations"]), early_stopping_rounds=int(cb_cfg["early_stopping_rounds"]), depth=int(m1_best_cfg["depth"]), learning_rate=float(m1_best_cfg["learning_rate"]), loss_function=str(m1_best_cfg["loss_function"]))
        m4_e3_m.fit(X_tr_tab.iloc[tr_pos], svd_m_tr, y_tr[tr_pos], X_va_tab.iloc[va_pos], svd_m_va, y_va[va_pos], cat_cols, seed=primary_seed)
        e3_rows.append({"variant": "M4_masked_slice", "MAE_log": float(np.mean(np.abs(m4_e3_m.predict(X_te_tab.iloc[te_pos], svd_m_te) - y_te[te_pos]))), "n_train": e3_tr_n, "n_test": e3_te_n})

        unmasked_tr_txt = [f"{a}. {b}" for a, b in zip(tr_e3["name_norm"], tr_e3["description_norm"])]
        unmasked_va_txt = [f"{a}. {b}" for a, b in zip(va_e3["name_norm"], va_e3["description_norm"])]
        unmasked_te_txt = [f"{a}. {b}" for a, b in zip(te_e3["name_norm"], te_e3["description_norm"])]

        # Sensitive leak control: full sparse TF-IDF + Ridge sees every token (SVD/PCA in M4/M5 compress away most price text).
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import Ridge

        def _sparse_ridge_mae(txt_tr: list[str], txt_te: list[str]) -> float:
            vec = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=int(feat_cfg["tfidf_max_features"]),
                min_df=int(feat_cfg["tfidf_min_df"]),
                sublinear_tf=True,
                dtype=np.float32,
            )
            rg = Ridge(alpha=1.0).fit(vec.fit_transform(txt_tr), y_tr[tr_pos])
            return float(np.mean(np.abs(rg.predict(vec.transform(txt_te)) - y_te[te_pos])))

        e3_rows.append({"variant": "M2sparse_masked_slice", "MAE_log": _sparse_ridge_mae(tr_e3["text_masked"].tolist(), te_e3["text_masked"].tolist()), "n_train": e3_tr_n, "n_test": e3_te_n})
        e3_rows.append({"variant": "M2sparse_unmasked_leak_control_slice", "MAE_log": _sparse_ridge_mae(unmasked_tr_txt, unmasked_te_txt), "n_train": e3_tr_n, "n_test": e3_te_n})

        m4_ext_u = TfidfSvdExtractor(max_features=int(feat_cfg["tfidf_max_features"]), min_df=int(feat_cfg["tfidf_min_df"]), ngram_range=(1, 2), svd_components=best_svd_k, seed=primary_seed)
        _, svd_u_tr = m4_ext_u.fit_transform(unmasked_tr_txt, fit_svd=True)
        svd_u_va = m4_ext_u.transform_svd(unmasked_va_txt)
        svd_u_te = m4_ext_u.transform_svd(unmasked_te_txt)
        m4_e3_u = M4TfidfCatBoostModel(iterations=int(cb_cfg["iterations"]), early_stopping_rounds=int(cb_cfg["early_stopping_rounds"]), depth=int(m1_best_cfg["depth"]), learning_rate=float(m1_best_cfg["learning_rate"]), loss_function=str(m1_best_cfg["loss_function"]))
        m4_e3_u.fit(X_tr_tab.iloc[tr_pos], svd_u_tr, y_tr[tr_pos], X_va_tab.iloc[va_pos], svd_u_va, y_va[va_pos], cat_cols, seed=primary_seed)
        e3_rows.append({"variant": "M4_unmasked_leak_control_slice", "MAE_log": float(np.mean(np.abs(m4_e3_u.predict(X_te_tab.iloc[te_pos], svd_u_te) - y_te[te_pos]))), "n_train": e3_tr_n, "n_test": e3_te_n})

        # Same-slice M5 masked (reuses cached formatted PhoBERT embeddings)
        m5_e3_m = M5PhoBERTCatBoostModel(pca_components=best_pca_k, iterations=int(cb_cfg["iterations"]), early_stopping_rounds=int(cb_cfg["early_stopping_rounds"]), depth=int(m1_best_cfg["depth"]), learning_rate=float(m1_best_cfg["learning_rate"]), loss_function=str(m1_best_cfg["loss_function"]))
        m5_e3_m.fit(X_tr_tab.iloc[tr_pos], emb_tr[tr_pos], y_tr[tr_pos], X_va_tab.iloc[va_pos], emb_va[va_pos], y_va[va_pos], cat_cols, seed=primary_seed)
        e3_rows.append({"variant": "M5_masked_slice", "MAE_log": float(np.mean(np.abs(m5_e3_m.predict(X_te_tab.iloc[te_pos], emb_te[te_pos]) - y_te[te_pos]))), "n_train": e3_tr_n, "n_test": e3_te_n})

        # Same-slice M5 title-only & M5 unmasked leak control (using identical format_phobert_input wrapper)
        e3_embedder = PhoBERTEmbedder(
            model_name=ph_cfg["model_name"],
            revision=ph_cfg["revision"],
            max_length=int(ph_cfg["max_length"]),
            batch_size=int(ph_cfg["batch_size"]),
            fp16=bool(ph_cfg["fp16"]),
            normalize_l2=bool(ph_cfg["normalize_l2"]),
            device=cfg.get("runtime", {}).get("device", "auto"),
        )
        title_tr_fmt = [format_phobert_input(pt, pr, dt, nm, "") for pt, pr, dt, nm in zip(tr_e3["property_type_name"], tr_e3["province_norm"], tr_e3["district_norm"], tr_e3["name_masked"])]
        title_va_fmt = [format_phobert_input(pt, pr, dt, nm, "") for pt, pr, dt, nm in zip(va_e3["property_type_name"], va_e3["province_norm"], va_e3["district_norm"], va_e3["name_masked"])]
        title_te_fmt = [format_phobert_input(pt, pr, dt, nm, "") for pt, pr, dt, nm in zip(te_e3["property_type_name"], te_e3["province_norm"], te_e3["district_norm"], te_e3["name_masked"])]
        emb_title_tr, _ = e3_embedder.embed_texts(title_tr_fmt, mask_token=cfg["project"]["mask_token"])
        emb_title_va, _ = e3_embedder.embed_texts(title_va_fmt, mask_token=cfg["project"]["mask_token"])
        emb_title_te, _ = e3_embedder.embed_texts(title_te_fmt, mask_token=cfg["project"]["mask_token"])
        m5_title = M5PhoBERTCatBoostModel(pca_components=best_pca_k, iterations=int(cb_cfg["iterations"]), early_stopping_rounds=int(cb_cfg["early_stopping_rounds"]), depth=int(m1_best_cfg["depth"]), learning_rate=float(m1_best_cfg["learning_rate"]), loss_function=str(m1_best_cfg["loss_function"]))
        m5_title.fit(X_tr_tab.iloc[tr_pos], emb_title_tr, y_tr[tr_pos], X_va_tab.iloc[va_pos], emb_title_va, y_va[va_pos], cat_cols, seed=primary_seed)
        e3_rows.append({"variant": "M5_masked_title_only_slice", "MAE_log": float(np.mean(np.abs(m5_title.predict(X_te_tab.iloc[te_pos], emb_title_te) - y_te[te_pos]))), "n_train": e3_tr_n, "n_test": e3_te_n})

        unm_tr_fmt = [format_phobert_input(pt, pr, dt, nm, dm) for pt, pr, dt, nm, dm in zip(tr_e3["property_type_name"], tr_e3["province_norm"], tr_e3["district_norm"], tr_e3["name_norm"], tr_e3["description_norm"])]
        unm_va_fmt = [format_phobert_input(pt, pr, dt, nm, dm) for pt, pr, dt, nm, dm in zip(va_e3["property_type_name"], va_e3["province_norm"], va_e3["district_norm"], va_e3["name_norm"], va_e3["description_norm"])]
        unm_te_fmt = [format_phobert_input(pt, pr, dt, nm, dm) for pt, pr, dt, nm, dm in zip(te_e3["property_type_name"], te_e3["province_norm"], te_e3["district_norm"], te_e3["name_norm"], te_e3["description_norm"])]
        emb_unm_tr, _ = e3_embedder.embed_texts(unm_tr_fmt, mask_token=cfg["project"]["mask_token"])
        emb_unm_va, _ = e3_embedder.embed_texts(unm_va_fmt, mask_token=cfg["project"]["mask_token"])
        emb_unm_te, _ = e3_embedder.embed_texts(unm_te_fmt, mask_token=cfg["project"]["mask_token"])
        m5_unm = M5PhoBERTCatBoostModel(pca_components=best_pca_k, iterations=int(cb_cfg["iterations"]), early_stopping_rounds=int(cb_cfg["early_stopping_rounds"]), depth=int(m1_best_cfg["depth"]), learning_rate=float(m1_best_cfg["learning_rate"]), loss_function=str(m1_best_cfg["loss_function"]))
        m5_unm.fit(X_tr_tab.iloc[tr_pos], emb_unm_tr, y_tr[tr_pos], X_va_tab.iloc[va_pos], emb_unm_va, y_va[va_pos], cat_cols, seed=primary_seed)
        e3_rows.append({"variant": "M5_unmasked_leak_control_slice", "MAE_log": float(np.mean(np.abs(m5_unm.predict(X_te_tab.iloc[te_pos], emb_unm_te) - y_te[te_pos]))), "n_train": e3_tr_n, "n_test": e3_te_n})

    pd.DataFrame(e3_rows).to_csv(metrics_dir / "e3_ablation.csv", index=False)

    # Bootstrap CI between M1 and M5 on Scenario A test
    boot_ci = cluster_bootstrap_mae_diff(pred_dfs_test["M5"], pred_dfs_test["M1"], clean_df, n_boot=1000, seed=primary_seed)
    pd.DataFrame([{"comparison": "M5_minus_M1_A_test", **boot_ci}]).to_csv(metrics_dir / "bootstrap_ci.csv", index=False)

    # Conformal C0-C3 on the best validation model among those with quantile heads (M1/M4/M5)
    if winner_id == "M1":
        win_cal, win_test = m1_cal_df, m1_test_df
        win_model_obj = m1
    elif winner_id == "M4":
        win_cal = pd.read_parquet(preds_dir / f"A_M4_cal_seed{primary_seed}.parquet")
        win_test = pd.read_parquet(preds_dir / f"A_M4_test_seed{primary_seed}.parquet")
        win_model_obj = best_m4
    elif winner_id == "M5":
        win_cal, win_test = m5_cal_df, m5_test_df
        win_model_obj = best_m5
    else:
        raise RuntimeError(f"No conformal-capable model for winner {winner_id!r}")
    win_kind = winner_id

    if win_cal is None or win_test is None:
        raise RuntimeError(f"Missing cal/test predictions for conformal model {win_kind}")
    win_cal_meta = win_cal.merge(ca[["row_id", "property_type_name"]], on="row_id", how="left")
    win_test_meta = win_test.merge(te[["row_id", "property_type_name", "text_status", "area_mismatch_flag", "text_masked"]], on="row_id", how="left")

    conformal_rows: list[dict[str, Any]] = []
    alpha_pri = float(cfg["conformal"]["alpha_primary"])
    alpha_sec = float(cfg["conformal"]["alpha_secondary"])
    min_g = int(cfg["conformal"]["c3_min_group_samples"])

    c2_test_out = None
    for m_name in ["C0", "C1", "C2", "C3"]:
        c_out = fit_apply_conformal(win_cal_meta, win_test_meta, method=m_name, alpha=alpha_pri, min_group_size=min_g)
        ev_c = evaluate_predictions_df(c_out, clean_df, train_df=tr, alpha=alpha_pri, min_eval_group_size=int(cfg["conformal"]["min_eval_group_size"]))
        conformal_rows.append({"scenario": "A", "model_id": win_kind, "alpha": alpha_pri, "method": m_name, **ev_c["overall"]})
        if m_name == "C2":
            c2_test_out = c_out
            ev_c["subgroups"].to_csv(metrics_dir / "conformal_subgroups_a_c2.csv", index=False)
            c_out.to_parquet(preds_dir / f"A_{win_kind}_test_C2.parquet", index=False)

    # Secondary alpha=0.20 check for C2
    c2_80 = fit_apply_conformal(win_cal_meta, win_test_meta, method="C2", alpha=alpha_sec, min_group_size=min_g)
    ev_c80 = evaluate_predictions_df(c2_80, clean_df, train_df=tr, alpha=alpha_sec, min_eval_group_size=int(cfg["conformal"]["min_eval_group_size"]))
    conformal_rows.append({"scenario": "A", "model_id": win_kind, "alpha": alpha_sec, "method": "C2_alpha0.20", **ev_c80["overall"]})

    # Stale calibration control: ONE model (M1 trained on the early train months), calibrated twice and evaluated on test:
    # stale = last train months (cfg stale_cal_months), fresh = the real cal split. Same model => differences = calibration only.
    stale_months = [str(m) for m in cfg["splits"]["scenario_a"]["stale_cal_months"]]
    late_mask = tr["month"].isin(stale_months).to_numpy()
    early_mask = ~late_mask
    if early_mask.sum() >= 100 and late_mask.sum() >= 50:
        m1_stale = M1CatBoostModel(
            iterations=int(cb_cfg["iterations"]),
            early_stopping_rounds=int(cb_cfg["early_stopping_rounds"]),
            loss_candidates=[m1_best_cfg["loss_function"]],
            depth_candidates=[int(m1_best_cfg["depth"])],
            lr_candidates=[float(m1_best_cfg["learning_rate"])],
        )
        m1_stale.fit_with_config(X_tr_tab.iloc[early_mask], y_tr[early_mask], X_va_tab, y_va, cat_cols, m1_best_cfg, seed=primary_seed)
        m1_stale.fit_quantiles(X_tr_tab.iloc[early_mask], y_tr[early_mask], X_va_tab, y_va, cat_cols, quantiles=tuple(cb_cfg["quantiles"]), seed=primary_seed)
        stale_q_lo, stale_q_hi = m1_stale.predict_quantiles(X_tr_tab.iloc[late_mask])
        stale_cal_df = pd.DataFrame({
            "row_id": tr.iloc[late_mask]["row_id"].tolist(),
            "y_true": y_tr[late_mask],
            "y_pred": m1_stale.predict(X_tr_tab.iloc[late_mask]),
            "q_lo": stale_q_lo,
            "q_hi": stale_q_hi,
            "property_type_name": tr.iloc[late_mask]["property_type_name"].tolist(),
        })
        te_stale_q_lo, te_stale_q_hi = m1_stale.predict_quantiles(X_te_tab)
        stale_test_df = pd.DataFrame({
            "row_id": te["row_id"].tolist(),
            "y_true": y_te,
            "y_pred": m1_stale.predict(X_te_tab),
            "q_lo": te_stale_q_lo,
            "q_hi": te_stale_q_hi,
            "property_type_name": te["property_type_name"].tolist(),
        })
        fresh_q_lo, fresh_q_hi = m1_stale.predict_quantiles(X_ca_tab)
        fresh_cal_df = pd.DataFrame({
            "row_id": ca["row_id"].tolist(),
            "y_true": y_ca,
            "y_pred": m1_stale.predict(X_ca_tab),
            "q_lo": fresh_q_lo,
            "q_hi": fresh_q_hi,
            "property_type_name": ca["property_type_name"].tolist(),
        })
        for label, cal_src in [("C2_stale_cal", stale_cal_df), ("C2_fresh_cal", fresh_cal_df)]:
            c2_ctl = fit_apply_conformal(cal_src, stale_test_df, method="C2", alpha=alpha_pri, min_group_size=min_g)
            ev_ctl = evaluate_predictions_df(c2_ctl, clean_df, train_df=tr, alpha=alpha_pri, min_eval_group_size=int(cfg["conformal"]["min_eval_group_size"]))
            conformal_rows.append({"scenario": "A", "model_id": "M1_stale_train", "alpha": alpha_pri, "method": label, **ev_ctl["overall"]})

    # E6 Flagging, audit export & error analysis
    print("[Scenario A] Running Conformal/Flagging/Audit export & saving bundle...", flush=True)
    assert c2_test_out is not None
    flagged_te = flag_mispriced_listings(c2_test_out)
    flag_audit_stats = export_flag_audit_samples(
        flagged_clean_df=flagged_te,
        meta_df=clean_df,
        noisy_df=noisy_df,
        out_path=audits_dir / "audit_flags_300_plus_300.csv",
        n_per_class=int(cfg["audits"]["flag_audit_size_per_class"]),
        seed=primary_seed,
    )
    err_buckets = sample_error_buckets(c2_test_out, clean_df, n_per_bucket=int(cfg["audits"]["error_bucket_samples"]), seed=primary_seed)
    err_buckets.to_csv(audits_dir / "error_buckets_45.csv", index=False, encoding="utf-8-sig")
    res_kw = analyze_residual_keywords(c2_test_out, clean_df)
    res_kw.to_csv(metrics_dir / "residual_keyword_corr.csv", index=False)

    # Save deployable bundle from Scenario A winner
    bundle_dir = paths["models_dir"] / "bundle_a"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(tab_ext, bundle_dir / "tabular_extractor.joblib")
    if win_kind == "M5":
        joblib.dump(best_m5.scaler, bundle_dir / "emb_scaler.joblib")
        joblib.dump(best_m5.pca, bundle_dir / "emb_pca.joblib")
    elif win_kind == "M4":
        joblib.dump(best_m4_ext, bundle_dir / "tfidf_svd.joblib")
    win_model_obj.point_model.save_model(str(bundle_dir / "point_model.cbm"))
    win_model_obj.q_lo_model.save_model(str(bundle_dir / "q_lo_model.cbm"))
    win_model_obj.q_hi_model.save_model(str(bundle_dir / "q_hi_model.cbm"))

    district_counts = tr["district_key"].value_counts().to_dict()
    (bundle_dir / "district_train_counts.json").write_text(json.dumps(district_counts, ensure_ascii=False), encoding="utf-8")

    # Retrieval index strictly from clean train with PII redacted
    comps_cols = [
        c for c in ["row_id", "property_type_name", "province_norm", "district_norm", "ward_norm", "area", "floor_count", "bedroom_count", "price", "name_masked"]
        if c in tr.columns
    ]
    comps = tr[comps_cols].copy()
    comps["name_masked"] = [redact_pii(t) for t in comps["name_masked"]]
    comps.to_parquet(bundle_dir / "train_comps.parquet", index=False)

    c2_q_val = float(c2_test_out["conformal_q"].iloc[0])
    bundle_meta = {
        "model_id": win_kind,
        "model_version": f"{cfg['project']['rules_version']}_{win_kind}_A",
        "uses_phobert": win_kind == "M5",
        "uses_svd": win_kind == "M4",
        "c2_q": c2_q_val,
        "phobert": cfg["phobert"],
        "hf_home": cfg.get("runtime", {}).get("hf_home"),
        "mask_token": cfg["project"]["mask_token"],
        "aliases_path": str(paths["aliases_path"].resolve()),
        "tabular_bounds": cfg["cleaning"]["tabular_bounds"],
        "reconcile_cfg": {k: v for k, v in cfg["cleaning"].items() if k not in {"tabular_bounds", "train_ppm2_quantiles"}},
    }
    (bundle_dir / "bundle_meta.json").write_text(json.dumps(bundle_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # Measure inference latency on a 500-row slice
    n_lat = min(500, len(X_te_tab))
    if win_kind == "M5":
        lat_info = _measure_latency(lambda: best_m5.predict(X_te_tab.iloc[:n_lat], emb_te[:n_lat]), n_lat)
    elif win_kind == "M4":
        lat_info = _measure_latency(lambda: best_m4.predict(X_te_tab.iloc[:n_lat], te_svd_b[:n_lat]), n_lat)
    else:
        lat_info = _measure_latency(lambda: m1.predict(X_te_tab.iloc[:n_lat]), n_lat)
    lat_info["batch_predict_only"] = True
    try:
        lat_info.update(_measure_single_row_latency(bundle_dir, te, seed=primary_seed))
    except Exception as exc:  # latency probe must never kill the run
        lat_info["single_row_error"] = f"{type(exc).__name__}: {exc}"

    return {
        "winner_id": val_winner_id,
        "conformal_model_id": win_kind,
        "val_mae_by_model": val_mae_by_model,
        "metrics": all_metrics,
        "tuning_logs": tuning_logs,
        "conformal_rows": conformal_rows,
        "flag_audit_stats": flag_audit_stats,
        "latency": lat_info,
        "m1_best_cfg": m1_best_cfg,
        "best_pca_k": best_pca_k,
        "best_svd_k": best_svd_k,
    }


def _run_scenario_b(
    cfg: dict[str, Any],
    paths: dict[str, Path],
    clean_df: pd.DataFrame,
    man_b: pd.DataFrame,
    res_a: dict[str, Any],
) -> dict[str, Any]:
    preds_dir = paths["outputs_dir"] / "predictions"
    splits = _slice_by_split(clean_df, man_b)
    tr, va, ca, te = splits["train"], splits["val"], splits["cal"], splits["test"]
    y_tr, y_va = tr["y_true"].to_numpy(), va["y_true"].to_numpy()
    primary_seed = int(cfg["project"]["primary_seed"])
    cb_cfg = cfg["models"]["catboost"]
    cache_dir = paths["processed_dir"] / "embeddings"
    m1_best_cfg = res_a["m1_best_cfg"]

    tab_ext = TabularFeatureExtractor(min_freq=int(cfg["features"]["rare_category_min_freq"]))
    X_tr_tab, cat_cols = tab_ext.fit_transform(tr)
    X_va_tab, _ = tab_ext.transform(va)
    X_ca_tab, _ = tab_ext.transform(ca)
    X_te_tab, _ = tab_ext.transform(te)

    emb_tr = load_embeddings_for_row_ids(tr["row_id"].tolist(), cache_dir, cfg["project"]["text_version"])
    emb_va = load_embeddings_for_row_ids(va["row_id"].tolist(), cache_dir, cfg["project"]["text_version"])
    emb_ca = load_embeddings_for_row_ids(ca["row_id"].tolist(), cache_dir, cfg["project"]["text_version"])
    emb_te = load_embeddings_for_row_ids(te["row_id"].tolist(), cache_dir, cfg["project"]["text_version"])

    metrics_b: list[dict[str, Any]] = []
    conformal_b: list[dict[str, Any]] = []

    def _rec(model_id: str, sp_name: str, sp_df: pd.DataFrame, yp: np.ndarray, q_lo=None, q_hi=None, train_sec: float = 0.0) -> pd.DataFrame:
        pdf = make_prediction_df(sp_df["row_id"].tolist(), sp_name, "B", model_id, primary_seed, sp_df["y_true"].to_numpy(), yp, q_lo=q_lo, q_hi=q_hi)
        save_predictions(pdf, preds_dir / f"B_{model_id}_{sp_name}_seed{primary_seed}.parquet")
        ev = evaluate_predictions_df(pdf, clean_df, train_df=tr, alpha=float(cfg["conformal"]["alpha_primary"]), min_eval_group_size=int(cfg["conformal"]["min_eval_group_size"]))
        metrics_b.append({"scenario": "B", "model_id": model_id, "seed": primary_seed, "split": sp_name, "train_time_sec": train_sec, **ev["overall"]})
        return pdf

    # M0 on B
    print(f"[Scenario B] Running unseen-district M0/M1/M4/M5 + Conformal (train={len(tr):,}, test={len(te):,})...", flush=True)
    val_mae_b: dict[str, float] = {}
    t0 = time.perf_counter()
    m0 = M0GroupMedianModel(min_group_samples=int(cfg["models"]["m0_min_group_samples"])).fit(tr)
    t_m0 = time.perf_counter() - t0
    for sp_name, sp_df in [("val", va), ("cal", ca), ("test", te)]:
        yp_m0 = m0.predict(sp_df)
        _rec("M0", sp_name, sp_df, yp_m0, train_sec=t_m0)
        if sp_name == "val":
            val_mae_b["M0"] = float(np.mean(np.abs(yp_m0 - y_va)))

    # M1 on B
    m1_b = M1CatBoostModel(iterations=int(cb_cfg["iterations"]), early_stopping_rounds=int(cb_cfg["early_stopping_rounds"]))
    t_m1 = m1_b.fit_with_config(X_tr_tab, y_tr, X_va_tab, y_va, cat_cols, m1_best_cfg, seed=primary_seed)
    m1_b.fit_quantiles(X_tr_tab, y_tr, X_va_tab, y_va, cat_cols, quantiles=tuple(cb_cfg["quantiles"]), seed=primary_seed)
    m1_cal = m1_test = None
    for sp_name, sp_df, X_sp in [("val", va, X_va_tab), ("cal", ca, X_ca_tab), ("test", te, X_te_tab)]:
        ql, qh = m1_b.predict_quantiles(X_sp)
        yp_m1 = m1_b.predict(X_sp)
        pdf = _rec("M1", sp_name, sp_df, yp_m1, q_lo=ql, q_hi=qh, train_sec=t_m1)
        if sp_name == "val":
            val_mae_b["M1"] = float(np.mean(np.abs(yp_m1 - y_va)))
        if sp_name == "cal":
            m1_cal = pdf
        if sp_name == "test":
            m1_test = pdf

    # M4 on B (TF-IDF SVD strictly fit on Scenario B train)
    feat_cfg = cfg["features"]
    tfidf_b = TfidfSvdExtractor(
        max_features=int(feat_cfg["tfidf_max_features"]),
        min_df=int(feat_cfg["tfidf_min_df"]),
        ngram_range=(1, 2),
        svd_components=int(res_a["best_svd_k"]),
        seed=primary_seed,
    ).fit(tr["text_masked"].tolist(), fit_svd=True)
    b_svd_tr = tfidf_b.transform_svd(tr["text_masked"].tolist())
    b_svd_va = tfidf_b.transform_svd(va["text_masked"].tolist())
    b_svd_ca = tfidf_b.transform_svd(ca["text_masked"].tolist())
    b_svd_te = tfidf_b.transform_svd(te["text_masked"].tolist())
    m4_b = M4TfidfCatBoostModel(
        iterations=int(cb_cfg["iterations"]),
        early_stopping_rounds=int(cb_cfg["early_stopping_rounds"]),
        depth=int(m1_best_cfg["depth"]),
        learning_rate=float(m1_best_cfg["learning_rate"]),
        loss_function=str(m1_best_cfg["loss_function"]),
    )
    t_m4 = m4_b.fit(X_tr_tab, b_svd_tr, y_tr, X_va_tab, b_svd_va, y_va, cat_cols, seed=primary_seed)
    m4_b.fit_quantiles(X_tr_tab, b_svd_tr, y_tr, X_va_tab, b_svd_va, y_va, cat_cols, quantiles=tuple(cb_cfg["quantiles"]), seed=primary_seed)
    m4_cal = m4_test = None
    for sp_name, sp_df, X_sp, svd_sp in [("val", va, X_va_tab, b_svd_va), ("cal", ca, X_ca_tab, b_svd_ca), ("test", te, X_te_tab, b_svd_te)]:
        ql, qh = m4_b.predict_quantiles(X_sp, svd_sp)
        yp_m4 = m4_b.predict(X_sp, svd_sp)
        pdf = _rec("M4", sp_name, sp_df, yp_m4, q_lo=ql, q_hi=qh, train_sec=t_m4)
        if sp_name == "val":
            val_mae_b["M4"] = float(np.mean(np.abs(yp_m4 - y_va)))
        if sp_name == "cal":
            m4_cal = pdf
        if sp_name == "test":
            m4_test = pdf

    # M5 on B
    m5_b = M5PhoBERTCatBoostModel(
        pca_components=int(res_a["best_pca_k"]),
        iterations=int(cb_cfg["iterations"]),
        early_stopping_rounds=int(cb_cfg["early_stopping_rounds"]),
        depth=int(m1_best_cfg["depth"]),
        learning_rate=float(m1_best_cfg["learning_rate"]),
        loss_function=str(m1_best_cfg["loss_function"]),
    )
    t_m5 = m5_b.fit(X_tr_tab, emb_tr, y_tr, X_va_tab, emb_va, y_va, cat_cols, seed=primary_seed)
    m5_b.fit_quantiles(X_tr_tab, emb_tr, y_tr, X_va_tab, emb_va, y_va, cat_cols, quantiles=tuple(cb_cfg["quantiles"]), seed=primary_seed)
    m5_cal = m5_test = None
    for sp_name, sp_df, X_sp, emb_sp in [("val", va, X_va_tab, emb_va), ("cal", ca, X_ca_tab, emb_ca), ("test", te, X_te_tab, emb_te)]:
        ql, qh = m5_b.predict_quantiles(X_sp, emb_sp)
        yp_m5 = m5_b.predict(X_sp, emb_sp)
        pdf = _rec("M5", sp_name, sp_df, yp_m5, q_lo=ql, q_hi=qh, train_sec=t_m5)
        if sp_name == "val":
            val_mae_b["M5"] = float(np.mean(np.abs(yp_m5 - y_va)))
        if sp_name == "cal":
            m5_cal = pdf
        if sp_name == "test":
            m5_test = pdf

    # Report the true validation winner (may be M0); pick the conformal model among M1/M4/M5 only.
    val_winner_b = min(val_mae_b, key=lambda k: val_mae_b[k])
    win_b_kind = _pick_conformal_model(val_mae_b)
    print(f"[Scenario B] Validation winner: {val_winner_b}; conformal model: {win_b_kind} ({val_mae_b})", flush=True)
    conformal_sources = {"M1": (m1_cal, m1_test), "M4": (m4_cal, m4_test), "M5": (m5_cal, m5_test)}
    sel_cal, sel_test = conformal_sources[win_b_kind]
    if sel_cal is None or sel_test is None:
        raise RuntimeError(f"Missing cal/test predictions for Scenario B conformal model {win_b_kind}")
    win_cal_b = sel_cal.merge(ca[["row_id", "property_type_name"]], on="row_id", how="left")
    win_test_b = sel_test.merge(te[["row_id", "property_type_name"]], on="row_id", how="left")
    alpha_pri = float(cfg["conformal"]["alpha_primary"])
    min_g = int(cfg["conformal"]["c3_min_group_samples"])

    for m_name in ["C0", "C1", "C2", "C3"]:
        c_out = fit_apply_conformal(win_cal_b, win_test_b, method=m_name, alpha=alpha_pri, min_group_size=min_g)
        ev_c = evaluate_predictions_df(c_out, clean_df, train_df=tr, alpha=alpha_pri, min_eval_group_size=int(cfg["conformal"]["min_eval_group_size"]))
        conformal_b.append({"scenario": "B", "model_id": win_b_kind, "alpha": alpha_pri, "method": m_name, **ev_c["overall"]})

    return {
        "winner_id": val_winner_b,
        "conformal_model_id": win_b_kind,
        "val_mae_by_model": val_mae_b,
        "metrics": metrics_b,
        "conformal_rows": conformal_b,
        "n_districts": {sp: int(d["district_key"].nunique()) for sp, d in splits.items()},
    }


def run_train_eval(cfg: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    """Run leakage checks, Scenario A (E1-E3, E5-E6), and Scenario B (E4-E5)."""
    set_global_seed(int(cfg["project"]["primary_seed"]))
    proc_dir = paths["processed_dir"]
    splits_dir = proc_dir / "splits"
    metrics_dir = paths["outputs_dir"] / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    man_a = pd.read_parquet(splits_dir / "scenario_a_manifest.parquet")
    man_b = pd.read_parquet(splits_dir / "scenario_b_manifest.parquet")
    union_ids = set(man_a["row_id"].astype(str)).union(set(man_b["row_id"].astype(str)))

    import pyarrow.parquet as pq

    clean_parts: list[pd.DataFrame] = []
    pf_clean = pq.ParquetFile(proc_dir / "clean.parquet")
    for batch in pf_clean.iter_batches(batch_size=100_000):
        bdf = batch.to_pandas()
        sub = bdf[bdf["row_id"].astype(str).isin(union_ids)]
        if len(sub) > 0:
            clean_parts.append(sub)
    clean_df = pd.concat(clean_parts, axis=0, ignore_index=True) if clean_parts else pd.DataFrame()

    noisy_path = proc_dir / "noisy_flagged.parquet"
    noisy_df = pd.read_parquet(noisy_path) if noisy_path.exists() else pd.DataFrame()

    # Strict leakage checks before training (tolerance 0 unless the config explicitly relaxes it)
    leak_tol = float(cfg.get("leakage", {}).get("max_leaked_row_rate", 0.0))
    leak_a =verify_split_leakage(man_a, text_df=clean_df[clean_df["row_id"].isin(man_a["row_id"])], scenario="A", strict_text=True, max_leak_rate=leak_tol)
    leak_b = verify_split_leakage(man_b, text_df=clean_df[clean_df["row_id"].isin(man_b["row_id"])], scenario="B", strict_text=True, max_leak_rate=leak_tol)
    (metrics_dir / "leakage_verification.json").write_text(json.dumps({"A": leak_a, "B": leak_b}, indent=2, ensure_ascii=False), encoding="utf-8")

    res_a = _run_scenario_a(cfg, paths, clean_df, man_a, noisy_df)
    res_b = _run_scenario_b(cfg, paths, clean_df, man_b, res_a)

    all_mc = pd.DataFrame(res_a["metrics"] + res_b["metrics"])
    all_mc.to_csv(metrics_dir / "model_comparison.csv", index=False)

    # Multi-seed summary (mean ± std across seeds on Scenario A test)
    seed_df = all_mc[(all_mc["scenario"] == "A") & (all_mc["split"] == "test")].groupby("model_id").agg(
        n_seeds=("seed", "count"),
        mae_mean=("MAE_log", "mean"),
        mae_std=("MAE_log", "std"),
        within10_mean=("within_10_pct", "mean"),
        within10_std=("within_10_pct", "std"),
    ).reset_index()
    seed_df.to_csv(metrics_dir / "multi_seed_summary.csv", index=False)

    pd.DataFrame(res_a["tuning_logs"]).to_csv(metrics_dir / "tuning_logs.csv", index=False)
    pd.DataFrame(res_a["conformal_rows"] + res_b["conformal_rows"]).to_csv(metrics_dir / "conformal_summary.csv", index=False)

    summary = {
        "winner_a": res_a["winner_id"],
        "winner_b": res_b["winner_id"],
        "conformal_model_a": res_a["conformal_model_id"],
        "conformal_model_b": res_b["conformal_model_id"],
        "n_districts_b": res_b["n_districts"],
        "val_mae_a": res_a["val_mae_by_model"],
        "val_mae_b": res_b["val_mae_by_model"],
        "leakage": {
            "A": leak_a["passed"],
            "B": leak_b["passed"],
            "leaks_a": leak_a["leaked_money_rows"],
            "leaks_b": leak_b["leaked_money_rows"],
        },
        "flag_audit": res_a["flag_audit_stats"],
        "latency": res_a["latency"],
    }
    (metrics_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="TV2-TV4 training & evaluation pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = resolve_project_paths(cfg)
    summary = run_train_eval(cfg, paths)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
