import numpy as np
import pandas as pd
import pytest

from src.eval.leakage_checks import verify_split_leakage
from src.eval.metrics import compute_point_metrics, compute_interval_metrics, evaluate_predictions_df
from src.eval.conformal import fit_apply_conformal
from src.eval.flagger import export_flag_audit_samples, flag_mispriced_listings
from src.eval.error_analysis import sample_error_buckets, analyze_residual_keywords


def test_verify_split_leakage_detects_overlaps():
    ok = pd.DataFrame(
        {
            "row_id": ["00_1", "00_2", "00_3", "00_4"],
            "split": ["train", "val", "cal", "test"],
            "dup_group_id": ["g1", "g2", "g3", "g4"],
            "district_key": ["HN|CG", "HN|HM", "HCM|Q1", "HCM|Q7"],
        }
    )
    res = verify_split_leakage(ok, scenario="B")
    assert res["passed"] is True

    bad_dup = ok.copy()
    bad_dup.loc[3, "dup_group_id"] = "g1"
    with pytest.raises(AssertionError):
        verify_split_leakage(bad_dup, scenario="A")

    bad_dist = ok.copy()
    bad_dist.loc[3, "district_key"] = "HN|CG"
    with pytest.raises(AssertionError):
        verify_split_leakage(bad_dist, scenario="B")


def test_point_and_interval_metrics_exact():
    y_true = np.log([1_000_000_000.0, 2_000_000_000.0])
    y_pred = np.log([1_090_000_000.0, 2_000_000_000.0])
    pm = compute_point_metrics(y_true, y_pred)
    assert pm["within_10_pct"] == 100.0
    assert pm["MAE_log"] > 0

    # Test crossing rate and interval score
    q_lo = np.array([20.5, 22.0])  # second sample crossed: q_lo > q_hi
    q_hi = np.array([21.5, 21.0])
    im = compute_interval_metrics(y_true, q_lo, q_hi, alpha=0.10)
    assert im["crossing_rate_raw"] == 0.5
    assert 0.0 <= im["coverage"] <= 1.0


def test_conformal_c0_c1_c2_c3_and_small_group_fallback():
    rng = np.random.default_rng(123)
    n_cal, n_test = 600, 300
    y_cal = rng.normal(22.5, 0.5, size=n_cal)
    pred_cal = y_cal + rng.normal(0.0, 0.2, size=n_cal)
    # Deliberately include a few crossed quantiles on cal/test
    q_lo_cal = pred_cal - 0.25
    q_hi_cal = pred_cal + 0.25
    q_lo_cal[0], q_hi_cal[0] = q_hi_cal[0], q_lo_cal[0]

    y_test = rng.normal(22.5, 0.5, size=n_test)
    pred_test = y_test + rng.normal(0.0, 0.2, size=n_test)
    q_lo_test = pred_test - 0.25
    q_hi_test = pred_test + 0.25

    cal_df = pd.DataFrame(
        {
            "row_id": [f"cal_{i}" for i in range(n_cal)],
            "y_true": y_cal,
            "y_pred": pred_cal,
            "q_lo": q_lo_cal,
            "q_hi": q_hi_cal,
            "property_type_name": ["Nhà"] * 520 + ["Shophouse"] * 80,
        }
    )
    test_df = pd.DataFrame(
        {
            "row_id": [f"test_{i}" for i in range(n_test)],
            "y_true": y_test,
            "y_pred": pred_test,
            "q_lo": q_lo_test,
            "q_hi": q_hi_test,
            "property_type_name": ["Nhà"] * 250 + ["Shophouse"] * 50,
        }
    )

    out_c0 = fit_apply_conformal(cal_df, test_df, method="C0", alpha=0.10)
    out_c1 = fit_apply_conformal(cal_df, test_df, method="C1", alpha=0.10)
    out_c2 = fit_apply_conformal(cal_df, test_df, method="C2", alpha=0.10)
    out_c3 = fit_apply_conformal(cal_df, test_df, method="C3", alpha=0.10, min_group_size=500)

    for out in (out_c0, out_c1, out_c2, out_c3):
        assert (out["q_hi"] >= out["q_lo"]).all()
        assert len(out) == n_test

    # C2 and C3 coverage should be around 0.90 on i.i.d. synthetic normal
    cov_c2 = float(((out_c2["y_true"] >= out_c2["q_lo"]) & (out_c2["y_true"] <= out_c2["q_hi"])).mean())
    cov_c3 = float(((out_c3["y_true"] >= out_c3["q_lo"]) & (out_c3["y_true"] <= out_c3["q_hi"])).mean())
    assert 0.82 <= cov_c2 <= 0.98
    assert 0.82 <= cov_c3 <= 0.98
    # Small group Shophouse (<500) must fall back cleanly to global C2 Q
    assert (out_c3["c3_fallback_used"] == True).any()


def test_flagger_and_error_analysis():
    eval_df = pd.DataFrame(
        {
            "row_id": ["00_1", "00_2", "00_3"],
            "y_true": [21.0, 23.0, 25.0],
            "y_pred": [23.0, 23.05, 22.0],
            "q_lo": [22.5, 22.5, 21.5],
            "q_hi": [23.5, 23.5, 22.5],
            "text_status": ["clean_consistent", "clean_consistent", "noisy_order_of_magnitude"],
            "area_mismatch_flag": [False, False, True],
            "text_masked": ["bán gấp MASK_GIA", "nhà đẹp chính chủ MASK_GIA", "ngộp bank MASK_GIA"],
        }
    )
    flagged = flag_mispriced_listings(eval_df)
    assert flagged.loc[0, "flag_g1"] == "below_interval"
    assert flagged.loc[1, "flag_g1"] == "in_interval"
    assert flagged.loc[2, "flag_g1"] == "above_interval"
    assert flagged.loc[2, "flag_g2"] == "order_of_magnitude_mismatch"
    assert bool(flagged.loc[0, "is_flagged"]) is True
    assert bool(flagged.loc[1, "is_flagged"]) is False

    corr = analyze_residual_keywords(eval_df, eval_df)
    assert "bán gấp" in set(corr["keyword"])


def test_export_flag_audit_samples_preserves_columns_pandas3(tmp_path):
    eval_df = pd.DataFrame(
        {
            "row_id": ["00_1", "00_2", "00_3", "00_4"],
            "y_true": [21.0, 23.0, 25.0, 22.8],
            "y_pred": [23.0, 23.05, 22.0, 22.9],
            "q_lo": [22.5, 22.5, 21.5, 22.3],
            "q_hi": [23.5, 23.5, 22.5, 23.4],
            "text_status": ["clean_consistent", "clean_consistent", "noisy_order_of_magnitude", "clean_consistent"],
            "area_mismatch_flag": [False, False, True, False],
        }
    )
    meta_df = pd.DataFrame(
        {
            "row_id": ["00_1", "00_2", "00_3", "00_4"],
            "property_type_name": ["Nhà", "Nhà", "Đất", "Căn hộ chung cư"],
            "province_norm": ["Hà Nội", "Hà Nội", "Hồ Chí Minh", "Đà Nẵng"],
            "district_norm": ["Cầu Giấy", "Đống Đa", "Quận 1", "Hải Châu"],
            "area": [50.0, 60.0, 100.0, 70.0],
            "price": [1.3e9, 9.7e9, 7.2e10, 8.0e9],
            "text_price_vnd": [1.3e9, 9.7e9, 7.2e9, 8.0e9],
            "text_status": ["clean_consistent", "clean_consistent", "noisy_order_of_magnitude", "clean_consistent"],
            "area_mismatch_flag": [False, False, True, False],
            "name_norm": ["Tin 1", "Tin 2", "Tin 3", "Tin 4"],
            "description_norm": ["Desc 1", "Desc 2", "Desc 3", "Desc 4"],
        }
    )
    flagged = flag_mispriced_listings(eval_df)
    out_csv = tmp_path / "audit_flags.csv"
    stats = export_flag_audit_samples(flagged, meta_df, noisy_df=None, out_path=out_csv, n_per_class=2, seed=42)
    assert out_csv.exists()
    assert stats["n_flagged_exported"] == 2
    assert stats["n_unflagged_exported"] == 2
    assert len(stats["subtype_counts"]) >= 1
    df_out = pd.read_csv(out_csv)
    assert df_out["audit_subtype"].notna().all()
    assert df_out["property_type_name"].notna().all()


def test_evaluate_predictions_df_preserves_subgroup_columns():
    pred_df = pd.DataFrame(
        {
            "row_id": ["00_1", "00_2", "00_3"],
            "y_true": [22.0, 22.5, 23.0],
            "y_pred": [22.1, 22.4, 22.9],
            "q_lo": [21.5, 21.9, 22.4],
            "q_hi": [22.6, 22.9, 23.5],
            "property_type_name": ["Nhà", "Đất", "Shophouse"],
        }
    )
    meta_df = pd.DataFrame(
        {
            "row_id": ["00_1", "00_2", "00_3"],
            "property_type_name": ["Nhà", "Đất", "Shophouse"],
            "province_norm": ["Hà Nội", "Hồ Chí Minh", "Đà Nẵng"],
            "district_key": ["HN|CG", "HCM|Q7", "DN|HC"],
            "month": ["2026-03", "2026-03", "2026-03"],
        }
    )
    res = evaluate_predictions_df(pred_df, meta_df, alpha=0.10, min_eval_group_size=1)
    dims = set(res["subgroups"]["group_dim"].tolist())
    assert "property_type_name" in dims
    assert "region_bucket" in dims

