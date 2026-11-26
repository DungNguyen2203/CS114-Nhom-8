import numpy as np
import pandas as pd
import pytest

from src.eval.conformal import finite_sample_quantile, fit_apply_conformal
from src.eval.flagger import flag_mispriced_listings
from src.eval.metrics import evaluate_predictions_df, cluster_bootstrap_mae_diff


@pytest.mark.parametrize("method", ["C2", "C3"])
def test_narrow_eval_interval_stays_ordered_when_calibration_scores_are_negative(method):
    cal = pd.DataFrame(dict(row_id=range(10), y_true=5., y_pred=5., q_lo=0., q_hi=10.,
                            property_type_name="Nhà"))
    test = cal.iloc[:1].assign(q_lo=4., q_hi=6., y_true=3.)
    out = fit_apply_conformal(cal, test, method)
    assert out.loc[0, ["q_lo", "q_hi", "conformal_q"]].tolist() == [4., 6., 0.]
    assert evaluate_predictions_df(out, out, train_df=cal)["overall"]["coverage"] == 0
    assert flag_mispriced_listings(out).loc[0, "flag_g1"] == "below_interval"


@pytest.mark.parametrize("scores,alpha", [([1.], .1), ([np.nan], .5), ([1.], 1), ([1.], 0)])
def test_finite_quantile_rejects_unavailable_rank_and_invalid_inputs(scores, alpha):
    with pytest.raises(ValueError):
        finite_sample_quantile(np.array(scores), alpha)


def test_finite_quantile_at_exact_decimal_rank():
    assert finite_sample_quantile(np.arange(1, 100), alpha=.29) == 71


def test_cluster_bootstrap_resamples_whole_districts():
    a = pd.DataFrame(dict(row_id=["a", "b", "c"], y_true=0., y_pred=[1., 1., 10.]))
    b = a.assign(y_pred=0.)
    meta = a.assign(district_key=["d", "d", "x"])
    assert cluster_bootstrap_mae_diff(a, b, meta, n_boot=1000, seed=42) == {
        "mae_diff_mean": 4., "ci_95_low": 1., "ci_95_high": 10.,
    }
