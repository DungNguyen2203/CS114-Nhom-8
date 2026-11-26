import pytest

from src.pipelines.train_eval import _pick_conformal_model


def test_conformal_model_is_best_among_quantile_capable_not_silent_m5():
    # M0 has the lowest validation MAE but no quantile heads: must not be picked, nor silently replaced by M5.
    val = {"M0": 0.55, "M1": 0.60, "M4": 0.566, "M5": 0.5658}
    assert min(val, key=val.get) == "M0"
    assert _pick_conformal_model(val) == "M5"
    assert _pick_conformal_model({"M0": 0.5, "M1": 0.6, "M4": 0.55, "M5": 0.58}) == "M4"
    with pytest.raises(RuntimeError):
        _pick_conformal_model({"M0": 0.5, "M2": 0.6})
