import numpy as np
from cwfc import metrics as M


def test_basic_metrics():
    y, p = np.array([1., 2, 3]), np.array([1., 3, 5])
    assert np.isclose(M.mae(y, p), 1.0) and np.isclose(M.rmse(y, p), np.sqrt(5 / 3))


def test_smape_handles_near_zero_denominator():
    assert M.smape([0.0, 0.0], [0.0, 0.0]) == 0.0
    assert np.isfinite(M.smape([0.0], [1e-12]))


def test_mase_nan_on_constant_series():
    assert np.isnan(M.mase([1, 1], [1, 1], M.naive_scale([5, 5, 5])))
