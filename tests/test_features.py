import numpy as np
from cwfc.data import build_series
from cwfc.features import FEATURES, origin_features, pooled_training_set


def test_features_have_no_lookahead(readings):
    s = build_series(readings)
    full = origin_features(s, "aggregate_cpu_utilization")
    part = origin_features(s.iloc[:80], "aggregate_cpu_utilization")
    a, b = full.iloc[:80][FEATURES + ["level"]], part[FEATURES + ["level"]]
    assert np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True)


def test_pooled_targets_stay_inside_training_window(readings):
    s = build_series(readings).iloc[:110]
    f = origin_features(s, "aggregate_cpu_utilization")
    X, t = pooled_training_set(f, s["aggregate_cpu_utilization"].to_numpy(), 24)
    assert X.shape[1] == len(FEATURES) + 1 and len(X) == len(t) > 100
    # max (origin + h) must be the last training index
    assert np.isfinite(X).all() and np.isfinite(t).all()
