import numpy as np
import pandas as pd
from cwfc.data import build_series, split_series
from cwfc.pipeline import make_origins, run_target
from cwfc.predict import make_predictions, validate
from conftest import synth_readings


def test_origins_expanding_and_inside_train():
    o = make_origins(110)
    assert o == [50, 62, 74, 86] and max(o) + 24 <= 110


def test_predictions_schema():
    s = build_series(synth_readings())
    preds, intervals, meta = make_predictions(s)
    validate(preds)
    assert len(intervals) == 24 and set(meta) == {"active_vm_count", "aggregate_cpu_utilization"}
    assert (preds["timestamp"].to_numpy() == s.index[-24:].to_numpy()).all()


def test_predictions_do_not_depend_on_holdout_values():
    """Leakage test: rewriting the 24 holdout readings must not change a single forecast."""
    a, _, _ = make_predictions(build_series(synth_readings(hold_shift=0.0)))
    b, _, _ = make_predictions(build_series(synth_readings(hold_shift=25.0)))
    pd.testing.assert_frame_equal(a, b)


def test_pipeline_runs_on_tiny_history():
    s = build_series(synth_readings(n_vm=8, n_t=60))
    tr, _ = split_series(s)
    r = run_target(tr, "aggregate_cpu_utilization")
    assert np.isfinite(r.forecasts[r.chosen]).all() and len(r.forecasts[r.chosen]) == 24
