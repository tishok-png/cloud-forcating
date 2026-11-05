import pandas as pd
import numpy as np
from cwfc.data import build_series, split_series, verify_checksums


def test_series_shape_and_counts(readings):
    s = build_series(readings)
    assert len(s) == 134 and (s["active_vm_count"] == 40).all()
    assert s["aggregate_cpu_utilization"].between(0, 100).all()
    assert np.allclose(np.diff(s.index), 300)


def test_invalid_rows_excluded_and_vm_counted_once(readings):
    r = readings.copy()
    r.loc[0, "avg_cpu"] = np.nan          # invalid reading -> VM not counted at that timestamp
    r = pd.concat([r, r.iloc[[5]]], ignore_index=True)            # duplicate row -> still counted once
    s = build_series(r)
    assert s["active_vm_count"].iloc[0] == 39
    assert s["active_vm_count"].iloc[5] == 40


def test_missing_grid_timestamp_is_nan_not_invented(readings):
    s = build_series(readings[readings.timestamp != 300 * 50])
    assert len(s) == 134 and s["aggregate_cpu_utilization"].isna().sum() == 1


def test_split_is_chronological(readings):
    s = build_series(readings)
    tr, ho = split_series(s)
    assert len(tr) == 110 and len(ho) == 24 and tr.index.max() < ho.index.min()


def test_checksum_missing_file_returns_empty(tmp_path):
    assert verify_checksums(tmp_path) == {}
