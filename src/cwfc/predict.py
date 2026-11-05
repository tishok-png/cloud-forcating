"""One command from raw release data to predictions.csv.

    python -m cwfc.predict --data challenge_data --output predictions.csv

Leakage guard: the series is split first; ONLY the first n-24 rows are passed to the pipeline.
The holdout target values are never read by fitting, tuning or model selection (see tests).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import TARGETS, build_series, load_readings, split_series, verify_checksums
from .pipeline import HORIZON, run_target


def make_predictions(series: pd.DataFrame, horizon: int = HORIZON, model: str = "auto"):
    train, hold = split_series(series, horizon)
    out = pd.DataFrame({"timestamp": hold.index.to_numpy()})
    meta, intervals = {}, pd.DataFrame({"timestamp": hold.index.to_numpy()})
    for tgt in TARGETS:
        res = run_target(train[[c for c in train.columns]], tgt, horizon, model)  # holdout rows are NOT passed
        f = res.forecasts[res.chosen]
        if tgt == "active_vm_count":
            f = np.maximum(np.round(f), 0).astype(int)
        out[tgt] = f
        hi = 100.0 if tgt == "aggregate_cpu_utilization" else np.inf
        intervals[f"{tgt}_lower90"] = np.clip(res.forecasts[res.chosen] - res.band_q90, 0, hi)
        intervals[f"{tgt}_upper90"] = np.clip(res.forecasts[res.chosen] + res.band_q90, 0, hi)
        meta[tgt] = (res.chosen, res.params, res.band_q90)
    return out[["timestamp", *TARGETS]], intervals, meta


def validate(df: pd.DataFrame, horizon: int = HORIZON) -> None:
    assert list(df.columns) == ["timestamp", *TARGETS]
    assert len(df) == horizon and df["timestamp"].is_monotonic_increasing and not df["timestamp"].duplicated().any()
    assert np.isfinite(df[TARGETS].to_numpy(float)).all()
    assert (df["active_vm_count"] >= 0).all()
    assert df["aggregate_cpu_utilization"].between(0, 100).all()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="challenge_data")
    ap.add_argument("--output", default="predictions.csv")
    ap.add_argument("--model", default="auto", help="auto = best rolling-origin CV MAE (default) or a model name")
    ap.add_argument("--skip-checksum", action="store_true")
    a = ap.parse_args(argv)

    if not a.skip_checksum:
        res = verify_checksums(a.data)
        bad = [k for k, v in res.items() if not v]
        if bad:
            print(f"WARNING: checksum mismatch for {bad}")
    series = build_series(load_readings(a.data))
    preds, intervals, meta = make_predictions(series, model=a.model)
    validate(preds)
    preds.to_csv(a.output, index=False)
    intervals.to_csv(Path(a.output).with_name("predictions_with_intervals.csv"), index=False)
    for t, (m, p, q) in meta.items():
        print(f"{t}: model={m} | 90% band = +/-{q:.4f} | tuned={p}")
    print(f"wrote {a.output} ({len(preds)} rows)")


if __name__ == "__main__":
    main()
