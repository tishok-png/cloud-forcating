"""Rolling-origin validation, tuning, model selection and final forecasting for one target.

Everything here receives ONLY the training frame (first n-24 rows). The holdout never enters this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import metrics as M
from .data import fill_for_features
from .models import GRIDS, holt, ml_forecast, moving_average, persistence, seasonal_naive

BASELINES = ["Persistence", "MovingAvg(12)", "SeasonalNaive(12)"]
MAIN = ["Holt", "Ridge", "RandomForest", "XGBoost"]
EXTRA = ["Holt(damped)", "Ensemble(main4)"]
ORDER = BASELINES + MAIN + EXTRA  # also the tie-break order (simplest first)

HORIZON = 24


def make_origins(n_train: int, horizon: int = HORIZON, step: int = 12, min_train: int = 40) -> list[int]:
    """Expanding-window origins. Fold k trains on rows [0, o) and forecasts rows [o, o+horizon)."""
    o, out = n_train - horizon, []
    while o >= min_train:
        out.append(o)
        o -= step
    return out[::-1]


def forecast_all(train: pd.DataFrame, target: str, horizon: int, params: dict | None = None) -> dict[str, np.ndarray]:
    params = params or {}
    tr = fill_for_features(train)
    y = tr[target].to_numpy(float)
    out = {
        "Persistence": persistence(y, horizon),
        "MovingAvg(12)": moving_average(y, horizon),
        "SeasonalNaive(12)": seasonal_naive(y, horizon),
        "Holt": holt(y, horizon),
        "Holt(damped)": holt(y, horizon, damped=True),
    }
    for fam in ("Ridge", "RandomForest", "XGBoost"):
        out[fam] = ml_forecast(tr, target, horizon, fam, params.get(fam, {}))
    out["Ensemble(main4)"] = np.mean([out[m] for m in MAIN], axis=0)
    hi = 100.0 if target == "aggregate_cpu_utilization" else np.inf
    return {k: np.clip(v, 0.0, hi) for k, v in out.items()}  # physical bounds


def tune(train: pd.DataFrame, target: str, origins: list[int], horizon: int = HORIZON) -> dict:
    """Grid search scored by rolling-origin MAE on the training window only."""
    tr = fill_for_features(train)
    best: dict = {}
    for fam, grid in GRIDS.items():
        scores = []
        for p in grid:
            errs = []
            for o in origins:
                y_true = tr[target].to_numpy(float)[o:o + horizon]
                pred = ml_forecast(tr.iloc[:o], target, horizon, fam, p)
                errs.append(np.abs(y_true - pred).mean())
            scores.append(np.mean(errs))
        best[fam] = grid[int(np.argmin(scores))]
    return best


@dataclass
class Result:
    target: str
    origins: list[int]
    params: dict
    cv_preds: pd.DataFrame              # long: fold, h, y, model, pred, peak
    cv_table: pd.DataFrame              # per-model pooled CV metrics
    chosen: str
    forecasts: dict[str, np.ndarray]    # all candidates, refit on full training window
    band_q90: float                     # 90% quantile of |CV error| of the chosen model
    train_scale: float = field(default=np.nan)
    peak_threshold: float = field(default=np.nan)


def run_target(train: pd.DataFrame, target: str, horizon: int = HORIZON, model: str = "auto") -> Result:
    tr = fill_for_features(train)
    y = tr[target].to_numpy(float)
    origins = make_origins(len(tr), horizon)
    if not origins:  # tiny smoke dataset: single fold at the end of a short history
        origins = [max(len(tr) - horizon, 12)]
    params = tune(tr, target, origins, horizon)

    rows, scales = [], []
    for k, o in enumerate(origins):
        fold_train = tr.iloc[:o]
        preds = forecast_all(fold_train, target, horizon, params)
        yt = y[o:o + horizon]
        thr = np.nanquantile(fold_train[target].to_numpy(float), 0.75)  # peak = top quartile of the *fold's* history
        scales.append(M.naive_scale(fold_train[target]))
        for name, p in preds.items():
            for h, (a, b) in enumerate(zip(yt, p), start=1):
                rows.append((k, h, a, name, b, a >= thr))
    cv = pd.DataFrame(rows, columns=["fold", "h", "y", "model", "pred", "peak"])
    scale = float(np.nanmean(scales))

    recs = {}
    for name in ORDER:
        d = cv[cv.model == name]
        recs[name] = M.summarize(d.y, d.pred, scale, d.peak)
    table = pd.DataFrame(recs).T
    ref = table.loc["Persistence", "MAE"]
    table["vs_Persistence%"] = [M.rel_improvement(m, ref) for m in table["MAE"]]
    best_base = table.loc[BASELINES, "MAE"].min()
    table["vs_BestBaseline%"] = [M.rel_improvement(m, best_base) for m in table["MAE"]]
    table["type"] = ["baseline" if n in BASELINES else ("main" if n in MAIN else "extra") for n in table.index]

    tol = 1e-6 * max(np.nanmean(np.abs(y)), 1.0)
    if model == "auto":  # CV-best; exact ties resolved in favour of the simplest model (ORDER)
        mn = table["MAE"].min()
        chosen = next(n for n in ORDER if table.loc[n, "MAE"] <= mn + tol)
    else:
        chosen = model
    d = cv[cv.model == chosen]
    q90 = float(np.quantile(np.abs(d.y - d.pred), 0.90))

    final = forecast_all(tr, target, horizon, params)
    return Result(target, origins, params, cv, table, chosen, final, q90, scale,
                  float(np.nanquantile(y, 0.75)))


def score_holdout(res: Result, y_hold: np.ndarray, y_train: np.ndarray) -> pd.DataFrame:
    """Score the (already fixed) forecasts against the public holdout. REPORTING ONLY."""
    scale = M.naive_scale(y_train)
    peak = y_hold >= res.peak_threshold
    t = pd.DataFrame({n: M.summarize(y_hold, f, scale, peak) for n, f in res.forecasts.items()}).T.loc[ORDER]
    ref = t.loc["Persistence", "MAE"]
    t["vs_Persistence%"] = [M.rel_improvement(m, ref) for m in t["MAE"]]
    bb = t.loc[BASELINES, "MAE"].min()
    t["vs_BestBaseline%"] = [M.rel_improvement(m, bb) for m in t["MAE"]]
    return t
