"""Leakage-safe forecast-origin features.

Information-availability assumption: at forecast origin ``t`` (the last observed 5-minute slot) we
know the target AND the cross-sectional VM summaries for every slot <= t, nothing later.
Every feature in row ``t`` is built from rows <= t only (shift / trailing rolling / expanding).
The pipeline additionally builds features on the *truncated* training frame, so a feature can never
see rows beyond the training window (verified in tests/test_features.py).

Features are expressed relative to the trailing 12-step level ("level"), so tree models need not
extrapolate the absolute level; the learners predict the deviation y[t+h] - level[t].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import EXOG

WIN = 12  # one hour of 5-minute slots
FEATURES = [
    "d0", "d1", "d2", "d5", "d11", "ma6_rel", "std12", "diff1", "rev",
    *[f"{c}_rel" for c in EXOG],
]


def origin_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    y = df[target].astype(float)
    level = y.rolling(WIN).mean()
    f = pd.DataFrame(index=df.index)
    f["level"] = level  # used to anchor the forecast, not a model input
    f["d0"] = y - level
    for k in (1, 2, 5, 11):
        f[f"d{k}"] = y.shift(k) - level
    f["ma6_rel"] = y.rolling(6).mean() - level
    f["std12"] = y.rolling(WIN).std()
    f["diff1"] = y.diff(1)
    f["rev"] = level - y.expanding().mean()  # mean-reversion pressure: level vs. history so far
    for c in EXOG:
        x = df[c].astype(float)
        f[f"{c}_rel"] = x - x.rolling(WIN).mean()
    return f


def pooled_training_set(feat: pd.DataFrame, y: np.ndarray, horizon: int):
    """Rows (origin t, horizon h) with t+h inside the training window. Target = y[t+h] - level[t]."""
    X, T = [], []
    n = len(feat)
    F = feat[FEATURES].to_numpy(float)
    lev = feat["level"].to_numpy(float)
    for t in range(n - 1):
        if not np.isfinite(F[t]).all() or not np.isfinite(lev[t]):
            continue
        for h in range(1, min(horizon, n - 1 - t) + 1):
            if np.isfinite(y[t + h]):
                X.append(np.r_[F[t], h])
                T.append(y[t + h] - lev[t])
    return np.asarray(X), np.asarray(T)


def forecast_row(feat: pd.DataFrame, horizon: int):
    """Feature matrix for origin = last training row, h = 1..horizon, and the level anchor."""
    last = feat.iloc[-1]
    base = last[FEATURES].to_numpy(float)
    X = np.vstack([np.r_[base, h] for h in range(1, horizon + 1)])
    return X, float(last["level"])
