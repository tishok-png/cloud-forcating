"""Baselines, Holt linear, and pooled-direct ML forecasters (Ridge / Random Forest / XGBoost)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import Holt
from xgboost import XGBRegressor

from .features import forecast_row, origin_features, pooled_training_set

SEED = 42
SEASON = 12  # 1 hour. The trace is only 11 h long, so there is NO daily seasonality to exploit.


# ------------------------------------------------------------------ baselines
def persistence(y: np.ndarray, h: int) -> np.ndarray:
    return np.repeat(y[-1], h)


def moving_average(y: np.ndarray, h: int, window: int = 12) -> np.ndarray:
    return np.repeat(np.nanmean(y[-window:]), h)


def seasonal_naive(y: np.ndarray, h: int, season: int = SEASON) -> np.ndarray:
    last = y[-season:]
    return np.array([last[i % season] for i in range(h)])


# ------------------------------------------------------------------ Holt
def holt(y: np.ndarray, h: int, damped: bool = False) -> np.ndarray:
    """Holt's linear-trend exponential smoothing (smoothing params estimated by MLE on the training window)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = Holt(np.asarray(y, float), damped_trend=damped, initialization_method="estimated").fit()
            return np.asarray(fit.forecast(h), float)
        except Exception:  # extremely short / degenerate series -> fall back to last value
            return persistence(y, h)


# ------------------------------------------------------------------ ML (pooled direct, horizon as a feature)
def make_estimator(family: str, **p):
    if family == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=p.get("alpha", 10.0)))
    if family == "RandomForest":
        return RandomForestRegressor(
            n_estimators=p.get("n_estimators", 200), max_depth=p.get("max_depth", 4),
            min_samples_leaf=p.get("min_samples_leaf", 10), random_state=SEED, n_jobs=1,
        )
    if family == "XGBoost":
        return XGBRegressor(
            n_estimators=p.get("n_estimators", 150), max_depth=p.get("max_depth", 2),
            learning_rate=p.get("learning_rate", 0.03), subsample=0.8, colsample_bytree=0.8,
            min_child_weight=p.get("min_child_weight", 5), random_state=SEED, n_jobs=1, verbosity=0,
        )
    raise ValueError(family)


def ml_forecast(train_df: pd.DataFrame, target: str, horizon: int, family: str, params: dict) -> np.ndarray:
    feat = origin_features(train_df, target)  # built on the TRUNCATED training frame only
    X, t = pooled_training_set(feat, train_df[target].to_numpy(float), horizon)
    if len(X) < 20:  # too little history -> degrade gracefully
        return persistence(train_df[target].to_numpy(float), horizon)
    est = make_estimator(family, **params).fit(X, t)
    Xf, level = forecast_row(feat, horizon)
    return level + est.predict(Xf)


GRIDS = {
    "Ridge": [{"alpha": a} for a in (0.1, 1.0, 10.0, 100.0, 1000.0)],
    "RandomForest": [{"max_depth": d, "min_samples_leaf": m} for d in (3, 6) for m in (5, 20)],
    "XGBoost": [{"max_depth": d, "n_estimators": n} for d in (2, 3) for n in (100, 300)],
}
