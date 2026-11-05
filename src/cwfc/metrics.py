from __future__ import annotations

import numpy as np

EPS = 1e-9


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def smape(y, p, eps: float = EPS):
    """Symmetric MAPE in %. Denominator is floored at `eps`; when y and p are both ~0 the term is 0."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(100 * np.mean(2 * np.abs(p - y) / np.maximum(np.abs(y) + np.abs(p), eps)))


def mase(y, p, scale: float):
    """MAE scaled by the in-sample one-step naive MAE (nan if that scale is ~0, e.g. constant series)."""
    return float(mae(y, p) / scale) if scale and scale > EPS else float("nan")


def rel_improvement(model_mae: float, ref_mae: float):
    """% MAE reduction vs reference (nan when the reference error is ~0)."""
    return float((ref_mae - model_mae) / ref_mae * 100) if ref_mae > EPS else float("nan")


def naive_scale(y_train) -> float:
    y = np.asarray(y_train, float)
    d = np.abs(np.diff(y[np.isfinite(y)]))
    return float(d.mean()) if len(d) else float("nan")


def summarize(y, p, scale, peak_mask=None) -> dict:
    out = {"MAE": mae(y, p), "RMSE": rmse(y, p), "sMAPE%": smape(y, p), "MASE": mase(y, p, scale)}
    if peak_mask is not None and np.asarray(peak_mask).any():
        m = np.asarray(peak_mask, bool)
        out["Peak_MAE"] = mae(np.asarray(y)[m], np.asarray(p)[m])
    else:
        out["Peak_MAE"] = float("nan")
    return out
