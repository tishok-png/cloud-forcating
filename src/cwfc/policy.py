"""Capacity-headroom policy evaluation (forecast + margin) on forecast errors."""
from __future__ import annotations

import numpy as np
import pandas as pd


def margin_tradeoff(y: np.ndarray, pred: np.ndarray, margins) -> pd.DataFrame:
    """For each additive margin m (percentage points): provisioned = pred + m.

    under_rate  : share of slots where demand exceeds the provisioned level (SLA-risk proxy)
    mean_short  : mean shortfall when under-provisioned (pp)
    mean_slack  : mean unused headroom = provisioned - demand, floored at 0 (pp) (waste proxy)
    """
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    rows = []
    for m in margins:
        prov = pred + m
        short = np.maximum(y - prov, 0)
        slack = np.maximum(prov - y, 0)
        rows.append({"margin_pp": m, "under_rate%": 100 * float((short > 0).mean()),
                     "mean_shortfall_pp": float(short[short > 0].mean()) if (short > 0).any() else 0.0,
                     "mean_slack_pp": float(slack.mean())})
    return pd.DataFrame(rows)


def peak_risk_trigger(y_hist: np.ndarray, k: float = 2.0, win: int = 12) -> bool:
    """True when the latest value sits > k robust-sigmas above the trailing `win`-step mean (level shift alert)."""
    y = np.asarray(y_hist, float)
    ref = y[-win - 1:-1]
    sd = max(np.std(np.diff(y[-4 * win:])) / np.sqrt(2), 1e-9)
    return bool(y[-1] > ref.mean() + k * sd)
