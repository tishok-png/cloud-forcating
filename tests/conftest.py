import numpy as np
import pandas as pd
import pytest


def synth_readings(n_vm=40, n_t=134, seed=0, hold_shift=0.0):
    rng = np.random.default_rng(seed)
    base = rng.gamma(2.0, 4.0, n_vm)[:, None]
    t = np.arange(n_t)
    wave = 1 + 0.05 * np.sin(t / 9)[None, :]
    avg = np.clip(base * wave + rng.normal(0, 0.8, (n_vm, n_t)), 0.01, 99)
    avg[:, -24:] += hold_shift  # only the holdout period differs between two synthetic variants
    avg = np.clip(avg, 0.01, 99)
    rows = pd.DataFrame({
        "timestamp": np.tile(t * 300, n_vm), "vm_id": np.repeat([f"vm{i}" for i in range(n_vm)], n_t),
        "min_cpu": avg.ravel() * 0.5, "max_cpu": np.minimum(avg.ravel() * 2 + 1, 100), "avg_cpu": avg.ravel()})
    return rows


@pytest.fixture
def readings():
    return synth_readings()
