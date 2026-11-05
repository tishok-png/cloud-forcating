"""Loading, integrity checks, audit and construction of the target series.

Treatment of irregularities (documented, see reports/final_report.md):
* rows with a null or out-of-range (outside [0, 100]) ``avg_cpu`` are NOT valid observations:
  they are excluded from BOTH targets (so active_vm_count counts only VMs with a valid reading);
* a VM is counted at most once per timestamp (``nunique`` on vm_id);
* timestamps are put on the 300 s grid. A missing grid timestamp stays NaN in the raw series
  (no invented target values); for *features only* it is carried forward causally (ffill).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

READINGS = "vm_cpu_readings.csv.gz"
METADATA = "vm_metadata.csv.gz"
STEP = 300
TARGETS = ["active_vm_count", "aggregate_cpu_utilization"]
EXOG = ["median_cpu", "p95_cpu", "mean_max_cpu", "share_hot"]
HOT = 50.0  # a VM-reading is "hot" if avg_cpu >= 50 %


def verify_checksums(data_dir: str | Path) -> dict[str, bool]:
    """Hash every file listed in SHA256SUMS (per-file hash, not the tarball hash)."""
    data_dir = Path(data_dir)
    sums = data_dir / "SHA256SUMS"
    out: dict[str, bool] = {}
    if not sums.exists():
        return out
    for line in sums.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.strip().lstrip("*")
        path = data_dir / name
        if not path.exists():
            out[name] = False
            continue
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        out[name] = h.hexdigest() == digest
    return out


def load_readings(data_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / READINGS)


def load_metadata(data_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / METADATA)


def audit(readings: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    ts = np.sort(readings["timestamp"].unique())
    per_vm = readings.groupby("vm_id").size()
    per_ts = readings.groupby("timestamp")["vm_id"].nunique()
    bad = (readings["min_cpu"] > readings["avg_cpu"] + 1e-9) | (readings["avg_cpu"] > readings["max_cpu"] + 1e-9)
    rows = {
        "reading rows": len(readings),
        "distinct VMs": readings["vm_id"].nunique(),
        "distinct timestamps": len(ts),
        "timestamp spacing (s)": ", ".join(map(str, np.unique(np.diff(ts)))),
        "duplicate (vm_id, timestamp)": int(readings.duplicated(["vm_id", "timestamp"]).sum()),
        "null CPU values": int(readings[["min_cpu", "avg_cpu", "max_cpu"]].isna().sum().sum()),
        "avg_cpu outside [0,100]": int((~readings["avg_cpu"].between(0, 100)).sum()),
        "rows violating min<=avg<=max": int(bad.sum()),
        "observations per VM (min/max)": f"{per_vm.min()} / {per_vm.max()}",
        "VMs per timestamp (min/max)": f"{per_ts.min()} / {per_ts.max()}",
        "metadata rows / unique vm_id": f"{len(meta)} / {meta['vm_id'].nunique()}",
        "readings VMs missing in metadata": len(set(readings["vm_id"]) - set(meta["vm_id"])),
        "metadata VMs missing in readings": len(set(meta["vm_id"]) - set(readings["vm_id"])),
        "VMs deleted inside trace window": int((meta["timestamp_vm_deleted"] <= readings["timestamp"].max()).sum()),
        "VMs created inside trace window (>0)": int((meta["timestamp_vm_created"] > 0).sum()),
    }
    return pd.Series(rows, name="value").to_frame()


def build_series(readings: pd.DataFrame, step: int = STEP) -> pd.DataFrame:
    """Aggregate VM-level readings to one row per 5-minute timestamp.

    Returns a frame indexed by ``timestamp`` (full grid) with the two required targets and
    cross-sectional summaries (``EXOG``) that are computable from readings *at that timestamp only*.
    """
    r = readings.dropna(subset=["avg_cpu"])
    r = r[r["avg_cpu"].between(0, 100)]
    g = r.groupby("timestamp")
    df = pd.DataFrame(
        {
            "active_vm_count": g["vm_id"].nunique(),
            "aggregate_cpu_utilization": g["avg_cpu"].mean(),
            "median_cpu": g["avg_cpu"].median(),
            "p95_cpu": g["avg_cpu"].quantile(0.95),
            "mean_max_cpu": g["max_cpu"].mean(),
            "share_hot": g["avg_cpu"].apply(lambda s: float((s >= HOT).mean())),
        }
    )
    df.index.name = "timestamp"
    grid = np.arange(df.index.min(), df.index.max() + step, step)
    return df.reindex(grid).rename_axis("timestamp")


def split_series(df: pd.DataFrame, horizon: int = 24) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split on the FULL target series: first n-horizon rows train, last `horizon` holdout."""
    return df.iloc[:-horizon].copy(), df.iloc[-horizon:].copy()


def fill_for_features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal forward-fill used only to build features when grid timestamps are missing."""
    return df.ffill()  # causal only: never back-fill (that would borrow future values)
