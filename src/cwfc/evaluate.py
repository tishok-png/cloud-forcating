"""Reproduce every table and figure in reports/ :  python -m cwfc.evaluate --data challenge_data

Protocol: the pipeline is fitted/tuned/selected on the first n-24 rows only. The (already fixed) forecasts are
then scored once against the public holdout, for REPORTING. Nothing in this script feeds back into selection.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import TARGETS, build_series, load_metadata, load_readings, split_series
from .pipeline import BASELINES, MAIN, ORDER, run_target, score_holdout
from .policy import margin_tradeoff

COL = {"baseline": "#8c8c8c", "main": "#1f77b4", "extra": "#9ecae1"}


def _bar(ax, table, title, col="MAE"):
    t = table[col]
    ax.barh(t.index[::-1], t.values[::-1], color=[COL[table.loc[n, "type"]] for n in t.index[::-1]])
    ax.set_title(title); ax.set_xlabel(col)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="challenge_data")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args(argv)
    out = Path(a.out); (out / "results").mkdir(parents=True, exist_ok=True); (out / "figures").mkdir(parents=True, exist_ok=True)

    readings = load_readings(a.data); meta = load_metadata(a.data)
    series = build_series(readings)
    series.to_csv(out / "results" / "target_series.csv")
    train, hold = split_series(series)
    R = {t: run_target(train, t) for t in TARGETS}

    for t, r in R.items():
        short = "cpu" if "cpu" in t else "count"
        tbl = r.cv_table.copy(); tbl.to_csv(out / "results" / f"cv_{short}.csv")
        ht = score_holdout(r, hold[t].to_numpy(float), train[t].to_numpy(float)); ht["type"] = tbl["type"]
        ht.to_csv(out / "results" / f"holdout_{short}.csv")
        pf = r.cv_preds.assign(err=lambda d: (d.y - d.pred).abs()).pivot_table(index="model", columns="fold", values="err").loc[ORDER]
        pf.to_csv(out / "results" / f"cv_fold_mae_{short}.csv")
        print(f"\n== {t}: chosen by CV = {r.chosen}; origins={r.origins}; params={r.params}")
        print(tbl.drop(columns="type").round(4).to_string()); print("-- holdout (reporting only)"); print(ht.drop(columns="type").round(4).to_string())

    r = R["aggregate_cpu_utilization"]; t = "aggregate_cpu_utilization"
    ht = pd.read_csv(out / "results" / "holdout_cpu.csv", index_col=0)

    # error by horizon (chosen + all main)
    byh = r.cv_preds.assign(err=lambda d: (d.y - d.pred).abs()).pivot_table(index="h", columns="model", values="err")
    byh.to_csv(out / "results" / "cv_mae_by_horizon_cpu.csv")

    # policy trade-off on CV forecasts of the chosen model, validated once on holdout
    d = r.cv_preds[r.cv_preds.model == r.chosen]
    margins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
    pol_cv = margin_tradeoff(d.y, d.pred, margins); pol_cv.to_csv(out / "results" / "policy_cv.csv", index=False)
    pol_ho = margin_tradeoff(hold[t].to_numpy(float), r.forecasts[r.chosen], margins); pol_ho.to_csv(out / "results" / "policy_holdout.csv", index=False)
    print("\n== policy (CV):\n", pol_cv.round(3).to_string()); print("== policy (holdout):\n", pol_ho.round(3).to_string())

    # ---------------- figures
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    _bar(ax[0], r.cv_table, "Rolling-origin CV MAE (4 folds x 24 steps)")
    _bar(ax[1], ht, "Public holdout MAE (reporting only)")
    plt.tight_layout(); plt.savefig(out / "figures" / "model_comparison.png", dpi=150); plt.close()

    x_tr = np.arange(len(train)); x_ho = np.arange(len(train), len(series))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(x_tr[-60:], train[t].to_numpy()[-60:], color="k", lw=1.2, label="train (last 60)")
    ax.plot(x_ho, hold[t], color="k", lw=2, label="holdout actual")
    for n, c in [("Persistence", "#8c8c8c"), ("MovingAvg(12)", "#bdbdbd"), ("Holt", "#d62728"), ("Ridge", "#2ca02c"), ("RandomForest", "#ff7f0e"), ("XGBoost", "#1f77b4")]:
        ax.plot(x_ho, r.forecasts[n], color=c, lw=1.4, ls="--" if n in BASELINES else "-", label=n)
    f = r.forecasts[r.chosen]
    ax.fill_between(x_ho, f - r.band_q90, f + r.band_q90, color="#8c8c8c", alpha=0.15, label=f"{r.chosen} ±90% CV band")
    ax.axvline(len(train) - 0.5, color="r", ls=":"); ax.set_ylabel("aggregate CPU (%)"); ax.set_xlabel("5-min step")
    ax.set_title("2-hour holdout: forecasts vs actual (models fitted on first 110 steps only)"); ax.legend(ncol=3, fontsize=8)
    plt.tight_layout(); plt.savefig(out / "figures" / "holdout_forecasts.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    for n in ["Persistence", "MovingAvg(12)", "Holt", "Ridge", "RandomForest", "XGBoost"]:
        ax.plot(byh.index, byh[n], label=n, ls="--" if n in BASELINES else "-")
    ax.set_xlabel("forecast horizon (steps of 5 min)"); ax.set_ylabel("CV MAE (pp)"); ax.set_title("Error vs horizon"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out / "figures" / "error_by_horizon.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(7, 4)); ax2 = ax.twinx()
    ax.plot(pol_cv["margin_pp"], pol_cv["under_rate%"], "o-", color="#d62728", label="under-provision rate % (CV)")
    ax.plot(pol_ho["margin_pp"], pol_ho["under_rate%"], "s:", color="#d62728", alpha=.6, label="under-provision rate % (holdout)")
    ax2.plot(pol_cv["margin_pp"], pol_cv["mean_slack_pp"], "o-", color="#1f77b4", label="mean slack pp (CV)")
    ax.set_xlabel("safety margin added to forecast (pp CPU)"); ax.set_ylabel("slots under-provisioned (%)"); ax2.set_ylabel("mean slack (pp)")
    ax.set_title("Under- vs over-provisioning trade-off"); ax.legend(loc="upper center", fontsize=8); ax2.legend(loc="center right", fontsize=8)
    plt.tight_layout(); plt.savefig(out / "figures" / "policy_tradeoff.png", dpi=150); plt.close()

    # EDA figures
    m = readings.merge(meta[["vm_id", "vm_category", "virtual_core_bucket"]], on="vm_id")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    s = series; xs = np.arange(len(s))
    ax[0].plot(xs, s["aggregate_cpu_utilization"], color="k", label="mean"); ax[0].plot(xs, s["median_cpu"], label="median"); ax[0].set_ylabel("avg_cpu (%)")
    ax[0].axvline(len(train) - .5, color="r", ls=":"); ax[0].set_title("Fleet mean / median CPU"); ax[0].legend()
    seg = m.groupby(["timestamp", "vm_category"]).avg_cpu.mean().unstack()
    for c in seg: ax[1].plot(xs, seg[c].to_numpy(), label=c)
    ax[1].set_title("Mean CPU by VM category"); ax[1].legend()
    ax[2].hist(readings.avg_cpu, bins=np.logspace(-2, 2, 60), color="#1f77b4"); ax[2].set_xscale("log"); ax[2].set_title("VM-level avg_cpu distribution (log x)")
    for a_ in ax[:2]: a_.set_xlabel("5-min step")
    plt.tight_layout(); plt.savefig(out / "figures" / "eda_overview.png", dpi=150); plt.close()

    seg_stats = m.groupby("vm_category").avg_cpu.agg(mean="mean", median="median", p95=lambda s: s.quantile(.95)).round(2)
    seg_stats["step_to_step_sd_of_segment_mean"] = seg.diff().std().round(3)
    seg_stats.to_csv(out / "results" / "segments_category.csv")
    m.groupby("virtual_core_bucket").avg_cpu.agg(["mean", "median", "count"]).round(2).to_csv(out / "results" / "segments_cores.csv")
    print("\n", seg_stats)


if __name__ == "__main__":
    main()
