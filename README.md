# Cloud Workload Forecasting — Azure VM trace (data-v1)

Forecasts `active_vm_count` and `aggregate_cpu_utilization` for the final 2 h (24 × 5 min) of the challenge trace.
Baselines: persistence, moving average, seasonal-naive. Main models: **Holt (linear), Ridge, Random Forest, XGBoost**
(+ Holt-damped and an equal-weight ensemble as extras). Model choice is made by **rolling-origin CV on the first 110 steps only**.

## Install (Python ≥ 3.11)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Data
Download `data-v1.tar.gz` from the release and extract to `challenge_data/` (not committed):
```bash
mkdir -p challenge_data && tar -xzf data-v1.tar.gz -C challenge_data
(cd challenge_data && sha256sum -c SHA256SUMS)
```

## Generate the required predictions (one command)
```bash
python -m cwfc.predict --data challenge_data --output predictions.csv
```
Writes `predictions.csv` (`timestamp,active_vm_count,aggregate_cpu_utilization`, 24 rows) and
`predictions_with_intervals.csv` (90 % band, used by the capacity policy). Runtime ≈ 1 min on a laptop CPU.
`--model auto` (default) = best rolling-origin CV MAE; `--model XGBoost` etc. forces a model (a forced choice
must not be made by looking at the holdout).

## Reproduce all tables/figures, tests, notebook
```bash
python -m cwfc.evaluate --data challenge_data     # reports/results/*.csv, reports/figures/*.png
python -m pytest -q                               # 14 tests incl. leakage tests
jupyter nbconvert --to notebook --execute --inplace notebooks/cpu_forecast.ipynb
```

## Layout
```
src/cwfc/  data.py (load, checksum, audit, targets)   features.py (origin-safe features)
           models.py (baselines, Holt, Ridge/RF/XGB)  pipeline.py (CV, tuning, selection)
           metrics.py  policy.py  predict.py (CLI)  evaluate.py (report artefacts)
notebooks/cpu_forecast.ipynb      executed EDA + modelling walkthrough
reports/final_report.md           4–6 page report;  reports/figures, reports/results
tests/                            schema, no-look-ahead, holdout-invariance, metrics, tiny-data smoke test
```

## Main modeling decisions
* **Targets:** per-timestamp `nunique(vm_id)` and `mean(avg_cpu)` over valid rows (null / outside [0,100] excluded from both). `avg_cpu` is already a percentage — no rescaling. Missing grid timestamps stay NaN (never invented); only *features* use a causal forward-fill.
* **Split:** full series built first, then first 110 / last 24. The pipeline is only ever handed the first 110 rows.
* **Validation:** expanding-window rolling origin inside the 110 (origins 50, 62, 74, 86; horizon 24). Hyper-parameters are tuned on the same folds (tiny grids); the public holdout is scored once, for reporting only.
* **Features (valid at forecast origin):** deviations of lags 0/1/2/5/11 and 6-step mean from the 12-step level, 12-step std, diff, mean-reversion term, deviations of cross-sectional VM summaries (median, p95, mean max_cpu, share ≥ 50 %). ML models are pooled-direct (horizon is a feature); target is `y[t+h] − level[t]`.
* **Selection rule:** lowest CV MAE; exact ties → simplest model. `active_vm_count` is constant (10,000) so all models tie and the last value is used.
* **No resampling:** modelling cadence = native 5 min, so no mapping back is required.
* **Reproducibility:** fixed seeds (42), single-thread estimators, per-file SHA-256 verification.
