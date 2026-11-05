# Cloud Workload Forecasting & Analysis — Final Report
**Author:** Gwong Tishok Samuel · **Data:** Azure Public Dataset V2 subset (`data-v1`) · **Code:** `src/cwfc`, reproduce with `python -m cwfc.predict`

## 1. Executive summary
* **Predictability.** Fleet-mean CPU is very stable: 9.39 % average, total swing 0.93 pp (≈ 10 % of the mean) over 11.1 h, with one ≈ +0.5 pp level shift around steps 53–72. `active_vm_count` is constant at 10,000 (no churn in this sample), so it is trivially forecastable.
* **Accuracy.** For `aggregate_cpu_utilization`, rolling-origin CV MAE is 0.21 pp (persistence, the CV winner) to 0.27 pp (Ridge), i.e. 2.2–2.9 % sMAPE. On the public 2-hour holdout the submitted forecast has MAE 0.147 pp / RMSE 0.185 pp (1.6 % sMAPE).
* **Baseline vs ML.** Under leakage-safe CV, **no main model (Holt, Ridge, Random Forest, XGBoost) beat persistence**; all were 21–32 % worse in MAE, though all differences are small in absolute terms (≤ 0.07 pp) and the fold-level ranking changes from fold to fold. The submission therefore uses the CV-selected model, persistence. The holdout, reported once and not used for selection, ranks models differently (XGBoost 0.122, Ridge 0.126, MovingAvg 0.123 < Persistence 0.147), which illustrates how noisy a 4-fold/24-point evaluation is.
* **Recommendation.** Plan on *forecast + 0.5 pp* for the fleet aggregate (zero under-provisioned slots in CV and on the holdout; mean slack ≈ 0.4–0.6 pp), use double headroom for Interactive/Unknown VMs, and add a level-shift trigger. Do **not** consolidate capacity from the 9 % mean: the tail of individual VMs is heavy.

## 2. Data understanding and preprocessing
| check | result |
|---|---|
| rows / VMs / timestamps | 1,340,000 / 10,000 / 134 |
| cadence | exactly 300 s between all timestamps |
| nulls, duplicates, `avg_cpu` outside [0,100], min ≤ avg ≤ max violations | 0, 0, 0, 0 |
| VMs per timestamp | 10,000 at every timestamp |
| metadata ↔ readings | 1:1 match; `vm_id` unique |
| VM lifetimes | all created at t = 0; earliest deletion 40,500 s > trace end 39,900 s → **zero churn inside the window** |

**Why churn = 0 matters:** the data card states only VMs with a complete trace were eligible, so the sample is survivorship-filtered. `active_vm_count` measures the *sample*, not the live fleet, and cannot be used to infer demand growth.

**Targets** (`cwfc.data.build_series`): `active_vm_count(t)` = distinct `vm_id` with a valid reading at *t* (each VM counted once); `aggregate_cpu_utilization(t)` = mean of VM `avg_cpu` at *t* in % (the field is already in percent — no rescaling). Null or out-of-range readings would be excluded from both targets; missing grid timestamps would stay NaN (no invented targets) and only *features* use a causal forward-fill. None of these occurred in this release, but the code and tests cover them. Modelling cadence is the native 5 min, so no resampling or mapping back is needed.

## 3. Workload analysis
![EDA](figures/eda_overview.png)

* **Distribution (VM-level `avg_cpu`).** Median 5.1 %, p90 19.6 %, p95 37.1 %, p99 87.4 %; 76 % of VM-readings are < 10 % and 3.1 % are ≥ 50 %. *Operationally:* the average VM is idle, but a small hot tail exists; capacity risk sits in individual VMs rather than in the fleet mean.
* **Segments.** Mean CPU: Delay-insensitive 8.2 % (p95 26 %), Interactive 11.5 % (p95 49 %), Unknown 10.2 % (p95 52 %). Step-to-step SD of the segment mean is 0.13 pp for Delay-insensitive vs 0.26–0.27 pp for Interactive/Unknown, i.e. ≈ 2× more volatile → they need larger headroom. By cores, 24-core VMs average 11.4 % and > 24-core 6.6 %.
* **Temporal pattern.** Lag-1 autocorrelation 0.67, lag-12 0.37, lag-24 −0.03 (training window): a slowly moving level plus noise. Mean level: 9.38 % (steps 0–49), **9.66 % (steps 50–74, the "bump")**, 9.28 % (75–109), 9.27 % (holdout). Peak slots are at steps 55–68 (max 9.91 %); the lowest slots are at steps 103–115 around the train/holdout boundary (min 8.98 % at step 111). The charts show the full series for description only; no modelling decision uses the holdout. Only 11 h are observed, so **no daily or weekly seasonality can be learned**; a 12-step "seasonal naive" is a convention, not a real seasonal model.

## 4. Forecasting approach
* **Baselines:** persistence (last value), 12-step moving average, 12-step seasonal-naive.
* **Main models:** Holt linear-trend exponential smoothing (statsmodels, MLE-estimated smoothing parameters); Ridge, Random Forest and XGBoost as *pooled direct* learners (one model per target, forecast horizon *h* = 1…24 is an input feature, target `y[t+h] − level[t]` with `level` = trailing 12-step mean so tree models need not extrapolate a level). Extras: Holt-damped and an equal-weight ensemble of the four main models.
* **Features** (valid at the forecast origin *t*; each uses rows ≤ *t* only): deviations of lags 0/1/2/5/11 and the 6-step mean from the 12-step level, 12-step std, 1-step diff, level vs expanding mean (mean-reversion), and deviations of cross-sectional VM summaries (median, p95, mean `max_cpu`, share of VMs ≥ 50 %). Features are built on the truncated training frame; no calendar features are used because 11 h contain no usable calendar signal.
* **Tuning:** small grids (Ridge α ∈ {0.1…1000}; RF depth × leaf; XGB depth × trees), scored by rolling-origin MAE. Selected: Ridge α = 1000, RF depth 3/leaf 5, XGB depth 3/300 trees.
* **`active_vm_count`:** same pipeline; the series is constant, every model ties at zero error and the simplest (last value = 10,000) is chosen by the tie-break rule.

## 5. Validation methodology
The full 134-step series is constructed first, then split chronologically: **steps 0–109 train, 110–133 holdout**. The pipeline receives only the 110 training rows. Model selection uses **expanding-window rolling origin inside the training window**: origins 50, 62, 74, 86; each fold trains on `[0,o)` and forecasts the next 24 steps (96 out-of-sample points, with the same 2-hour horizon the team needs). This mimics production: refit on all history, forecast the next two hours, repeat later with more data. The public holdout is scored **once** after selection, for reporting only.

Leakage controls: (i) the holdout rows are never passed to fitting/tuning/selection; (ii) `tests/test_features.py` shows features for rows ≤ *t* are identical with and without future rows; (iii) `tests/test_pipeline.py::test_predictions_do_not_depend_on_holdout_values` rewrites the 24 holdout readings and requires bit-identical forecasts.

## 6. Results vs baseline
**Rolling-origin CV (selection set), `aggregate_cpu_utilization`:**

| model | MAE (pp) | RMSE | sMAPE % | MASE* | Peak MAE | MAE vs Persistence |
|---|---|---|---|---|---|---|
| **Persistence (selected)** | **0.206** | **0.254** | 2.16 | 1.67 | **0.238** | 0 % |
| MovingAvg(12) | 0.254 | 0.297 | 2.68 | 2.06 | 0.266 | −23.8 % |
| SeasonalNaive(12) | 0.263 | 0.314 | 2.77 | 2.13 | 0.306 | −28.1 % |
| Holt (linear) | 0.269 | 0.325 | 2.82 | 2.18 | 0.302 | −30.7 % |
| Ridge | 0.272 | 0.304 | 2.87 | 2.20 | 0.287 | −32.4 % |
| Random Forest | 0.249 | 0.288 | 2.63 | 2.02 | 0.272 | −21.3 % |
| XGBoost | 0.250 | 0.291 | 2.64 | 2.02 | 0.269 | −21.4 % |
| Holt (damped) | 0.258 | 0.307 | 2.71 | 2.09 | 0.271 | −25.4 % |
| Ensemble (4 main) | 0.255 | 0.294 | 2.69 | 2.07 | 0.275 | −24.2 % |

*MASE = MAE ÷ in-sample one-step naive MAE (> 1 is expected for a 24-step horizon). Negative = worse than persistence. Peak = actual ≥ the fold's training P75.

Per-fold MAE (`reports/results/cv_fold_mae_cpu.csv`) shows no model wins every fold (persistence is best in folds 0 and 2, Ridge in fold 1, SeasonalNaive/Holt in fold 3), so the CV differences should be read as "similar within noise, persistence slightly best on average".

**Public holdout (reporting only; forecasts were fixed before scoring):**

| model | MAE | RMSE | sMAPE % | vs Persistence | vs best baseline |
|---|---|---|---|---|---|
| **Persistence (submitted)** | 0.147 | 0.185 | 1.59 | 0 % | −19.3 % |
| MovingAvg(12) | 0.123 | 0.161 | 1.33 | +16.2 % | 0 % |
| SeasonalNaive(12) | 0.140 | 0.159 | 1.51 | +4.6 % | −13.9 % |
| Holt | 0.169 | 0.208 | 1.82 | −14.9 % | −37.1 % |
| Ridge | 0.126 | 0.155 | 1.36 | +14.4 % | −2.2 % |
| Random Forest | 0.150 | 0.187 | 1.61 | −1.9 % | −21.5 % |
| XGBoost | 0.122 | 0.164 | 1.31 | +17.1 % | +1.0 % |
| Ensemble | 0.130 | 0.169 | 1.40 | +11.7 % | −5.4 % |

**Honest reading.** (1) CV and holdout rankings disagree, so neither supports a strong claim that ML beats the simple baselines; the evidence is "all models are within ≈ 0.05 pp of each other; none is reliably better". (2) Persistence did worse on the holdout partly because the last training value (9.16 %) was a low draw (the holdout mean is 9.27 %); averaging-type forecasts benefit from that. (3) We did **not** switch models after seeing the holdout — that would turn the holdout into a tuning set. A reviewer can rerun with `--model XGBoost` to see the alternative, but the pre-declared selection rule yields persistence. For `active_vm_count`, every model has MAE 0.

## 7. Error and failure analysis
![Holdout](figures/holdout_forecasts.png)
![Comparison](figures/model_comparison.png)

* **Level shift (steps 53–72).** Fold 0 forecasts through the bump onset; every model's error is largest there (MAE 0.27–0.34 pp) because no feature in the history signals an upcoming jump. Persistence re-anchors fastest *after* a shift; models that include a mean-reversion term (Ridge, RF, XGBoost) are, plausibly, more reluctant to follow it (not isolated by an ablation).
* **Peak periods.** Peak-slot CV MAE is 4–16 % higher than overall MAE for every model (persistence 0.238 vs 0.206). The holdout has only **1** slot ≥ train P75, so a holdout peak metric is not meaningful; peak performance must be judged from CV.
* **Horizon.** CV MAE does not grow cleanly with horizon (0.2–0.4 pp from h = 6 to 24 for all models; each *h* has only 4 points), which is consistent with an error dominated by noise (step-to-step SD 0.16 pp) rather than by trend mis-extrapolation. Holt's linear trend is the worst main model in CV and on the holdout because extrapolating the last slope over 2 h amplifies noise.
* **Feature value.** The cross-sectional features (median, p95, share hot) did not produce a reliable gain over simple level models; with 110 points × 24 horizons the learners are data-starved, and RF/XGB essentially predict the recent level.
* **Metrics.** sMAPE has its denominator floored at 1e-9 (both-zero ⇒ error 0); fleet CPU never approaches zero here (min 8.98 %), so near-zero denominators do not arise. MASE is `nan` for the constant count series.

## 8. Capacity-planning recommendation
Margin analysis (additive margin on the forecast, in pp of fleet-mean CPU; `reports/results/policy_*.csv`, figure below). *Under* = share of slots where demand > forecast + margin; *slack* = mean unused headroom.

| margin | CV under / slack | holdout under / slack |
|---|---|---|
| 0.0 pp | 32 % / 0.14 | 79 % / 0.02 |
| 0.3 pp | 10 % / 0.38 | 8 % / 0.20 |
| 0.4 pp | 6 % / 0.47 | 4 % / 0.30 |
| **0.5 pp** | **0 % / 0.57** | **0 % / 0.40** |

![Policy](figures/policy_tradeoff.png)

**Policy (simple, usable today):**
1. **Fleet headroom:** provision/alert at *forecast + 0.5 pp* (≈ 5 % of the mean level; ≈ the 90 % CV error band of ±0.43 pp plus buffer). Re-forecast every 5 min; the 24-step forecast is refreshed from the latest value.
2. **Segment rule:** apply 2× margin (≈ 1.0 pp) to Interactive and Unknown VMs, whose segment means are ≈ 2× as volatile as Delay-insensitive.
3. **Peak-risk trigger:** if the latest aggregate exceeds its trailing 1-hour mean by > 2 robust σ (`cwfc.policy.peak_risk_trigger`), hold an extra +0.5 pp for the next hour — this is the shape of the steps 53–72 shift that no model predicted.
4. **Do not right-size from this mean.** A 9 % average with 3 % of VM-readings ≥ 50 % and median per-VM source-trace `max_cpu` ≈ 95 % (metadata) means averages hide bursts; consolidation decisions need per-VM percentiles and multi-day history.

**Trade-off.** Under-provisioning risks throttling/latency: margins of 0.2 pp or less leave 16–29 % of slots short (0 pp: 32 % in CV, 79 % on the holdout); each extra 0.1 pp of margin adds ≈ 0.08 pp of average idle headroom (CV). Over-provisioning is cheap here in relative terms (0.5 pp ≈ 5 % of mean utilisation) so the asymmetric choice favours a margin at the 0.4–0.5 pp end.

## 9. Limitations and future work
* **Only 11.1 h** and 4 CV folds: no daily/weekly cycle, noisy model ranking, quantile estimates from 96 errors; tuning shares the same folds as selection (mildly optimistic CV); one holdout window.
* **Sample is survivorship-filtered** (no churn); results do not transfer to a live fleet with arrivals and departures.
* The margin is expressed in pp of *mean utilisation*, not in cores; translating to cores requires VM sizes (metadata buckets give only ranges).
* Future work: segment-level forecasts (category × core bucket), conformal/quantile intervals with more folds, multi-day history to learn diurnal patterns, anomaly detection on the level shift, cross-cloud comparison (Google ClusterData2019).

## Appendix — reproducibility
`python -m cwfc.predict --data challenge_data --output predictions.csv` · `python -m cwfc.evaluate` · `python -m pytest -q` (14 tests) · seeds fixed (42) · see README.
