# Methods changes

This note records what was checked against the review of the Drive copy, what the code now does, and the numbers produced by `python scripts/regenerate_article.py` on the attached Medellín panel (21 territories, complete IVC rows for 2013–2024). The index acronym in the new tables and figures is **IVC**.

Regenerate everything with:

```bash
pip install -r requirements.txt
python scripts/regenerate_article.py
```

That command rewrites `article_outputs/` (CSV tables, 300 dpi PNG, vector PDF), including the rainfall comparison. On this machine it finished in about two minutes. Unit checks: `python -m unittest tests.test_methodology`. The rainfall section below uses Gabriel’s full SIATA run.

Pins that produced the numbers below: Python 3.12.3, xgboost 2.1.3, scikit-learn 1.5.2, numpy 2.1.3, pandas 2.2.3, SALib 1.5.1. The full freeze is `requirements.lock`.

## This repository and the Drive copy

The review was of this codebase.

| File | Comparison |
|---|---|
| `config.yaml` | Byte-identical to the Drive copy before the edits below |
| `data/geo/comunas.geojson` | Byte-identical |
| `ModeloDatos/ModeloDatos/data/processed/indices_comunas.csv` | Same 378 rows and same values. The Drive file uses CRLF |
| `models/cv_metrics.json` | Same cross-validation metrics (mean R² 0.948105). Only `training_time_s` differs (19.32 s here, 18.35 s in the Drive file) |
| `models/ivc_predictions_oos.csv` | In the Drive bundle, absent from git. Added here (13 KB) |
| Station assignment, SIATA catalog, rain-file inventory | Not in git. Added under `data/external/` |

The minute-level rain files (about 20 GB) are not in the bundle and were not run.

## What changed in the code

- `ml/lagged_forecast.py` predicts IVC at t+1 from subindices at t. Each split fits its own min-max scaler on training rows. Hyperparameters are chosen on the last training target-year, then the model is refit and scored once on the held-out year. Baselines on the same rows: persistence (predict IVC at t), linear regression, random forest, XGBoost. A separate leave-one-territory-out run retunes without the held-out territory.
- `ml/03_xgboost_model.py` is kept as the legacy same-year reconstruction. It is no longer the forecast used for the article tables.
- `ml/clima_hook.py` and `bigdata/siata_territory_annual.py` accept rainfall either as the climate subindex inside IVC or as a lagged predictor. `clima.uso` is `ninguno`, `subindice`, or `predictor`. Passing rainfall columns in the other role raises. Fixture rows are refused as model input.
- The focused rule’s ranking is a config field: `dynamic_ivc` (re-rank every year by current IVC) or `fixed_shap` (one SHAP ranking for the whole horizon). The within-group split is `igual` or `proporcional`. The priority-set size is `n_prioritarios`.
- Saturation η(C) = min(1, η₀ − C) and the noise standard deviation are config fields. Medellín is set to saturation off and a yearly IVC ranking with equal shares inside the top 5, which is the specification the article text described. Other city files omit the keys and keep the previous behaviour (fixed SHAP, proportional shares, saturation on).
- `abm/historical_allocation.py` reads an investment-by-territory CSV. The file is not in the repository, so the scenario is skipped.
- Sobol sampling in `abm/05_sensitivity_analysis.py` covers focused, equal, and adaptive allocation. Outputs are mean IVC, the coefficient of variation (CV), and the Gini coefficient. Parameters add the budget share rho, the priority-set size, and the noise standard deviation to the four structural coefficients.
- IVC weights are declared in `config.yaml`. They are 1, 1, 1, 0.5, 1, 1.

## Hypothesis 1 — circularity

Confirmed. On the 252 complete territory-years (2013–2024), IVC equals the weighted mean of the six subindices with weights proportional to (1, 1, 1, 0.5, 1, 1). The maximum absolute residual is 1.1×10⁻¹⁶. A uniform mean misses by up to 0.0254. `idx_riesgo_clima` is 0.5 in every row of the panel, including years before 2013. An in-sample linear regression of IVC on the six same-year subindices has R² = 1.

The published mean out-of-sample R² of 0.948 is this reconstruction, approximated by trees (depth 2–3 cannot represent the linear combination exactly). Same-year R² is not evidence that the subindices forecast IVC.

## Hypothesis 2 — leakage in the legacy validation

Confirmed in the legacy script, and removed from the lagged pipeline.

- `ml/02_vulnerability_indices.py` fits `MinMaxScaler` on every year in the merged panel, including years that later fall in a test fold. Rebuilding the historical IVC with a fold-wise scaler would change the published index, so the stored panel was left as the historical object. The lagged pipeline never reuses that global year scaling: `año_normalizado` in the legacy script maps 2020 to 0.636 when 2024 is in the denominator, and to 0.875 when the scale is fit on 2013–2021 only.
- Inner search uses `GridSearchCV` with an integer `cv` (KFold on rows). It does not walk forward in time.
- The saved model’s hyperparameters are those of the outer fold with the lowest RMSE. In `models/cv_metrics.json` that is fold 5 (test years 2023–2024). The headline 0.948 is the mean of the five outer-fold R² values; the best-fold choice affects the saved model and the SHAP file the old focused rule reads.

## Hypothesis 3 — the −68.97% CV result

Confirmed in substance. The published cv₀ of 0.187532 is the population CV (ddof = 0) of the 2024 out-of-sample predicted IVC across the 21 territories. The previous default focused rule freezes a single SHAP ranking. Those five territories are the same five with the highest initial IVC (San Sebastián de Palmitas, Manrique, Santa Cruz, Popular, Villa Hermosa). Because the set never updates, dispersion rises.

Deterministic runs (noise SD = 0, rho = 0.80, top 5, initial state = 2024 predicted IVC, horizon 10):

| Specification | CV at t=10 | Change in CV | Gini at t=10 | Mean IVC change |

A negative change means the CV fell.
|---|---:|---:|---:|---:|
| Focused, yearly IVC rank, equal shares, saturation off | 0.059367 | −68.34% | 0.03301 | −42.66% |
| Focused, yearly IVC rank, proportional shares, saturation off | 0.057809 | −69.17% | 0.03214 | −42.66% |
| Focused, fixed SHAP rank, proportional shares, saturation off | 0.425466 | +126.88% | 0.23059 | −33.44% |
| Focused, fixed SHAP, saturation on | 0.259007 | +38.11% | 0.14641 | −26.05% |
| Focused, yearly IVC rank, saturation on | 0.063230 | −66.28% | 0.02886 | −30.86% |
| Equal allocation, saturation off | 0.187532 | 0.00% | 0.10619 | −40.13% |
| Equal allocation, saturation on | 0.169804 | −9.45% | 0.09569 | −27.11% |
| Adaptive, saturation off | 0.125198 | −33.24% | 0.07006 | −40.79% |
| Adaptive, saturation on | 0.139820 | −25.44% | 0.07832 | −27.47% |

The published point (−68.97%, CV 0.058193, Gini 0.032524) lies between the two yearly-rank rules and is not recovered to the last digit by either. The fixed-SHAP rule reproduces the review’s +127% figure (+126.88%). Saturation was hardcoded on (`η = min(1, 1.5 − capacity)` in every step). With it switched on, the yearly-rank reduction is −66.28%, farther from the published figure.

Equal allocation with saturation off multiplies every territory’s IVC by the same factor, so the CV does not move. The mean IVC still falls by 40%. Thirty Monte Carlo replications with noise SD = 0.05 give a mean CV change of −68.21% (SD 1.24) for the yearly IVC rank and +126.93% (SD 1.38) for the fixed SHAP rank.

Starting from observed 2024 IVC instead of the predicted IVC (CV₀ = 0.2106), the yearly equal-share rule cuts CV by 72.43%.

## Hypothesis 4 — persistence versus lagged models

Confirmed on the temporal task. Pairs are subindices at t and IVC at t+1 (231 territory-years, 2013–2023 predicting 2014–2024). The constant climate column is dropped. The walk-forward test years are 2017–2024 (168 rows) after at least three training target-years.

| Model | Walk-forward R² | RMSE | MAE | Leave-one-territory-out R² |
|---|---:|---:|---:|---:|
| Persistence | 0.821 | 0.0312 | 0.0238 | 0.783 |
| Random forest | 0.704 | 0.0402 | 0.0310 | 0.757 |
| XGBoost | 0.667 | 0.0426 | 0.0321 | 0.787 |
| Linear regression | 0.488 | 0.0528 | 0.0388 | 0.799 |

Persistence leads the forecast. The reviewer’s preliminary figures (persistence about 0.79, lagged XGBoost about 0.69–0.72, linear about 0.65–0.67) are the same ordering. On all 231 consecutive pairs, persistence alone has R² 0.783. XGBoost’s walk-forward R² is 0.667, a bit below that preliminary band; 2022 is the weak year (XGBoost R² −0.213, linear −1.379, persistence 0.411).

Leave-one-territory-out is a spatial check. It uses other territories’ whole time series, including years after the row being scored, and it is not a forecast skill. There, linear regression (R² 0.799) is slightly ahead of XGBoost (0.787) and persistence (0.783).

## Hypothesis 5 — unpinned XGBoost

Confirmed. Re-running the legacy same-year grid on xgboost 2.1.3 gives mean outer-fold R² **0.9589** (fold R² 0.963, 0.968, 0.956, 0.925, 0.983). The saved file, produced with an unpinned `xgboost>=2.0.0`, reports 0.9481. The reviewer’s rerun (0.954) sits in the same gap. The lowest-RMSE fold is still fold 5, but the selected hyperparameters changed (max depth 2 and `reg_lambda` 2.0, against max depth 3 and `reg_lambda` 1.0 in the saved file).

## Hypothesis 6 — Sobol coverage

Confirmed for the previous script: `abm/05_sensitivity_analysis.py` sampled only the focused scenario and analysed only mean IVC at t=10. The structural coefficients were beta, gamma, delta, and theta.

The new sample uses Saltelli’s design with second-order terms off, N = 128, which is 1,152 evaluations, on all three scenarios. Outputs are mean IVC, CV, and Gini. The ranges are ±20% on the four structural coefficients, rho in [0.50, 0.95], priority-set size in [1, 10] (rounded), and noise SD in [0, 0.10]. Saturation is off and the focused rule re-ranks by current IVC, matching the Medellín config. Total-order indices (ST) for the CV:

| Parameter | Focused | Equal | Adaptive |
|---|---:|---:|---:|
| Priority-set size | 0.90 | 0.00 | 0.00 |
| Budget share rho | 0.09 | 0.00 | 0.00 |
| IVC response (gamma) | 0.05 | 0.07 | 1.00 |
| Noise SD | 0.00 | 0.97 | 0.01 |
| Wellbeing rate, depreciation, infrastructure recovery | 0.00 | 0.00 | 0.00 |

Confidence intervals on the focused priority-set index are still wide (ST confidence about 0.45). The zeros are structural: with saturation off, capacity does not enter the IVC update, so depreciation and infrastructure recovery cannot move IVC, CV, or Gini. Rho and the priority-set size affect only the focused rule. Under equal allocation the deterministic CV is invariant, so the CV’s sample variance is the noise term. For mean IVC, gamma dominates in every scenario (focused ST 0.69, with priority-set size ST 0.25). Gini tracks the CV ranking. Full indices are in `article_outputs/table_sobol_indices.csv`.

## Rainfall from the local full run

Gabriel’s full run is `data/processed/siata_territorio_anual.csv` (21 territories, 2012–2026, 315 rows). The 48 station files he excluded (42, 61, 72, 171, 311, 393, 433, 539, 540, 543, 551, 558, 578, 601, 607, 612, 619, 636, 641, 704, 707, 717, 719, 727, 730, 744, 757, 759, 762, 763, 765, 766, 767, 769, 775, 780, 781, 787, 798, 816, 820, 826, 833, 842, 862, 863, 867, 870) are in neither the official assignment nor the catalog with coordinates. The processor cannot place them inside a territory or use them as IDW donors, and it now prints that list.

### Interior stations versus IDW

The rule was applied as written. A territory-year is `estaciones_interior` when at least one officially assigned gauge has rows that year, and `idw` otherwise. No row uses IDW while one of its assigned gauges is active that year.

Popular’s only official gauge is station 274. It has no rows in 2012–2017, so Popular 2012 is IDW from the 21 gauges that were already recording (coverage 0.31, observed sum 318 mm). From 2018 onward Popular is the mean of station 274 alone. Santa Cruz, Aranjuez, and Robledo have no assigned gauge in any year, so they stay on IDW. The other IDW rows are years before a territory’s own gauges start.

2012 is a partial network (median coverage 0.31, maximum 0.54). 2026 stops in February (median coverage 0.16), which matches files ending on 2026-02-28.

### Why some annual totals are near 300 mm

`precip_anual_mm` is the sum of valid minutes (`calidad == 1`, `p1 >= 0`), not a gap-filled year. It correlates 0.83 with coverage. Dividing by coverage removes that correlation (0.04 on interior rows). Interior rows with coverage at least 0.80 have a median of **1,549 mm** (range 780–2,940), which sits in Medellín’s roughly 1,500–2,500 mm/yr climatology. The units are millimetres. Days with at least 50 mm are uncommon even in wet years: several full-coverage station-years have a maximum 24-hour total just under 50 mm, so a count of zero is consistent with the maxima rather than a unit error.

The forecast uses a coverage-adjusted total, `precip_anual_ajustada_mm = precip_anual_mm / min(coverage, 1)`, and only territory-years with coverage at least 0.70 (220 of the 252 rows in 2013–2024; median adjusted total about 1,640 mm). This scaling is exact for a one-station territory. For a multi-station mean or an IDW field it scales the aggregate by the mean coverage, which equals the mean of the station ratios only when those coverages are similar. The column `ajuste` is `territorio` on this file. The processor now annualizes each eligible station before averaging or interpolating (`ajuste=estacion`), and it uses 8,784 hours in a leap year (the 2024 coverages slightly above 1 came from dividing by 8,760).

A re-run is not required for the numbers below. It is the way to replace the territory-level scaling with the station-level one:

```bash
python -m bigdata.siata_territory_annual --mode full \
  --rain-dir pluviometrica \
  --assignment data/external/asignacion_estacion_UT_oficial.json \
  --catalog data/external/pluvio_siata_catalogo.json \
  --out data/processed/siata_territorio_anual.csv
```

### Does rainfall forecast IVC?

Same 154 walk-forward rows (eligible rainfall at t and at t+1; target years 2016–2024). Three treatments, two targets. Persistence on the change predicts 0, which is what persistence of the level implies.

Walk-forward R² for the level of IVC at t+1:

| Model | No rainfall | Rainfall as predictor | Rainfall inside IVC |
|---|---:|---:|---:|
| Persistence | 0.810 | 0.810 | 0.814 |
| Random forest | 0.670 | 0.690 | 0.696 |
| XGBoost | 0.613 | 0.629 | 0.666 |
| Linear regression | 0.454 | 0.355 | 0.495 |

The subindex column rebuilds IVC, so its persistence (0.814) is the baseline for that column only. Rainfall as a predictor lifts random forest by 0.020 and XGBoost by 0.016 and lowers linear regression by 0.099. None of those models passes persistence.

Walk-forward R² for the change IVC at t+1 minus IVC at t (negative means worse than predicting the mean change):

| Model | No rainfall | Rainfall as predictor | Rainfall inside IVC |
|---|---:|---:|---:|
| Persistence (predict 0) | −0.003 | −0.003 | −0.002 |
| Random forest | −0.812 | −0.719 | −0.561 |
| XGBoost | −1.163 | −0.680 | −0.772 |
| Linear regression | −1.890 | −2.413 | −1.716 |

Every fitted change model is worse than predicting no change. Adding the predicted change back onto IVC at t, the best rainfall model (random forest, rainfall inside IVC) reaches a level R² of 0.710, against 0.814 for persistence of that same index. Rainfall does not add predictive value beyond persistence. The small tree-model gains on the level are real on this sample and too small to change that conclusion.

## Historical allocation

Schema: `data/external/inversion_historica_schema.json`. A header-only template is `data/external/inversion_historica_por_territorio.template.csv`. Columns:

| Column | Meaning |
|---|---|
| `unidad_territorial` | Exact `territorios.nombre` from `config.yaml` |
| `anio` | Calendar year. Simulation year 1 maps to `anio_inicio` (2013). Set `tipo_anio: simulacion` to use years 1…horizon instead |
| `inversion` | Allocation in the same units as `simulacion.presupuesto_anual`. Not rescaled |

Place the filled file at `data/external/inversion_historica_por_territorio.csv`. Until that file exists the scenario is skipped (`state=missing` in this run). A missing territory-year is allocated 0 and logged.

## Figures

- `article_outputs/fig_forecast_r2.pdf` — walk-forward and leave-one-territory-out R²
- `article_outputs/fig_forecast_by_year.pdf` — R² by target year
- `article_outputs/fig_abm_cv_paths.pdf` — CV paths for yearly IVC rank, fixed SHAP rank, equal allocation, and adaptive allocation
- `article_outputs/fig_sobol_cv.pdf` — total-order indices for the CV of IVC
- `article_outputs/fig_rain_forecast_r2.pdf` — walk-forward R² with and without rainfall, for the level and for the change
