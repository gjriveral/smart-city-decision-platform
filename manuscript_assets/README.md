# Manuscript assets

Files for Socio-Economic Planning Sciences (Elsevier).
Figure files contain no titles; the caption in this list is the figure caption.
CSV files are the unrounded code output. DOCX files are three-line Word tables
with R², RMSE and MAE rounded to 3 decimal places and percentages to 2.
Vector fonts are embedded (PDF/EPS font type 42; SVG text stored as paths).
PNG and TIFF are 600 dpi. TIFF uses LZW compression.
One-column width is 90 mm. Two-column width is 190 mm.
The smallest type is 7 pt DejaVu Sans.

## Figures

| Number | File stem | Width (mm) | Generator | Caption |
|---|---|---:|---|---|
| 1 | `fig_01_territories` | 190 | `scripts/regenerate_article.py::plot_territory_map` | The 21 study territories of Medellín. Every territory is numbered in the key. Numerals are drawn on the map only where the polygon is large enough to hold 7 pt type. Longitude is negative west of Greenwich. |
| 2 | `fig_02_forecast_r2` | 190 | `scripts/regenerate_article.py::plot_forecast` | Out-of-sample R² for IVC at t+1. Solid blue bars are the temporal walk-forward test. Hatched orange bars are leave-one-territory-out. R² is dimensionless. |
| 3 | `fig_03_forecast_by_year` | 90 | `scripts/regenerate_article.py::plot_forecast` | Walk-forward R² by target year for IVC at t+1. The horizontal line is R² = 0. |
| 4 | `fig_04_abm_cv` | 190 | `scripts/regenerate_article.py::plot_abm` | Coefficient of variation of IVC across the 21 territories over a 10-year simulation. Saturation is off and the noise term is zero. |
| 5 | `fig_05_rain_forecast` | 190 | `scripts/regenerate_article.py::plot_rain` | Walk-forward R² with rainfall unused (black), used as a predictor (hatched blue) or placed inside IVC (dotted orange). (a) Level of IVC at t+1. (b) Change in IVC from t to t+1. R² is dimensionless. |
| 6 | `fig_06_sobol_cv` | 190 | `scripts/regenerate_article.py::plot_sobol` | Total-order Sobol indices for the coefficient of variation of IVC at year 10. (a) Focused allocation. (b) Equal allocation. (c) Adaptive allocation. Whiskers are the SALib confidence intervals. The index is dimensionless. |

## Tables

| Number | CSV | DOCX | Generator | Title |
|---|---|---|---|---|
| 1 | `table_01_forecast_metrics.csv` | `table_01_forecast_metrics.docx` | `ml.lagged_forecast.run_lagged_suite` | Walk-forward forecast of IVC at t+1 from subindices at t. R², RMSE and MAE are computed on the pooled out-of-sample rows. |
| 2 | `table_02_loto_metrics.csv` | `table_02_loto_metrics.docx` | `ml.lagged_forecast.leave_one_territory_out` | Leave-one-territory-out R², RMSE and MAE for the same lagged IVC task. This is a spatial check, not a temporal forecast. |
| 3 | `table_03_forecast_by_year.csv` | `table_03_forecast_by_year.docx` | `ml.lagged_forecast.walk_forward` | Walk-forward R², RMSE and MAE by target year. |
| 4 | `table_04_rain_forecast.csv` | `table_04_rain_forecast.docx` | `ml.lagged_forecast.run_climate_comparison` | Walk-forward comparison of no rainfall, rainfall as a predictor, and rainfall inside IVC, for the level of IVC and for the annual change. |
| 5 | `table_05_abm_dispersion.csv` | `table_05_abm_dispersion.docx` | `scripts/regenerate_article.py::run_abm_grid` | Change in the coefficient of variation and in mean IVC after 10 simulated years under each allocation rule. |
| 6 | `table_06_abm_monte_carlo.csv` | `table_06_abm_monte_carlo.docx` | `scripts/regenerate_article.py::run_abm_grid` | Monte Carlo distribution of the percentage change in the coefficient of variation of IVC. Thirty replications, noise standard deviation 0.05. |
| 7 | `table_07_sobol_indices.csv` | `table_07_sobol_indices.docx` | `abm/05_sensitivity_analysis.py::run_sobol_scenarios` | First-order and total-order Sobol indices for mean IVC, the coefficient of variation and the Gini coefficient. |
| 8 | `table_08_rain_qc.csv` | `table_08_rain_qc.docx` | `scripts/regenerate_article.py::rain_qc_table` | Rainfall coverage and annual totals by calendar year after the station-level 80 percent rule. |
| S1 | `table_s1_loto_by_territory.csv` | `table_s1_loto_by_territory.docx` | `ml.lagged_forecast.leave_one_territory_out` | Leave-one-territory-out metrics for each held-out territory. |
| S2 | `table_s2_legacy_cv.csv` | `table_s2_legacy_cv.docx` | `ml.lagged_forecast.legacy_contemporaneous_cv` | Re-run of the legacy same-year XGBoost grid on the pinned package versions. |
| S3 | `table_s3_circularity.csv` | `table_s3_circularity.docx` | `ml.lagged_forecast.contemporaneous_linear_r2` | In-sample linear reconstruction of IVC from the six same-year subindices. |
