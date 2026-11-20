# Tables and figures for Urban Science

Prepared for Urban Science (MDPI) from the saved article results. The Sobol table and Figure 6 use the N = 1024 Saltelli sample already reported in `docs/sobol_update.md` (9,216 evaluations, 95% percentile bootstrap intervals, seed 42). The forecast and the allocation model were not refit.

File names in this delivery use the suffix `_v2`.

## Style applied to every item

- Typeface: Arial. Figure titles 10 pt, axis labels 9 pt, tick labels and legends 8 pt. Table body 9 pt, caption 10 pt, note 8 pt italic.
- Figure width: 180 mm. Vector PDF (TrueType fonts embedded) and TIFF at 600 dpi with LZW compression.
- Tables: editable DOCX and CSV. The academic note is the italic paragraph under the DOCX table. The CSV is the same grid without that paragraph.
- Decimals: R², RMSE, MAE, coefficient of variation, Gini, mean index and Sobol indices to 3 decimal places; percentage changes to 1 decimal place; rainfall to the nearest millimetre; hourly coverage to 1 decimal place, in percent; counts as integers.
- Notes name the data and the design. They do not cite file names, repository paths, code checks, or an earlier specification of the forecast.

The same-year reconstruction of the index is not in Tables 1–8 or S1–S3. Those numbered items are the out-of-sample forecast, the allocation scenarios, the rainfall comparison and the Sobol indices.

## Figure 1. Administrative territories

New map of the 21 units (communes and corregimientos) drawn from the official administrative boundaries. Each polygon is numbered with the official code; the names sit under the map. There was no territory figure in the previous output set.

Note: Source: authors' elaboration based on the official administrative boundaries of Medellín.

## Figure 2. Walk-forward and leave-one-territory-out R²

Same comparison as the previous unnumbered forecast figure. Bars now share the 180 mm width, the Arial sizes above, a common 0–1 axis, and an academic note. Persistence remains the highest walk-forward R² (0.821) and linear regression the highest leave-one-territory-out R² (0.799).

Note: Source: authors' elaboration based on the Medellín quality-of-life panel. Out-of-sample R² for the index at t+1 given subindices at t.

## Figure 3. Walk-forward R² by target year

Same four series as the previous year-by-year figure, restyled. The vertical axis is no longer clipped at zero, so years in which a model falls below the sample mean stay visible.

Note: Source: authors' elaboration based on the same walk-forward evaluation, by target year.

## Figure 4. Allocation paths

The four allocation rules from the previous path figure (focused yearly rank, focused fixed rank, equal, adaptive), restyled. Saturation variants stay in Table 4 and are not added to the figure.

Note: Source: authors' elaboration based on the agent-based allocation model (10-year horizon, 21 territories).

## Figure 5. Rainfall and the forecast

The previous two-panel rainfall figure, restyled. The title is now a description of the comparison. Both panels report walk-forward R², for the level of the index and for its change.

Note: Source: authors' elaboration based on the quality-of-life panel and the territorial rainfall series.

## Figure 6. Sobol indices

Redrawn from the N = 1024 sample, not from the earlier N = 128 figure. First-order and total-order indices for the coefficient of variation, with the 95% percentile bootstrap interval as error bars. For the focused rule, the priority-set size has S1 = 0.789 and ST = 0.946 (interval 0.805–1.110).

Note: Source: authors' elaboration. Saltelli sample, N = 1024; bars are 95% percentile bootstrap intervals.

## Table 1. Walk-forward forecast

Pooled metrics for persistence, linear regression, random forest and XGBoost (168 territory-years). Rounded to 3 decimal places. The previous file had unrounded floats and no note.

Note: Source: authors' elaboration based on the Medellín quality-of-life panel (2013–2024). Walk-forward R², RMSE and MAE for the index at t+1 given subindices at t, pooled over target years 2017–2024.

## Table 2. Walk-forward forecast by target year

One row per target year and model, from the previous year-by-year metrics, with the same rounding.

Note: Source: authors' elaboration based on the same walk-forward evaluation, reported by target year.

## Table 3. Leave-one-territory-out forecast

Pooled spatial check (231 territory-years). Linear regression 0.799, XGBoost 0.787, persistence 0.783, random forest 0.757.

Note: Source: authors' elaboration based on a leave-one-territory-out evaluation of the same lagged specification. Each territory is scored only in the folds that hold it out.

## Table 4. Allocation scenarios

Terminal coefficient of variation, its percentage change, terminal Gini and the percentage change in the mean index, for every saved specification. Percentage changes use 1 decimal place.

Note: Source: authors' elaboration based on the agent-based allocation model (21 territories, 10-year horizon, no idiosyncratic noise). A positive change in the coefficient of variation means dispersion increased.

## Table 5. Monte Carlo

Thirty replications at noise standard deviation 0.05. Mean, standard deviation, minimum and maximum of the percentage change in the coefficient of variation, at 1 decimal place.

Note: Source: authors' elaboration based on 30 replications of the same model with idiosyncratic noise (standard deviation 0.05).

## Table 6. Rainfall treatments

Walk-forward R², RMSE and MAE for three rainfall treatments and two targets (index level and change), 126 territory-years each.

Note: Source: authors' elaboration based on the quality-of-life panel and the territorial rainfall series. A negative R² means the forecast is worse than the mean of the evaluation sample.

## Table 7. Sobol indices for the coefficient of variation

Replaces the N = 128 indices. Twenty-one rows: three allocation rules and seven parameters. Columns are S1, ST and the 2.5% and 97.5% bootstrap percentiles, at 3 decimal places. Focused priority-set size: S1 = 0.789, ST = 0.946 (0.805–1.110).

Note: Source: authors' elaboration based on a Saltelli sample (N = 1024; 9,216 evaluations) of the allocation model. Intervals are the 2.5th and 97.5th percentiles of 1,000 bootstrap resamples. The outcome is the coefficient of variation of the index.

## Table 8. Rainfall coverage by year

Median hourly coverage (percent, 1 decimal), median observed rainfall (millimetres) and the median adjusted total for territory-years that meet the 80% coverage rule. The adjusted cell is blank in 2012 and 2026 because no territory meets that rule.

Note: Source: authors' elaboration based on the municipal rain-gauge network aggregated to the 21 territories. Adjusted totals are shown only for territory-years with at least 80% hourly coverage. 2012 is the first year of the network and 2026 is a partial year.

## Table S1. Leave-one-territory-out results by territory

R² and RMSE for each territory and model. A negative R² means the held-out territory is harder to predict than its own sample mean.

Note: Source: authors' elaboration. Leave-one-territory-out R² and RMSE by territory for the lagged forecast of the index.

## Table S2. Sobol indices for the mean index and the Gini coefficient

The other two outcomes from the same N = 1024 sample. Table 7 keeps the coefficient of variation, which is the outcome in Figure 6.

Note: Source: authors' elaboration based on the same Saltelli sample as Table 7. Outcomes are the mean index and the Gini coefficient.

## Table S3. Allocation path behind Figure 4

Coefficient of variation, Gini coefficient and mean index by simulation year for the four rules drawn in Figure 4.

Note: Source: authors' elaboration based on the allocation paths shown in Figure 4.
