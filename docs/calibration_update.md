# Calibration of the allocation model

The index transition was calibrated. The other free inputs of the agent-based model were not, because the observed series do not identify them. No territory-year figure was imputed.

## Data

Observed index. `indices_comunas.csv` has a non-missing composite index for all 21 territories in every year from 2013 to 2024.

Executed public investment. Official year-end georeferenced project records of the Alcaldía de Medellín, summed to the territory. Citywide rows ("Toda la Ciudad") and institutional rows ("Fortalecimiento Institucional") are not a territory and were left out of the sum. A negative project adjustment was floored at zero after the sum; the count of territory-years floored is 0.

| Year | File | Cutoff stated in the file |
| --- | --- | --- |
| 2013–2015 | Workbooks "Inversión por Comunas y Corregimientos", sheet ALL | 31 December of that year |
| 2016–2018 | Workbooks of the same series, sheet ALL | 31 December of that year |
| 2019 | MEData, vigencia 2019, identifier 1-002-11-000278, CC BY-SA 4.0 | The corte field reads 31/12/20; the dataset coverage is calendar year 2019 |
| 2020 | Georreferenciación a Diciembre 31 de 2020 | 31 December 2020 |

The index continues through 2024. Closed-year machine-readable investment for 2021–2024 was not on the same publication page (later files are mid-year PDF cuts, not annual totals), so those index years are not in the loss.

Names were aligned to the 21 units in `config.yaml` (for example "Laureles" and "Laureles - Estadio" to Laureles Estadio, "Palmitas" to Corregimiento de San Sebastián de Palmitas). The stored series is in pesos, as published. The model uses the within-year ratio of a territory's pesos to the mean of the 21 territories, so the peso unit cancels.

## Sample split

Eight transitions can be built: investment in year t and the index in t and t+1, for t = 2013,…,2020. The first six (investment 2013–2018, scored index years 2014–2019) are the calibration sample. That is 75 percent; 70 percent of eight transitions is 5.6, which rounds to six, and a year is not split. The last two (investment 2019–2020, scored index years 2020–2021) are the validation sample. Gamma is not a function of the 2020 or 2021 index. The validation path starts from the observed 2019 index, which is the last calibration outcome used only as a state.

## Method

With saturation off, the coded transition is

IVC(i, t+1) = IVC(i, t) × (1 − γ × r(i, t)),

where r(i, t) is investment in territory i divided by that year's mean investment. Noise is off in the calibration path. The loss is the RMSE of this multi-step path against the observed index, starting from the observed 2013 cross-section and stepping through investment in 2013–2018. γ is the bounded minimizer on [−0.05, 0.40] (Brent). Seed 42 is used for the bootstrap and the rejection sample, not for the one-dimensional minimizer.

The 95 percent interval is the 2.5th and 97.5th percentiles of 1000 territory-block bootstrap replications. An Approximate Bayesian Computation rejection sample draws 4000 values from Uniform(−0.05, 0.40) and keeps the 5% with the smallest calibration RMSE (tolerance 0.051929). That sample is a check. The reported point is the RMSE minimizer, because the rejection interval depends on the tolerance.

## What was not calibrated

- Wellbeing rate (β = 0.15). It updates wellbeing and does not enter the index transition.
- Capacity decay (δ = 0.03) and infrastructure recovery (θ = 0.10). They update capacity, and capacity enters the index only when saturation is on. Saturation is off in the article setting, so these two parameters are not identified by the index path. They are not replaced by a depreciation rate from the public-capital literature: that rate is a geometric rate on a capital stock, and δ here is an absolute step on a unit-interval capacity. Bom and Ligthart (2014, doi:10.1111/joes.12037) put the average output elasticity of public capital near 0.106. That elasticity is not γ and is not used as a substitute.
- Budget share (ρ = 0.80), priority-set size (5) and the ranking rule. These are the policy levers of the scenarios. Feeding the observed allocation makes them unnecessary for the fit.
- Rainfall. The index transition has no rainfall coefficient. The climate subindex is an input to the composite index, not a dynamic sensitivity. No rainfall parameter is proposed, because inserting one would change the model.
- Idiosyncratic noise. It is zero on the calibration path and 0.05 in the Monte Carlo, as in the uncalibrated scenarios.

## Estimates

| Quantity | Value |
| --- | --- |
| γ, calibration path | 0.0144 |
| Bootstrap 95 percent interval | 0.0083 to 0.0218 |
| ABC accepted median | 0.0147 |
| ABC accepted 2.5–97.5 percent | 0.0052 to 0.0246 |
| Design value used before calibration | 0.05 |

Calibration path, index years 2014–2019 (126 territory-years). RMSE 0.049, MAE 0.033, R² 0.575. Persistence, which keeps the 2013 index unchanged, has RMSE 0.055, MAE 0.038, R² 0.473.

Validation path, index years 2020–2021 (42 territory-years), started from the observed 2019 index. RMSE 0.049, MAE 0.043, R² 0.556. Persistence, which keeps the 2019 index unchanged, has RMSE 0.042, MAE 0.035, R² 0.682.

Spearman rank correlation between the simulated and observed index: 2020 ρ = 0.979; 2021 ρ = 0.897. A persistence ranking that keeps the 2019 order has Spearman 0.978 in 2020 and 0.897 in 2021, the same as the simulated ranking at three decimals. The held-out rank pattern is the persistence of the index, not a reordering by relative investment. Coefficients of variation: 2020 observed 0.182, simulated 0.183; 2021 observed 0.193, simulated 0.182. The simulated dispersion stays near the 2019 value and does not follow the widening observed in 2021.

## Scenarios after calibration

The ten allocation specifications and the 30-replication Monte Carlo were rerun with γ = 0.0144. Every other input matches the uncalibrated run (ρ = 0.80, priority set of 5, saturation as in each row, seed 42 on the deterministic paths, seeds 0–29 and noise standard deviation 0.05 in the Monte Carlo).

Focused allocation with a yearly index rank reduced the coefficient of variation by 43.0 percent. The same specification with γ = 0.05 reduced it by 68.3 percent. Equal allocation left the coefficient of variation unchanged (the same result at the design value). Adaptive allocation changed it by 10.6 percent (uncalibrated 33.2). Under the fixed priority rank the coefficient of variation rose by 126.9 percent at the design value and fell by 14.0 percent at the calibrated response.

The Monte Carlo means of that percentage change are Focused, yearly index rank, equal shares 43.0 (SD 0.2); Focused, fixed priority rank 13.9 (SD 0.5); Equal allocation 0.0 (SD 0.2); Adaptive allocation 10.6 (SD 0.2).

## Paragraph for Methods

The response of the composite vulnerability index to relative public investment was calibrated on the agent-based transition used in the scenarios. Executed investment is the year-end georeferenced series published by the Alcaldía de Medellín for 2013–2020, summed to the 21 territories; citywide and institutional spending was excluded. The index is observed for every territory in 2013–2021. Six of the eight year-to-year transitions (investment in 2013–2018) formed the calibration sample and the last two were held out. The single identified parameter, the index response γ, minimises the root mean squared error of the deterministic multi-step path on [−0.05, 0.40]. A territory-block bootstrap of 1,000 replications, with seed 42, supplies the 95 percent interval. Wellbeing, capacity decay, infrastructure recovery, the allocation-rule levers and rainfall do not enter this transition when saturation is off, and they were left at their design values.

## Paragraph for Results

The calibrated index response is γ = 0.014 (95 percent bootstrap interval 0.008 to 0.022), against a design value of 0.05. On the calibration path the root mean squared error is 0.049, compared with 0.055 for a persistence path that never moves. On the two held-out years the calibrated path has root mean squared error 0.049 and the persistence path 0.042, so persistence is the more accurate held-out forecast. The held-out rank correlation matches persistence at three decimals. Rerunning the allocation rules at the calibrated response changes the ten-year reduction in the coefficient of variation under yearly focused allocation from 68.3 percent to 43.0 percent.
