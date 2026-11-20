# Sobol update (N = 1024)

The previous global sensitivity run used Saltelli sampling with second-order terms off and **N = 128** (1,152 evaluations). That design is the file committed as `article_outputs/table_sobol_indices.csv` before this change. Its largest total-order index was **ST = 1.49** for the priority-set size on the focused scenario, coefficient of variation of IVC (the same defect as an ST reported near 1.30: the Jansen estimator is not bounded by 1, and N = 128 is too small).

This run uses the same design, with **N = 1024** (9,216 evaluations; `calc_second_order=False` in both `SALib.sample.sobol.sample` and `SALib.analyze.sobol.analyze`). The Saltelli matrix is a deterministic Sobol sequence. The ABM seed and the bootstrap seed are **42**. Each interval is the 2.5–97.5 percentile of **1,000** bootstrap resamples. Wall-clock time on this machine: **49.2 seconds**.

## CV of IVC, total-order index (previous → new)

| Parameter | Focused, N=128 | Focused, N=1024 (95% CI) | Equal, N=128 | Equal, N=1024 (95% CI) | Adaptive, N=128 | Adaptive, N=1024 (95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| Priority-set size | 1.489 | 0.946 (0.805–1.110) | 0.000 | 0.000 | 0.000 | 0.000 |
| Budget share rho | 0.257 | 0.143 (0.100–0.199) | 0.000 | 0.000 | 0.000 | 0.000 |
| IVC response (gamma) | 0.067 | 0.052 (0.034–0.076) | 0.064 | 0.065 (0.057–0.073) | 0.994 | 0.995 (0.936–1.061) |
| Noise SD | 0.001 | 0.001 (0.000–0.001) | 0.949 | 0.955 (0.892–1.025) | 0.005 | 0.005 (0.005–0.006) |
| Wellbeing, depreciation, recovery | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Check 0 ≤ S1 ≤ ST ≤ 1

All **63** indices (3 scenarios × 3 outputs × 7 parameters) have a 95% interval that meets [0, 1], and the first-order interval is not entirely above the total-order interval. N was not raised above 1024.

Seven point estimates still sit just outside 0 ≤ S1 ≤ ST ≤ 1. They are numerical noise on inactive or near-inactive inputs, and each interval covers a feasible value:

- Focused, rho, CV: S1 = −0.0037, interval (−0.031, 0.023), ST = 0.143.
- Focused, noise, mean IVC / CV / Gini: S1 is about −0.0004 to −0.0007, with intervals that include 0, and ST below 0.001.
- Adaptive, wellbeing rate: S1 exceeds ST by less than 0.00002 (both indices are about 10⁻⁴ or smaller). The two intervals overlap.

The focused priority-set ST is now 0.95. Its interval still reaches 1.11 because the estimator is unbounded. That tail is not evidence that the index is greater than 1.

## Replacement paragraph (English)

Global sensitivity of the agent-based allocation model was re-estimated with Saltelli sampling at N = 1024 (9,216 evaluations; second-order terms omitted so that the sample and the Sobol estimator use the same design). Ninety-five percent intervals are the 2.5 and 97.5 percentiles of 1,000 bootstrap resamples; the model and the bootstrap both use seed 42, and the run took 49 seconds. For the coefficient of variation of IVC, the priority-set size dominates the focused rule (ST = 0.95, 95% interval 0.80–1.11), noise dominates equal allocation (ST = 0.95, 0.89–1.03), and the IVC response dominates the adaptive rule (ST = 0.99, 0.94–1.06). The N = 128 estimate that placed the focused priority-set index at 1.49 was a small-sample artefact: every new interval intersects the feasible range 0 ≤ S1 ≤ ST ≤ 1.
