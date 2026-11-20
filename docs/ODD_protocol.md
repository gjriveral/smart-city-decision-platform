# ODD+D description of the urban allocation model

This annex follows the ODD+D protocol for agent-based models of human decisions (Müller et al., 2013, doi:10.1016/j.envsoft.2013.06.003), which extends the ODD protocol (Grimm et al., 2010, doi:10.1016/j.ecolmodel.2010.08.019). Pattern checks follow the pattern-oriented modelling programme (Grimm et al., 2005, doi:10.1126/science.1116681). The description matches the implementation in the allocation model, including the calibration constraint that only the index response is identified.

## 1. Purpose

The model compares allocation rules for a fixed annual public-investment budget across the 21 territorial units of Medellín. It asks how the dispersion of a composite vulnerability index evolves over a ten-year horizon when the budget is focused on the most vulnerable units, split equally, or reweighted by recent improvement. It is a decision-support simulation, not a forecast of the municipal budget.

## 2. Entities, state variables and scales

One government agent allocates the annual budget. Each territory has a population agent and an infrastructure agent.

Population state: vulnerability index on [0, 1], wellbeing defined as one minus that index, and investment received in the current year. Infrastructure state: capacity on [0, 1], initialised as one minus the initial index and clipped to [0.1, 0.9], plus investment received.

Spatial scale: the 16 communes and five corregimientos named in the city configuration. Temporal scale: one step is one year. The scenario horizon is ten years. The calibration uses calendar years 2013–2021 and stops when the observed investment series stops. There is no within-year sequence and no network among territories.

## 3. Process overview and scheduling

Within each year the government reads the current index (and, under one rule, a fixed interpretability score), allocates the budget, and passes the allocation to both agents in the territory. The population agent then updates wellbeing and the index. The infrastructure agent then depreciates capacity and adds a recovery term. Records are written after both updates. The same order is used in every year. Agents do not act in parallel.

## 4. Design concepts

### 4.1 Theoretical and empirical background

The allocation rules are the author's instruments: a focused share on the highest-scoring territories, an equal split, and a heuristic that down-weights territories whose wellbeing has just improved. They are not estimated decision rules. The index transition is a proportional response to investment relative to the mean allocation. Only that response was estimated, and only from the observed index and the official executed investment. The public-capital elasticity in Bom and Ligthart (2014, doi:10.1111/joes.12037) is a different object and is not used as this response.

### 4.2 Individual decision-making

The government is the only decision-maker. Its objective is implicit in the selected rule, not in an explicit utility function. The focused rule ranks territories by the current index, or by a score fixed at the start, and assigns a share ρ of the budget to the first n territories. The equal rule assigns the same amount to every territory. The adaptive rule assigns the budget in proportion to the current index times a penalty for recent wellbeing gains. Population and infrastructure agents do not choose; they apply the accounting identities below. There is no adaptation of the rule during a run.

### 4.3 Learning

Agents do not learn. The adaptive penalty uses the recent change in wellbeing as a signal, but the weights of that penalty (the literals 5 and 0.01, and the two-year delay) stay fixed.

### 4.4 Individual sensing

The government observes the current index of every territory, the fixed interpretability score when that rule is selected, and the history of wellbeing it has itself produced. It does not observe a noisy signal. In the calibration the allocation is replaced by the observed investment, so the government does not choose.

### 4.5 Individual prediction

The government does not forecast. The focused ranking uses the current index, not an expected future index.

### 4.6 Interaction

Interaction is only through the budget constraint: one territory's allocation is the residual of the others. Territories do not imitate, trade, or influence each other's index directly.

### 4.7 Collectives

The priority set is a collective defined each year (or once, under the fixed ranking) as the n highest-scoring territories. It has no separate state.

### 4.8 Heterogeneity

Territories differ in the initial index and, in the calibration, in executed investment. They share β, γ, δ and θ. The government is unique.

### 4.9 Stochasticity

The index update can multiply the investment term by a shock, drawn once per territory-year from a normal distribution, recentred at one and truncated to [0.85, 1.15]. The calibration path sets the shock to one. The Monte Carlo uses a standard deviation of 0.05 and replication seeds 0 through 29. The structural minimizer is deterministic; seed 42 is reserved for the bootstrap and the rejection sample.

### 4.10 Observation

A run records, for every territory-year, the allocation, the index, wellbeing, capacity and cumulative investment. The calibration loss uses the index only. Pattern checks compare the cross-sectional dispersion and the rank order of the index with the observed cross-section.

## 5. Details

### 5.1 Implementation

The model is plain Python. It does not use an agent-based framework. One replication of the ten-year horizon is a few milliseconds.

### 5.2 Initialisation

Scenario runs start from the latest out-of-sample predicted index for each territory, except one robustness row that starts from the latest observed index. Capacity starts at one minus that index, clipped to [0.1, 0.9]. Wellbeing starts at one minus the index. The calibration path starts from the observed 2013 index and, for validation, from the observed 2019 index.

### 5.3 Input data

The scenario budget is the configured annual total and is held fixed. The calibration instead reads official executed investment by territory and year and converts it to a ratio to that year's mean. The composite index is the observed panel. No missing investment cell inside 2013–2020 was filled. Years without a closed machine-readable investment file were dropped rather than extrapolated.

### 5.4 Submodels

Index. IVC ← clip(IVC × (1 − γ × r × ε), 0, 1), with r the investment relative to the mean and ε the shock.

Wellbeing. The increment is β × r × (1 − wellbeing) × ε, then clipped to [0, 1]. This state does not feed back into the index.

Capacity. Capacity ← clip(capacity − δ + θ × r, 0, 1), applied as a decay followed by a recovery. It affects the index only when the saturation switch is on, through an effectiveness factor min(1, 1.5 − capacity). That switch is off in the calibrated scenarios, except in the rows that turn it on as a sensitivity.

Allocation. Described in section 4.2. Under calibration the allocation step is replaced by the observed investment ratio.
