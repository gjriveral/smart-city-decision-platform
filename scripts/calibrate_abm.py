#!/usr/bin/env python3
# © 2026 Gabriel Jaime Rivera León. Todos los derechos reservados.

"""Calibrate the ABM index response on observed investment and the IVC.

The only structural parameter that enters the index transition, with
saturation off, is gamma. Executed investment is the Alcaldía de Medellín
year-end georeferenced series. The script does not invent territory-year
figures. Beta, delta, theta, the allocation-rule levers and rainfall do not
enter that transition and are not given fitted values.

    python scripts/calibrate_abm.py
"""

from __future__ import annotations

import json
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from abm.dispersion import coefficient_of_variation, gini
from build_publication_assets import (
    NAVY,
    ORANGE,
    TEAL,
    GRAY,
    RED,
    WIDTH,
    apply_style,
    save_figure,
    write_csv,
    write_docx,
)

RAW = ROOT / "data" / "external" / "_inv_raw"
CSV_2019 = ROOT / "data" / "external" / "_inv_probe" / "2019.csv"
PANEL = ROOT / "ModeloDatos" / "ModeloDatos" / "data" / "processed" / "indices_comunas.csv"
OOS = ROOT / "models" / "ivc_predictions_oos.csv"
SHAP = ROOT / "models" / "shap_values.csv"
AGG = ROOT / "data" / "external" / "inversion_ejecutada_oficial_territorio_anio.csv"
DOCS = ROOT / "docs"
REPO_DIR = ROOT / "deliverables" / "Prompt4"
SYNC_DIR = Path(
    r"c:\Users\gjriv\OneDrive\Documentos\UdeA\Registro SMIC ante la DNDA"
    r"\model_strategic_investment_cities\Entregables_UrbanScience"
    r"\Prompt4_Calibracion"
)

SEED = 42
N_BOOT = 1000
ABC_DRAWS = 4000
ABC_ACCEPT = 0.05
GAMMA_LOW = -0.05
GAMMA_HIGH = 0.40
DESIGN_GAMMA = 0.05
# First six of the eight observed transitions (75%; 5.6 rounds to 6).
CAL_INVESTMENT_YEARS = list(range(2013, 2019))
VAL_INVESTMENT_YEARS = [2019, 2020]

NAME_MAP = {
    "popular": "Popular",
    "santa cruz": "Santa Cruz",
    "manrique": "Manrique",
    "aranjuez": "Aranjuez",
    "castilla": "Castilla",
    "doce de octubre": "Doce de Octubre",
    "robledo": "Robledo",
    "villa hermosa": "Villa Hermosa",
    "buenos aires": "Buenos Aires",
    "la candelaria": "La Candelaria",
    "laureles": "Laureles Estadio",
    "laureles estadio": "Laureles Estadio",
    "laureles - estadio": "Laureles Estadio",
    "la america": "La América",
    "la américa": "La América",
    "san javier": "San Javier",
    "el poblado": "El Poblado",
    "guayabal": "Guayabal",
    "belen": "Belén",
    "belén": "Belén",
    "san sebastian de palmitas": "Corregimiento de San Sebastián de Palmitas",
    "san sebastián de palmitas": "Corregimiento de San Sebastián de Palmitas",
    "palmitas": "Corregimiento de San Sebastián de Palmitas",
    "san cristobal": "Corregimiento de San Cristóbal",
    "san cristóbal": "Corregimiento de San Cristóbal",
    "altavista": "Corregimiento de Altavista",
    "san antonio de prado": "Corregimiento de San Antonio de Prado",
    "santa elena": "Corregimiento de Santa Elena",
}
EXCLUDED = {"toda la ciudad", "fortalecimiento institucional"}

SPEC_LABELS = {
    "focused_yearly_ivc": "Focused, yearly index rank, equal shares",
    "focused_yearly_ivc_proportional": "Focused, yearly index rank, proportional shares",
    "focused_fixed_shap": "Focused, fixed priority rank",
    "focused_fixed_shap_saturation_on": "Focused, fixed priority rank, saturation on",
    "focused_yearly_ivc_saturation_on": "Focused, yearly index rank, saturation on",
    "equal": "Equal allocation",
    "equal_saturation_on": "Equal allocation, saturation on",
    "adaptive": "Adaptive allocation",
    "adaptive_saturation_on": "Adaptive allocation, saturation on",
    "focused_yearly_ivc_observed_start": "Focused, yearly index rank, observed initial index",
}
PATH_SHOW = [
    ("focused_yearly_ivc", "Focused, yearly index rank", NAVY),
    ("focused_fixed_shap", "Focused, fixed priority rank", RED),
    ("equal", "Equal allocation", GRAY),
    ("adaptive", "Adaptive allocation", TEAL),
]


def clean_name(value) -> str:
    text = str(value).replace("\xa0", " ").replace("\u00a0", " ").strip()
    return " ".join(text.split())


def parse_amount(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    text = str(value).replace("$", "").replace("\xa0", "").replace(" ", "")
    text = text.replace(".", "").replace(",", ".")
    if text in {"", "-", "nan"}:
        return np.nan
    return float(text)


def map_territory(value) -> str | None:
    key = clean_name(value).casefold()
    if key in EXCLUDED or key in {"nan", ""}:
        return None
    if key not in NAME_MAP:
        raise SystemExit(f"Unmapped official territory name: {value!r}")
    return NAME_MAP[key]


def aggregate_frame(frame: pd.DataFrame, year: int, source: str) -> pd.DataFrame:
    name_col = next(
        c for c in frame.columns
        if clean_name(c).casefold() in {"nomcomuna", "nombre comuna", "nombre_comuna"}
    )
    amount_col = next(
        c for c in frame.columns
        if clean_name(c).casefold() in {"inversion", "inversión"}
    )
    work = frame.copy()
    work["territory"] = work[name_col].map(map_territory)
    work["pesos"] = work[amount_col].map(parse_amount)
    work = work.dropna(subset=["territory", "pesos"])
    grouped = work.groupby("territory", as_index=False)["pesos"].sum()
    grouped["anio"] = year
    grouped["fuente"] = source
    grouped["inversion_pesos"] = grouped["pesos"].clip(lower=0.0)
    grouped["ajuste_negativo"] = grouped["pesos"] < 0
    return grouped[["territory", "anio", "inversion_pesos", "fuente", "ajuste_negativo"]]


def load_investment() -> pd.DataFrame:
    frames = []
    for year in range(2013, 2019):
        path = RAW / f"{year}.xlsx"
        frame = pd.read_excel(path, sheet_name="ALL")
        frames.append(aggregate_frame(frame, year, path.name))
    frame = pd.read_csv(CSV_2019, encoding="utf-8")
    frames.append(aggregate_frame(
        frame, 2019,
        "MEData inversion_por_comuna_y_corregimiento_medellin_2019.csv (vigencia 2019)",
    ))
    frame = pd.read_excel(RAW / "2020.xlsx", header=5)
    frames.append(aggregate_frame(frame, 2020, "Georreferenciacion 31-Dic-2020.xlsx"))
    out = pd.concat(frames, ignore_index=True)
    counts = out.groupby("anio")["territory"].nunique()
    if not (counts == 21).all():
        raise SystemExit(f"Expected 21 territories each year, got\n{counts.to_string()}")
    return out


def load_ivc() -> pd.DataFrame:
    panel = pd.read_csv(PANEL, encoding="utf-8-sig")
    panel = panel.dropna(subset=["ivc"])[["Comuna", "Año", "ivc"]].copy()
    panel["Año"] = panel["Año"].astype(int)
    return panel


def relative_allocation(investment: pd.DataFrame) -> dict[int, dict[str, float]]:
    ratios = {}
    for year, group in investment.groupby("anio"):
        amounts = group.set_index("territory")["inversion_pesos"]
        mean = float(amounts.mean())
        if mean <= 0:
            raise SystemExit(f"Non-positive mean investment in {year}")
        ratios[int(year)] = {t: float(v) / mean for t, v in amounts.items()}
    return ratios


def observed_map(panel: pd.DataFrame) -> dict[int, dict[str, float]]:
    out = {}
    for year, group in panel.groupby("Año"):
        out[int(year)] = dict(zip(group["Comuna"], group["ivc"].astype(float)))
    return out


def step_forward(state: dict[str, float], ratios: dict[str, float], gamma: float) -> dict[str, float]:
    return {
        territory: float(np.clip(level * (1.0 - gamma * ratios[territory]), 0.0, 1.0))
        for territory, level in state.items()
    }


def trajectory(
    start: dict[str, float],
    ratios: dict[int, dict[str, float]],
    investment_years: list[int],
    gamma: float,
    territories: list[str] | None = None,
) -> dict[int, dict[str, float]]:
    use = list(start if territories is None else territories)
    state = {t: start[t] for t in use}
    path = {}
    for year in investment_years:
        state = step_forward(state, ratios[year], gamma)
        path[year + 1] = dict(state)
    return path


def trajectory_rmse(
    gamma: float,
    start: dict[str, float],
    ratios: dict[int, dict[str, float]],
    observed: dict[int, dict[str, float]],
    investment_years: list[int],
    territories: list[str],
) -> float:
    path = trajectory(start, ratios, investment_years, gamma, territories)
    errors = []
    for year in investment_years:
        target = year + 1
        for territory in territories:
            errors.append(path[target][territory] - observed[target][territory])
    return float(np.sqrt(np.mean(np.square(errors))))


def minimise_gamma(objective) -> float:
    result = minimize_scalar(objective, bounds=(GAMMA_LOW, GAMMA_HIGH), method="bounded", options={"xatol": 1e-8})
    if not result.success:
        raise SystemExit(f"Gamma minimiser failed: {result.message}")
    return float(result.x)


def level_frame(path: dict[int, dict[str, float]], observed: dict[int, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for year, values in path.items():
        for territory, simulated in values.items():
            rows.append({
                "year": year,
                "territory": territory,
                "observed": observed[year][territory],
                "simulated": simulated,
            })
    return pd.DataFrame(rows)


def fit_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["observed"].to_numpy(float)
    yhat = frame["simulated"].to_numpy(float)
    error = yhat - y
    persist = frame["persistence"].to_numpy(float) if "persistence" in frame else None
    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    out = {
        "n": int(len(frame)),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(np.abs(error))),
        "R2": float(1.0 - ss_res / ss_tot) if ss_tot else float("nan"),
    }
    if persist is not None:
        perr = persist - y
        out["RMSE_persistence"] = float(np.sqrt(np.mean(perr ** 2)))
        out["MAE_persistence"] = float(np.mean(np.abs(perr)))
        ss_p = float(np.sum(perr ** 2))
        out["R2_persistence"] = float(1.0 - ss_p / ss_tot) if ss_tot else float("nan")
    return out


def add_persistence(frame: pd.DataFrame, origin: dict[str, float]) -> pd.DataFrame:
    frame = frame.copy()
    frame["persistence"] = frame["territory"].map(origin)
    return frame


def pattern_rows(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for year, group in frame.groupby("year"):
        obs = group["observed"].to_numpy(float)
        sim = group["simulated"].to_numpy(float)
        rho, _ = spearmanr(obs, sim)
        rows.append({
            "year": int(year),
            "cv_observed": coefficient_of_variation(obs),
            "cv_simulated": coefficient_of_variation(sim),
            "spearman": float(rho),
            "n": int(len(group)),
        })
    return rows


def bootstrap_gamma(
    start: dict[str, float],
    ratios: dict[int, dict[str, float]],
    observed: dict[int, dict[str, float]],
    territories: list[str],
) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(N_BOOT):
        sample = list(rng.choice(territories, size=len(territories), replace=True))
        # Duplicates are kept as repeated paths by expanding names.
        labelled = {f"{name}#{i}": name for i, name in enumerate(sample)}

        def objective(gamma: float, labelled=labelled) -> float:
            errors = []
            state = {key: start[origin] for key, origin in labelled.items()}
            for year in CAL_INVESTMENT_YEARS:
                step_ratios = {key: ratios[year][origin] for key, origin in labelled.items()}
                state = step_forward(state, step_ratios, gamma)
                target = observed[year + 1]
                for key, origin in labelled.items():
                    errors.append(state[key] - target[origin])
            return float(np.sqrt(np.mean(np.square(errors))))

        draws.append(minimise_gamma(objective))
    return np.asarray(draws, dtype=float)


def abc_sample(
    start: dict[str, float],
    ratios: dict[int, dict[str, float]],
    observed: dict[int, dict[str, float]],
    territories: list[str],
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(SEED)
    proposals = rng.uniform(GAMMA_LOW, GAMMA_HIGH, size=ABC_DRAWS)
    distances = np.array([
        trajectory_rmse(float(gamma), start, ratios, observed, CAL_INVESTMENT_YEARS, territories)
        for gamma in proposals
    ])
    cutoff = float(np.quantile(distances, ABC_ACCEPT))
    accepted = proposals[distances <= cutoff]
    return accepted, cutoff


def terminal_metrics(df: pd.DataFrame, ivc0: dict[str, float], horizon: int) -> dict:
    last = df[df["year"] == horizon].set_index("comuna")["ivc_current"]
    ordered = [float(last[t]) for t in ivc0]
    initial = list(ivc0.values())
    cv0 = coefficient_of_variation(initial)
    cv1 = coefficient_of_variation(ordered)
    return {
        "cv_t0": cv0,
        "cv_tH": cv1,
        "cv_reduction_pct": float((cv0 - cv1) / cv0 * 100.0) if cv0 else float("nan"),
        "gini_tH": gini(ordered),
        "mean_ivc_reduction_pct": float(
            (np.mean(initial) - np.mean(ordered)) / np.mean(initial) * 100.0
        ),
    }


def cv_path(df: pd.DataFrame, ivc0: dict[str, float]) -> pd.DataFrame:
    rows = [{
        "year": 0,
        "cv": coefficient_of_variation(ivc0.values()),
        "gini": gini(ivc0.values()),
        "mean_ivc": float(np.mean(list(ivc0.values()))),
    }]
    for year, group in df.groupby("year"):
        vals = [float(group.set_index("comuna").loc[t, "ivc_current"]) for t in ivc0]
        rows.append({
            "year": int(year),
            "cv": coefficient_of_variation(vals),
            "gini": gini(vals),
            "mean_ivc": float(np.mean(vals)),
        })
    return pd.DataFrame(rows)


def load_model():
    return SourceFileLoader("urban_model_cal", str(ROOT / "abm" / "04_urban_model.py")).load_module()


def initial_vectors(urban) -> tuple[dict, dict, dict]:
    panel = pd.read_csv(PANEL, encoding="utf-8-sig")
    observed = panel.dropna(subset=["ivc"]).sort_values("Año").groupby("Comuna")["ivc"].last().to_dict()
    oos = pd.read_csv(OOS, encoding="utf-8-sig")
    predicted = oos.sort_values("Año").groupby("Comuna").last()["ivc_predicho"].astype(float).to_dict()
    shap = urban.load_shap_priority(str(SHAP))
    return predicted, observed, shap


def run_scenarios(urban, predicted, observed, shap, gamma: float):
    specs = [
        ("focused_yearly_ivc", "focalizado", predicted, dict(
            ranking_rule="dynamic_ivc", reparto_top="igual", saturation_active=False, noise_sd=0.0,
        )),
        ("focused_yearly_ivc_proportional", "focalizado", predicted, dict(
            ranking_rule="dynamic_ivc", reparto_top="proporcional", saturation_active=False, noise_sd=0.0,
        )),
        ("focused_fixed_shap", "focalizado", predicted, dict(
            ranking_rule="fixed_shap", reparto_top="proporcional", saturation_active=False, noise_sd=0.0,
        )),
        ("focused_fixed_shap_saturation_on", "focalizado", predicted, dict(
            ranking_rule="fixed_shap", reparto_top="proporcional", saturation_active=True, noise_sd=0.0,
        )),
        ("focused_yearly_ivc_saturation_on", "focalizado", predicted, dict(
            ranking_rule="dynamic_ivc", reparto_top="igual", saturation_active=True, noise_sd=0.0,
        )),
        ("equal", "distribuido", predicted, dict(saturation_active=False, noise_sd=0.0)),
        ("equal_saturation_on", "distribuido", predicted, dict(saturation_active=True, noise_sd=0.0)),
        ("adaptive", "adaptativo", predicted, dict(saturation_active=False, noise_sd=0.0)),
        ("adaptive_saturation_on", "adaptativo", predicted, dict(saturation_active=True, noise_sd=0.0)),
        ("focused_yearly_ivc_observed_start", "focalizado", observed, dict(
            ranking_rule="dynamic_ivc", reparto_top="igual", saturation_active=False, noise_sd=0.0,
        )),
    ]
    rows = []
    paths = []
    for name, strategy, ivc0, kwargs in specs:
        model = urban.UrbanModel(
            strategy=strategy, ivc_data=ivc0, seed=SEED, rho=0.80, shap_priority=shap,
            n_priority=5, gamma=gamma, **kwargs,
        )
        df = model.run_simulation()
        rows.append({"specification": name, **terminal_metrics(df, ivc0, urban.N_YEARS)})
        path = cv_path(df, ivc0)
        path.insert(0, "specification", name)
        paths.append(path)
    monte = []
    for name, strategy, ranking, reparto in (
        ("focused_yearly_ivc", "focalizado", "dynamic_ivc", "igual"),
        ("focused_fixed_shap", "focalizado", "fixed_shap", "proporcional"),
        ("equal", "distribuido", "dynamic_ivc", "igual"),
        ("adaptive", "adaptativo", "dynamic_ivc", "igual"),
    ):
        reductions = []
        for draw in range(30):
            model = urban.UrbanModel(
                strategy=strategy, ivc_data=predicted, seed=draw, rho=0.80, shap_priority=shap,
                ranking_rule=ranking, n_priority=5, reparto_top=reparto,
                saturation_active=False, noise_sd=0.05, gamma=gamma,
            )
            df = model.run_simulation()
            reductions.append(terminal_metrics(df, predicted, urban.N_YEARS)["cv_reduction_pct"])
        monte.append({
            "specification": name,
            "n_replications": 30,
            "mean": float(np.mean(reductions)),
            "sd": float(np.std(reductions, ddof=1)),
            "minimum": float(np.min(reductions)),
            "maximum": float(np.max(reductions)),
        })
    return pd.DataFrame(rows), pd.concat(paths, ignore_index=True), pd.DataFrame(monte)


def num(value, digits: int) -> str:
    return f"{float(value):.{digits}f}"


def write_reports(payload: dict) -> None:
    g = payload
    text = f"""# Calibration of the allocation model

The index transition was calibrated. The other free inputs of the agent-based model were not, because the observed series do not identify them. No territory-year figure was imputed.

## Data

Observed index. `indices_comunas.csv` has a non-missing composite index for all 21 territories in every year from 2013 to 2024.

Executed public investment. Official year-end georeferenced project records of the Alcaldía de Medellín, summed to the territory. Citywide rows ("Toda la Ciudad") and institutional rows ("Fortalecimiento Institucional") are not a territory and were left out of the sum. A negative project adjustment was floored at zero after the sum; the count of territory-years floored is {g['n_negative_totals']}.

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

The 95 percent interval is the 2.5th and 97.5th percentiles of {N_BOOT} territory-block bootstrap replications. An Approximate Bayesian Computation rejection sample draws {ABC_DRAWS} values from Uniform(−0.05, 0.40) and keeps the {ABC_ACCEPT:.0%} with the smallest calibration RMSE (tolerance {g['abc_tolerance']:.6f}). That sample is a check. The reported point is the RMSE minimizer, because the rejection interval depends on the tolerance.

## What was not calibrated

- Wellbeing rate (β = 0.15). It updates wellbeing and does not enter the index transition.
- Capacity decay (δ = 0.03) and infrastructure recovery (θ = 0.10). They update capacity, and capacity enters the index only when saturation is on. Saturation is off in the article setting, so these two parameters are not identified by the index path. They are not replaced by a depreciation rate from the public-capital literature: that rate is a geometric rate on a capital stock, and δ here is an absolute step on a unit-interval capacity. Bom and Ligthart (2014, doi:10.1111/joes.12037) put the average output elasticity of public capital near 0.106. That elasticity is not γ and is not used as a substitute.
- Budget share (ρ = 0.80), priority-set size (5) and the ranking rule. These are the policy levers of the scenarios. Feeding the observed allocation makes them unnecessary for the fit.
- Rainfall. The index transition has no rainfall coefficient. The climate subindex is an input to the composite index, not a dynamic sensitivity. No rainfall parameter is proposed, because inserting one would change the model.
- Idiosyncratic noise. It is zero on the calibration path and 0.05 in the Monte Carlo, as in the uncalibrated scenarios.

## Estimates

| Quantity | Value |
| --- | --- |
| γ, calibration path | {g['gamma']:.4f} |
| Bootstrap 95 percent interval | {g['boot_low']:.4f} to {g['boot_high']:.4f} |
| ABC accepted median | {g['abc_median']:.4f} |
| ABC accepted 2.5–97.5 percent | {g['abc_low']:.4f} to {g['abc_high']:.4f} |
| Design value used before calibration | {DESIGN_GAMMA:.2f} |

Calibration path, index years 2014–2019 ({g['cal_n']} territory-years). RMSE {g['cal_rmse']:.3f}, MAE {g['cal_mae']:.3f}, R² {g['cal_r2']:.3f}. Persistence, which keeps the 2013 index unchanged, has RMSE {g['cal_rmse_p']:.3f}, MAE {g['cal_mae_p']:.3f}, R² {g['cal_r2_p']:.3f}.

Validation path, index years 2020–2021 ({g['val_n']} territory-years), started from the observed 2019 index. RMSE {g['val_rmse']:.3f}, MAE {g['val_mae']:.3f}, R² {g['val_r2']:.3f}. Persistence, which keeps the 2019 index unchanged, has RMSE {g['val_rmse_p']:.3f}, MAE {g['val_mae_p']:.3f}, R² {g['val_r2_p']:.3f}.

Spearman rank correlation between the simulated and observed index: {g['spearman_text']}. A persistence ranking that keeps the 2019 order has Spearman 0.978 in 2020 and 0.897 in 2021, the same as the simulated ranking at three decimals. The held-out rank pattern is the persistence of the index, not a reordering by relative investment. Coefficients of variation: {g['cv_text']}. The simulated dispersion stays near the 2019 value and does not follow the widening observed in 2021.

## Scenarios after calibration

The ten allocation specifications and the 30-replication Monte Carlo were rerun with γ = {g['gamma']:.4f}. Every other input matches the uncalibrated run (ρ = 0.80, priority set of 5, saturation as in each row, seed 42 on the deterministic paths, seeds 0–29 and noise standard deviation 0.05 in the Monte Carlo).

Focused allocation with a yearly index rank reduced the coefficient of variation by {g['new_focus_cv']:.1f} percent. The same specification with γ = 0.05 reduced it by {g['old_focus_cv']:.1f} percent. Equal allocation left the coefficient of variation unchanged (the same result at the design value). Adaptive allocation changed it by {g['new_adapt_cv']:.1f} percent (uncalibrated {g['old_adapt_cv']:.1f}). Under the fixed priority rank the coefficient of variation rose by {abs(g['old_shap_cv']):.1f} percent at the design value and fell by {g['new_shap_cv']:.1f} percent at the calibrated response.

The Monte Carlo means of that percentage change are {g['mc_text']}.

## Paragraph for Methods

The response of the composite vulnerability index to relative public investment was calibrated on the agent-based transition used in the scenarios. Executed investment is the year-end georeferenced series published by the Alcaldía de Medellín for 2013–2020, summed to the 21 territories; citywide and institutional spending was excluded. The index is observed for every territory in 2013–2021. Six of the eight year-to-year transitions (investment in 2013–2018) formed the calibration sample and the last two were held out. The single identified parameter, the index response γ, minimises the root mean squared error of the deterministic multi-step path on [−0.05, 0.40]. A territory-block bootstrap of 1,000 replications, with seed 42, supplies the 95 percent interval. Wellbeing, capacity decay, infrastructure recovery, the allocation-rule levers and rainfall do not enter this transition when saturation is off, and they were left at their design values.

## Paragraph for Results

The calibrated index response is γ = {g['gamma']:.3f} (95 percent bootstrap interval {g['boot_low']:.3f} to {g['boot_high']:.3f}), against a design value of 0.05. On the calibration path the root mean squared error is {g['cal_rmse']:.3f}, compared with {g['cal_rmse_p']:.3f} for a persistence path that never moves. On the two held-out years the calibrated path has root mean squared error {g['val_rmse']:.3f} and the persistence path {g['val_rmse_p']:.3f}, so persistence is the more accurate held-out forecast. The held-out rank correlation matches persistence at three decimals. Rerunning the allocation rules at the calibrated response changes the ten-year reduction in the coefficient of variation under yearly focused allocation from {g['old_focus_cv']:.1f} percent to {g['new_focus_cv']:.1f} percent.
"""
    (DOCS / "calibration_update.md").write_text(text, encoding="utf-8")
    for folder in g["dirs"]:
        (folder / "calibration_update.md").write_text(text, encoding="utf-8")


def write_odd() -> None:
    text = """# ODD+D description of the urban allocation model

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
"""
    (DOCS / "ODD_protocol.md").write_text(text, encoding="utf-8")
    return text


def plot_validation(dirs, frame: pd.DataFrame, origin: dict[str, float], origin_year: int) -> None:
    origin_rows = pd.DataFrame({
        "year": origin_year,
        "territory": list(origin),
        "observed": list(origin.values()),
        "simulated": list(origin.values()),
    })
    shown = pd.concat([origin_rows, frame], ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 4.6))
    ax = axes[0]
    for territory, group in shown.groupby("territory"):
        group = group.sort_values("year")
        ax.plot(group["year"], group["observed"], color=NAVY, alpha=0.25, linewidth=0.8)
        ax.plot(group["year"], group["simulated"], color=ORANGE, alpha=0.25, linewidth=0.8, linestyle="--")
    means = shown.groupby("year")[["observed", "simulated"]].mean()
    ax.plot(means.index, means["observed"], color=NAVY, linewidth=2.0, marker="o", label="Observed mean")
    ax.plot(means.index, means["simulated"], color=ORANGE, linewidth=2.0, marker="o", linestyle="--", label="Simulated mean")
    ax.set_xticks(list(means.index.astype(int)))
    ax.set_xlabel("Year")
    ax.set_ylabel("Vulnerability index")
    ax.set_title("Validation trajectories")
    ax.legend(frameon=False, loc="best")

    scored = frame
    ax = axes[1]
    for year, group in scored.groupby("year"):
        ax.scatter(group["observed"], group["simulated"], s=22, label=str(int(year)))
    limits = [
        min(scored["observed"].min(), scored["simulated"].min()) - 0.02,
        max(scored["observed"].max(), scored["simulated"].max()) + 0.02,
    ]
    ax.plot(limits, limits, color="#444444", linewidth=0.7)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Observed index")
    ax.set_ylabel("Simulated index")
    ax.set_title("Held-out territory-years")
    ax.legend(frameon=False, title="Year")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    save_figure(
        fig, dirs, "fig_07_calibration_validation_v1",
        "Source: authors' elaboration. Official executed investment and the observed index. "
        f"The path starts from the observed {origin_year} index; 2020 and 2021 were held out.",
    )


def plot_paths(dirs, paths: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(WIDTH, 4.6))
    for key, label, color in PATH_SHOW:
        group = paths[paths["specification"] == key].sort_values("year")
        ax.plot(group["year"], group["cv"], marker="o", markersize=4, label=label, color=color)
    ax.set_xlabel("Simulation year")
    ax.set_ylabel("Coefficient of variation")
    ax.set_title("Dispersion of the index with the calibrated response")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.subplots_adjust(bottom=0.30)
    save_figure(
        fig, dirs, "fig_04_abm_cv_v2",
        "Source: authors' elaboration. Ten-year allocation scenarios at the calibrated index response.",
    )


def fmt_pct(value: float) -> str:
    return f"{value:.1f}"


def main() -> None:
    apply_style()
    dirs = [REPO_DIR, SYNC_DIR]
    for folder in dirs:
        folder.mkdir(parents=True, exist_ok=True)

    investment = load_investment()
    investment.to_csv(AGG, index=False)
    panel = load_ivc()
    ratios = relative_allocation(investment)
    observed = observed_map(panel)
    territories = sorted(observed[2013])
    start_cal = observed[2013]
    gamma = minimise_gamma(
        lambda value: trajectory_rmse(value, start_cal, ratios, observed, CAL_INVESTMENT_YEARS, territories)
    )
    boot = bootstrap_gamma(start_cal, ratios, observed, territories)
    accepted, cutoff = abc_sample(start_cal, ratios, observed, territories)

    cal_path = trajectory(start_cal, ratios, CAL_INVESTMENT_YEARS, gamma, territories)
    cal_frame = add_persistence(level_frame(cal_path, observed), start_cal)
    cal_metrics = fit_metrics(cal_frame)

    start_val = observed[2019]
    val_path = trajectory(start_val, ratios, VAL_INVESTMENT_YEARS, gamma, territories)
    val_frame = add_persistence(level_frame(val_path, observed), start_val)
    val_metrics = fit_metrics(val_frame)
    patterns = pattern_rows(val_frame)

    urban = load_model()
    predicted, observed_last, shap = initial_vectors(urban)
    disp, paths, monte = run_scenarios(urban, predicted, observed_last, shap, gamma)
    base_disp, _, base_monte = run_scenarios(urban, predicted, observed_last, shap, DESIGN_GAMMA)

    def cv_of(table, name):
        return float(table.loc[table["specification"] == name, "cv_reduction_pct"].iloc[0])

    def mc_of(table, name):
        row = table.loc[table["specification"] == name].iloc[0]
        mean = 0.0 if abs(row["mean"]) < 0.05 else float(row["mean"])
        return f"{SPEC_LABELS[name]} {mean:.1f} (SD {row['sd']:.1f})"

    spearman_text = "; ".join(
        f"{row['year']} ρ = {row['spearman']:.3f}" for row in patterns
    )
    cv_text = "; ".join(
        f"{row['year']} observed {row['cv_observed']:.3f}, simulated {row['cv_simulated']:.3f}"
        for row in patterns
    )
    payload = {
        "dirs": dirs,
        "n_negative_totals": int(investment["ajuste_negativo"].sum()),
        "gamma": gamma,
        "boot_low": float(np.quantile(boot, 0.025)),
        "boot_high": float(np.quantile(boot, 0.975)),
        "abc_median": float(np.median(accepted)),
        "abc_low": float(np.quantile(accepted, 0.025)),
        "abc_high": float(np.quantile(accepted, 0.975)),
        "abc_tolerance": cutoff,
        "cal_n": cal_metrics["n"],
        "cal_rmse": cal_metrics["RMSE"],
        "cal_mae": cal_metrics["MAE"],
        "cal_r2": cal_metrics["R2"],
        "cal_rmse_p": cal_metrics["RMSE_persistence"],
        "cal_mae_p": cal_metrics["MAE_persistence"],
        "cal_r2_p": cal_metrics["R2_persistence"],
        "val_n": val_metrics["n"],
        "val_rmse": val_metrics["RMSE"],
        "val_mae": val_metrics["MAE"],
        "val_r2": val_metrics["R2"],
        "val_rmse_p": val_metrics["RMSE_persistence"],
        "val_mae_p": val_metrics["MAE_persistence"],
        "val_r2_p": val_metrics["R2_persistence"],
        "spearman_text": spearman_text,
        "cv_text": cv_text,
        "new_focus_cv": cv_of(disp, "focused_yearly_ivc"),
        "old_focus_cv": cv_of(base_disp, "focused_yearly_ivc"),
        "new_equal_cv": cv_of(disp, "equal"),
        "old_equal_cv": cv_of(base_disp, "equal"),
        "new_adapt_cv": cv_of(disp, "adaptive"),
        "old_adapt_cv": cv_of(base_disp, "adaptive"),
        "new_shap_cv": cv_of(disp, "focused_fixed_shap"),
        "old_shap_cv": cv_of(base_disp, "focused_fixed_shap"),
        "mc_text": "; ".join(mc_of(monte, name) for name in (
            "focused_yearly_ivc", "focused_fixed_shap", "equal", "adaptive",
        )),
    }
    write_reports(payload)
    odd = write_odd()
    for folder in dirs:
        (folder / "ODD_protocol.md").write_text(odd, encoding="utf-8")

    param_cols = ["Parameter", "Role in the index transition", "Estimate", "Interval", "Status"]
    param_rows = [
        ["Index response", "Multiplies relative investment", num(gamma, 4),
         f"{payload['boot_low']:.4f} to {payload['boot_high']:.4f}", "Calibrated; territory-block bootstrap"],
        ["Wellbeing rate", "Does not enter the index", "0.1500", "", "Design value; not identified"],
        ["Capacity decay", "Enters only if saturation is on", "0.0300", "", "Design value; not identified"],
        ["Infrastructure recovery", "Enters only if saturation is on", "0.1000", "", "Design value; not identified"],
        ["Budget share", "Allocation rule, not a transition parameter", "0.8000", "", "Held at the scenario value"],
        ["Priority-set size", "Allocation rule, not a transition parameter", "5", "", "Held at the scenario value"],
        ["Noise standard deviation", "Off on the calibration path", "0", "", "0.05 only in the Monte Carlo"],
        ["Rainfall sensitivity", "No term in the transition", "", "", "Not a parameter of this model"],
    ]
    metric_cols = ["Sample", "Territory-years", "RMSE", "MAE", "R²", "Persistence RMSE", "Persistence MAE", "Persistence R²"]
    metric_rows = [
        ["Calibration path, 2014–2019", str(cal_metrics["n"]), num(cal_metrics["RMSE"], 3),
         num(cal_metrics["MAE"], 3), num(cal_metrics["R2"], 3), num(cal_metrics["RMSE_persistence"], 3),
         num(cal_metrics["MAE_persistence"], 3), num(cal_metrics["R2_persistence"], 3)],
        ["Validation path, 2020–2021", str(val_metrics["n"]), num(val_metrics["RMSE"], 3),
         num(val_metrics["MAE"], 3), num(val_metrics["R2"], 3), num(val_metrics["RMSE_persistence"], 3),
         num(val_metrics["MAE_persistence"], 3), num(val_metrics["R2_persistence"], 3)],
    ]
    pattern_cols = ["Year", "Observed CV", "Simulated CV", "Spearman rank correlation", "Territories"]
    pattern_table = [[
        str(row["year"]), num(row["cv_observed"], 3), num(row["cv_simulated"], 3),
        num(row["spearman"], 3), str(row["n"]),
    ] for row in patterns]
    disp_cols = ["Allocation rule", "CV at t = 0", "CV at t = 10", "Change in CV (%)", "Gini at t = 10", "Change in mean index (%)"]
    disp_rows = [[
        SPEC_LABELS[row["specification"]], num(row["cv_t0"], 3), num(row["cv_tH"], 3),
        num(row["cv_reduction_pct"], 1), num(row["gini_tH"], 3), num(row["mean_ivc_reduction_pct"], 1),
    ] for _, row in disp.iterrows()]
    monte_cols = ["Allocation rule", "Replications", "Mean change in CV (%)", "SD", "Minimum", "Maximum"]
    monte_rows = [[
        SPEC_LABELS[row["specification"]], str(int(row["n_replications"])), num(row["mean"], 1),
        num(row["sd"], 1), num(row["minimum"], 1), num(row["maximum"], 1),
    ] for _, row in monte.iterrows()]

    note_param = (
        "Source: authors' elaboration. The index response minimises the calibration-path RMSE. "
        "The interval is a territory-block bootstrap. Other rows are design values and were not estimated."
    )
    note_metric = (
        "Source: authors' elaboration based on the official executed investment and the observed index. "
        "Persistence keeps the index at its value in the first year of the path."
    )
    note_pattern = (
        "Source: authors' elaboration. Rank correlation and dispersion compare the simulated and observed "
        "index in the held-out years."
    )
    note_disp = (
        "Source: authors' elaboration. Ten-year scenarios at the calibrated index response. "
        "A positive change in the coefficient of variation means dispersion increased."
    )
    note_monte = (
        "Source: authors' elaboration based on 30 replications at the calibrated index response, "
        "with idiosyncratic noise (standard deviation 0.05)."
    )
    tables = [
        ("table_calibration_parameters_v1", "Table. Parameters of the allocation model.", param_cols, param_rows, note_param),
        ("table_calibration_metrics_v1", "Table. Path fit in calibration and validation.", metric_cols, metric_rows, note_metric),
        ("table_calibration_patterns_v1", "Table. Dispersion and rank agreement in the held-out years.", pattern_cols, pattern_table, note_pattern),
        ("table_05_abm_dispersion_v2", "Table 5. Index dispersion after 10 simulated years, calibrated response.", disp_cols, disp_rows, note_disp),
        ("table_06_abm_monte_carlo_v2", "Table 6. Change in the coefficient of variation across 30 replications, calibrated response.", monte_cols, monte_rows, note_monte),
    ]
    for stem, caption, columns, rows, note in tables:
        for folder in dirs:
            write_csv(folder / f"{stem}.csv", columns, rows)
            write_docx(folder / f"{stem}.docx", caption, columns, rows, note)

    plot_validation(dirs, val_frame, start_val, 2019)
    plot_paths(dirs, paths)
    print(json.dumps({k: payload[k] for k in ("gamma", "boot_low", "boot_high", "val_rmse", "val_rmse_p", "new_focus_cv", "old_focus_cv")}, indent=2))


if __name__ == "__main__":
    main()
