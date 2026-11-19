#!/usr/bin/env python3
# © 2026 Gabriel Jaime Rivera León. Todos los derechos reservados.

"""
Regenerate the revised article tables and figures.

    python scripts/regenerate_article.py
    python scripts/regenerate_article.py --skip-legacy --sobol-n 8

Writes article_outputs/ (CSV tables, 300 dpi PNG, vector PDF) and prints
the numbers that METHODS_CHANGES.md reports. Labels are in English and the
index acronym is IVC.

Rainfall full mode is not executed here: the minute files are not in the
repository. The script runs the fixture processor only as a schema check and
refuses to put those rows into the forecast.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from abm.dispersion import coefficient_of_variation, gini
from abm.historical_allocation import load_historical_allocation
from bigdata import siata_territory_annual as siata
from ml.clima_hook import DEFAULT_WEIGHTS, SUBINDEX_COLS, composite_ivc
from elsevier_assets import (
    BLUE,
    DOUBLE_MM,
    GREEN,
    ORANGE,
    SINGLE_MM,
    VERMILLION,
    apply_elsevier_style,
    export_table,
    new_figure,
    panel_letter,
    save_figure,
    verify_figure,
)
from ml.lagged_forecast import legacy_contemporaneous_cv, run_climate_comparison, run_lagged_suite

OUT = ROOT / "article_outputs"
ASSETS = ROOT / "manuscript_assets"
FIG_DIRS = [OUT, ASSETS / "figures"]
PANEL = ROOT / "ModeloDatos/ModeloDatos/data/processed/indices_comunas.csv"
OOS = ROOT / "models/ivc_predictions_oos.csv"
SHAP = ROOT / "models/shap_values.csv"
SAVED_METRICS = ROOT / "models/cv_metrics.json"
ASSIGNMENT = ROOT / "data/external/asignacion_estacion_UT_oficial.json"

PUBLISHED_CV0 = 0.187532
PUBLISHED_CV10 = 0.058193
PUBLISHED_REDUCTION = 68.9692
PUBLISHED_R2 = 0.9481053938302593

MODEL_LABELS = {
    "persistence": "Persistence",
    "linear": "Linear regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
}
SCENARIO_LABELS = {
    "focalizado": "Focused",
    "distribuido": "Equal allocation",
    "adaptativo": "Adaptive allocation",
}


def _urban():
    spec = importlib.util.spec_from_file_location("urban_model_article", ROOT / "abm" / "04_urban_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.logger.setLevel(40)
    return mod


def _sensitivity():
    spec = importlib.util.spec_from_file_location(
        "sensitivity_article", ROOT / "abm" / "05_sensitivity_analysis.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def apply_style() -> None:
    apply_elsevier_style()


def terminal_metrics(df: pd.DataFrame, ivc0: dict[str, float], horizon: int) -> dict:
    ut = "comuna"
    last = df[df["year"] == horizon].set_index(ut)["ivc_current"]
    # Align to the initial vector's territories.
    ordered = [last[t] for t in ivc0]
    initial = list(ivc0.values())
    cv0 = coefficient_of_variation(initial)
    cv1 = coefficient_of_variation(ordered)
    return {
        "cv_t0": cv0,
        "cv_tH": cv1,
        "cv_reduction_pct": float((cv0 - cv1) / cv0 * 100.0) if cv0 else float("nan"),
        "gini_t0": gini(initial),
        "gini_tH": gini(ordered),
        "mean_ivc_t0": float(np.mean(initial)),
        "mean_ivc_tH": float(np.mean(ordered)),
        "mean_ivc_reduction_pct": float(
            (np.mean(initial) - np.mean(ordered)) / np.mean(initial) * 100.0
        ),
    }


def cv_path(df: pd.DataFrame, ivc0: dict[str, float]) -> pd.DataFrame:
    ut = "comuna"
    rows = [{
        "year": 0,
        "cv": coefficient_of_variation(ivc0.values()),
        "gini": gini(ivc0.values()),
        "mean_ivc": float(np.mean(list(ivc0.values()))),
    }]
    for year, g in df.groupby("year"):
        vals = g.set_index(ut).loc[list(ivc0)]["ivc_current"]
        rows.append({
            "year": int(year),
            "cv": coefficient_of_variation(vals),
            "gini": gini(vals),
            "mean_ivc": float(np.mean(vals)),
        })
    return pd.DataFrame(rows)


def load_initial_vectors() -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    panel = pd.read_csv(PANEL, encoding="utf-8-sig")
    observed = (
        panel.dropna(subset=["ivc"]).sort_values("Año").groupby("Comuna")["ivc"].last().to_dict()
    )
    oos = pd.read_csv(OOS, encoding="utf-8-sig")
    latest = oos.sort_values("Año").groupby("Comuna").last()
    predicted = latest["ivc_predicho"].astype(float).to_dict()
    urban = _urban()
    shap = urban.load_shap_priority(str(SHAP))
    return predicted, observed, shap


def hypothesis_block(panel: pd.DataFrame) -> dict:
    d = panel.dropna(subset=["ivc", *SUBINDEX_COLS])
    coded = composite_ivc(d, DEFAULT_WEIGHTS).to_numpy()
    y = d["ivc"].to_numpy(float)
    uniform_w = {k: 1.0 for k in DEFAULT_WEIGHTS}
    uniform = composite_ivc(d, uniform_w).to_numpy()
    # Global versus train-only scaling of the calendar year, the leak in
    # ml/03_xgboost_model.py (año_normalizado uses every year, including the test).
    years = np.arange(2013, 2025)
    global_scale = (years - years.min()) / (years.max() - years.min())
    train_years = years[years <= 2021]
    train_scale_denom = train_years.max() - train_years.min()
    year_2020_global = float((2020 - 2013) / (2024 - 2013))
    year_2020_train = float((2020 - 2013) / train_scale_denom)
    assignment = siata.load_assignment(ASSIGNMENT)
    cfg_territories = [
        {"nombre": t.nombre, "lat": t.lat, "lon": t.lon}
        for t in __import__("config_loader", fromlist=["get_config"]).get_config().territorios
    ]
    coverage = siata.assignment_coverage(assignment, cfg_territories)
    saved = json.loads(SAVED_METRICS.read_text(encoding="utf-8"))
    return {
        "n_rows": int(len(panel)),
        "n_complete_ivc_rows": int(len(d)),
        "years_complete": [int(v) for v in sorted(d["Año"].unique())],
        "n_territories": int(d["Comuna"].nunique()),
        "climate_unique_values": [float(v) for v in sorted(panel["idx_riesgo_clima"].unique())],
        "coded_weight_max_abs_error": float(np.max(np.abs(coded - y))),
        "uniform_weight_max_abs_error": float(np.max(np.abs(uniform - y))),
        "coded_weights_normalized": {
            k: float(v / sum(DEFAULT_WEIGHTS.values())) for k, v in DEFAULT_WEIGHTS.items()
        },
        "year_2020_scaled_with_2024_in_denominator": year_2020_global,
        "year_2020_scaled_on_2013_2021_only": year_2020_train,
        "saved_mean_R2": float(saved["mean_R2"]),
        "saved_best_params_match_last_fold": saved["best_params"] == {
            "colsample_bytree": 0.7,
            "gamma": 0.0,
            "learning_rate": 0.1,
            "max_depth": 3,
            "min_child_weight": 3,
            "n_estimators": 200,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "subsample": 0.7,
        },
        "territories_without_official_station": coverage["territorios_sin_estacion_idw"],
        "stations_assigned": coverage["n_estaciones_asignadas"],
        "global_year_scale_check": global_scale.tolist(),
    }


def run_abm_grid(urban, predicted, observed, shap) -> tuple[pd.DataFrame, pd.DataFrame]:
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
            strategy=strategy,
            ivc_data=ivc0,
            seed=42,
            rho=0.80,
            shap_priority=shap,
            n_priority=5,
            **kwargs,
        )
        df = model.run_simulation()
        metrics = terminal_metrics(df, ivc0, urban.N_YEARS)
        rows.append({"specification": name, "strategy": strategy, **kwargs, **metrics})
        path = cv_path(df, ivc0)
        path.insert(0, "specification", name)
        paths.append(path)
    # Monte Carlo around the configured noise, saturation off, both ranking rules.
    rng_rows = []
    for name, ranking, reparto in (
        ("focused_yearly_ivc", "dynamic_ivc", "igual"),
        ("focused_fixed_shap", "fixed_shap", "proporcional"),
        ("equal", "dynamic_ivc", "igual"),
        ("adaptive", "dynamic_ivc", "igual"),
    ):
        strategy = {"focused_yearly_ivc": "focalizado", "focused_fixed_shap": "focalizado",
                    "equal": "distribuido", "adaptive": "adaptativo"}[name]
        reductions = []
        for seed in range(30):
            model = urban.UrbanModel(
                strategy=strategy, ivc_data=predicted, seed=seed, rho=0.80,
                shap_priority=shap, ranking_rule=ranking, n_priority=5,
                reparto_top=reparto, saturation_active=False, noise_sd=0.05,
            )
            df = model.run_simulation()
            reductions.append(terminal_metrics(df, predicted, urban.N_YEARS)["cv_reduction_pct"])
        rng_rows.append({
            "specification": name,
            "noise_sd": 0.05,
            "n_replications": 30,
            "cv_reduction_pct_mean": float(np.mean(reductions)),
            "cv_reduction_pct_std": float(np.std(reductions, ddof=1)),
            "cv_reduction_pct_min": float(np.min(reductions)),
            "cv_reduction_pct_max": float(np.max(reductions)),
        })
    table = pd.DataFrame(rows)
    pd.DataFrame(rng_rows).to_csv(OUT / "table_abm_monte_carlo.csv", index=False)
    return table, pd.concat(paths, ignore_index=True)


def plot_forecast(pooled: pd.DataFrame, by_year: pd.DataFrame, loto: pd.DataFrame) -> None:
    order = ["persistence", "linear", "random_forest", "xgboost"]
    fig = new_figure(DOUBLE_MM, 80)
    ax = fig.add_subplot(111)
    x = np.arange(len(order))
    width = 0.36
    temp = pooled.set_index("model").loc[order, "R2"]
    spat = loto.set_index("model").loc[order, "R2"]
    ax.bar(x - width / 2, temp, width, label="Temporal walk-forward", color=BLUE, edgecolor="black", linewidth=0.3)
    ax.bar(
        x + width / 2, spat, width, label="Leave-one-territory-out",
        color=ORANGE, edgecolor="black", linewidth=0.3, hatch="///",
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in order])
    ax.set_ylabel("Out-of-sample R² (dimensionless)")
    ax.set_xlabel("Model")
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, "fig_02_forecast_r2", FIG_DIRS)

    fig = new_figure(SINGLE_MM, 95)
    ax = fig.add_subplot(111)
    styles = [
        ("persistence", BLUE, "o"),
        ("linear", VERMILLION, "s"),
        ("random_forest", GREEN, "D"),
        ("xgboost", "black", "^"),
    ]
    for model, color, marker in styles:
        g = by_year[by_year["model"] == model].sort_values("anio_objetivo")
        ax.plot(
            g["anio_objetivo"], g["R2"], marker=marker, label=MODEL_LABELS[model],
            color=color, markerfacecolor="white", markeredgewidth=0.6,
        )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Target year")
    ax.set_ylabel("R² in the target year (dimensionless)")
    fig.legend(frameon=False, ncol=2, loc="outside lower center")
    save_figure(fig, "fig_03_forecast_by_year", FIG_DIRS)


def plot_abm(paths: pd.DataFrame) -> None:
    show = [
        ("focused_yearly_ivc", "Focused, yearly IVC rank", BLUE, "o"),
        ("focused_fixed_shap", "Focused, fixed SHAP rank", VERMILLION, "s"),
        ("equal", "Equal allocation", "black", "D"),
        ("adaptive", "Adaptive allocation", GREEN, "^"),
    ]
    fig = new_figure(DOUBLE_MM, 85)
    ax = fig.add_subplot(111)
    for key, label, color, marker in show:
        g = paths[paths["specification"] == key]
        ax.plot(
            g["year"], g["cv"], marker=marker, label=label, color=color,
            markerfacecolor="white", markeredgewidth=0.6,
        )
    ax.set_xlabel("Simulation year")
    ax.set_ylabel("Coefficient of variation of IVC (dimensionless)")
    fig.legend(frameon=False, ncol=2, loc="outside lower center")
    save_figure(fig, "fig_04_abm_cv", FIG_DIRS)


SOBOL_LABELS = {
    "beta": "Wellbeing rate",
    "gamma": "IVC response",
    "delta": "Depreciation",
    "theta": "Infrastructure recovery",
    "rho": "Budget share rho",
    "n_prioritarios": "Priority-set size",
    "noise_sd": "Noise SD",
}


def plot_rain(metrics: pd.DataFrame) -> None:
    order = ["persistence", "linear", "random_forest", "xgboost"]
    usos = ["ninguno", "predictor", "subindice"]
    uso_labels = {
        "ninguno": "No rainfall",
        "predictor": "Rainfall as predictor",
        "subindice": "Rainfall inside IVC",
    }
    colors = {
        "ninguno": ("black", ""),
        "predictor": (BLUE, "///"),
        "subindice": (ORANGE, "..."),
    }
    fig = new_figure(DOUBLE_MM, 100)
    axes = [fig.add_subplot(121), fig.add_subplot(122)]
    for ax, target, letter, ylabel in (
        (axes[0], "level", "(a)", "Walk-forward R² of IVC at t+1 (dimensionless)"),
        (axes[1], "change", "(b)", "Walk-forward R² of the IVC change (dimensionless)"),
    ):
        sub = metrics[metrics["target"] == target]
        x = np.arange(len(order))
        width = 0.25
        for i, uso in enumerate(usos):
            vals = [
                float(sub[(sub["clima_uso"] == uso) & (sub["model"] == m)]["R2"].iloc[0])
                for m in order
            ]
            color, hatch = colors[uso]
            ax.bar(
                x + (i - 1) * width, vals, width,
                label=uso_labels[uso] if ax is axes[0] else None,
                color=color, edgecolor="black", linewidth=0.3, hatch=hatch,
            )
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in order], rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Model")
        panel_letter(ax, letter)
    fig.legend(frameon=False, ncol=3, loc="outside lower center")
    save_figure(fig, "fig_05_rain_forecast", FIG_DIRS)


def rain_qc_table(raw: pd.DataFrame, adjusted: pd.DataFrame) -> pd.DataFrame:
    """One row per year: coverage, method mix, and observed versus adjusted totals."""
    rows = []
    for year, g in raw.groupby("anio"):
        interior = g[g["metodo_agregacion"] == "estaciones_interior"]
        adj = adjusted[adjusted["anio"] == year]
        eligible = adj[adj["elegible"]]
        rows.append({
            "year": int(year),
            "n": int(len(g)),
            "n_interior": int(len(interior)),
            "n_idw": int((g["metodo_agregacion"] == "idw").sum()),
            "coverage_median": float(g["cobertura_horaria_media"].median()),
            "observed_mm_median": float(g["precip_anual_mm"].median()),
            "interior_observed_mm_median": float(interior["precip_anual_mm"].median()) if len(interior) else float("nan"),
            "eligible_n": int(len(eligible)),
            "adjusted_mm_median": float(eligible["precip_anual_ajustada_mm"].median()) if len(eligible) else float("nan"),
        })
    return pd.DataFrame(rows)


def plot_sobol(indices: pd.DataFrame) -> None:
    fig = new_figure(DOUBLE_MM, 95)
    axes = [fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)]
    sub = indices[indices["output"] == "cv"].copy()
    order = list(SOBOL_LABELS)
    letters = ("(a)", "(b)", "(c)")
    for ax, scenario, letter in zip(axes, ("focalizado", "distribuido", "adaptativo"), letters):
        g = sub[sub["scenario"] == scenario].set_index("parameter").loc[order]
        ax.barh(
            list(SOBOL_LABELS.values()), g["ST"], xerr=g["ST_conf"],
            color=BLUE, edgecolor="black", linewidth=0.3, capsize=2, ecolor="black",
        )
        ax.axvline(0, color="black", linewidth=0.5)
        panel_letter(ax, letter)
    axes[0].set_ylabel("Parameter")
    axes[1].set_xlabel("Total-order Sobol index (dimensionless)")
    save_figure(fig, "fig_06_sobol_cv", FIG_DIRS)


def plot_territory_map() -> None:
    """Numbered boundaries of the 21 study territories. Names are listed beside the map."""
    from matplotlib.patches import Polygon as MplPolygon

    from bigdata.siata_territory_annual import load_territory_polygons
    from config_loader import get_config

    cfg = get_config()
    polygons = {p["nombre"]: p for p in load_territory_polygons(ROOT / "data" / "geo" / "comunas.geojson")}
    names = [t.nombre for t in cfg.territorios]
    candidates = []
    fig = new_figure(DOUBLE_MM, 120)
    ax = fig.add_subplot(121)
    key = fig.add_subplot(122)
    key.axis("off")
    for i, name in enumerate(names, start=1):
        poly = polygons[name]
        ring = max(poly["rings"], key=len)
        ax.add_patch(MplPolygon(
            ring, closed=True, facecolor="#F2F2F2", edgecolor="black", linewidth=0.4, zorder=1,
        ))
        for extra in poly["rings"]:
            if extra is ring:
                continue
            ax.add_patch(MplPolygon(
                extra, closed=True, facecolor="#F2F2F2", edgecolor="black", linewidth=0.4, zorder=1,
            ))
        xs = [pt[0] for pt in ring]
        ys = [pt[1] for pt in ring]
        candidates.append((
            max(max(xs) - min(xs), max(ys) - min(ys)),
            i,
            float(np.mean(xs)),
            float(np.mean(ys)),
        ))
    # Label the largest polygons first and skip a numeral that would sit on another.
    placed = []
    for _span, i, cx, cy in sorted(candidates, reverse=True):
        if any((cx - px) ** 2 + (cy - py) ** 2 < 0.0012 for px, py in placed):
            continue
        ax.text(cx, cy, str(i), ha="center", va="center", fontsize=7, zorder=2)
        placed.append((cx, cy))
    ax.set_aspect(1.0 / np.cos(np.deg2rad(6.25)))
    ax.autoscale_view()
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    lines = [f"{i}  {name}" for i, name in enumerate(names, start=1)]
    key.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=7, linespacing=1.15)
    save_figure(fig, "fig_01_territories", FIG_DIRS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate revised article outputs")
    parser.add_argument("--skip-legacy", action="store_true")
    parser.add_argument("--sobol-n", type=int, default=128)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    apply_style()
    started = time.time()
    panel = pd.read_csv(PANEL, encoding="utf-8-sig")

    print("Territory map...", flush=True)
    plot_territory_map()

    print("Hypothesis checks...", flush=True)
    checks = hypothesis_block(panel)
    (OUT / "hypothesis_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")

    print("Lagged forecast (temporal CV, nested tuning, LOTO)...", flush=True)
    suite = run_lagged_suite(panel, uso="ninguno")
    suite["temporal_pooled"].to_csv(OUT / "table_forecast_metrics.csv", index=False)
    suite["temporal_by_year"].to_csv(OUT / "table_forecast_by_year.csv", index=False)
    suite["loto_pooled"].to_csv(OUT / "table_loto_metrics.csv", index=False)
    suite["loto_by_territory"].to_csv(OUT / "table_loto_by_territory.csv", index=False)
    suite["temporal_predictions"].to_csv(OUT / "table_forecast_predictions.csv", index=False)
    plot_forecast(suite["temporal_pooled"], suite["temporal_by_year"], suite["loto_pooled"])
    circ = suite["contemporaneous_linear"]
    pd.DataFrame([circ]).to_csv(OUT / "table_circularity_linear.csv", index=False)

    print(
        "Rainfall quality check and climate comparison "
        "(station-level 80% coverage, 2012 and 2026 excluded, station 352 in Popular)...",
        flush=True,
    )
    rain_raw = pd.read_csv(ROOT / "data/processed/siata_territorio_anual.csv")
    # Station-level aggregation already applied the 80% rule. Re-annotating
    # would replace those values with a territory-mean scaling.
    if "ajuste" in rain_raw.columns and (rain_raw["ajuste"] == "estacion").all():
        rain = rain_raw
    else:
        rain = siata.annotate_observed_table(rain_raw)
        rain.to_csv(ROOT / "data/processed/siata_territorio_anual.csv", index=False)
    qc = rain_qc_table(rain_raw, rain)
    qc.to_csv(OUT / "table_rain_qc_by_year.csv", index=False)
    usable = rain[rain["elegible"]].copy()
    climate = run_climate_comparison(panel, usable)
    climate["metrics"].to_csv(OUT / "table_rain_forecast.csv", index=False)
    climate["by_year"].to_csv(OUT / "table_rain_forecast_by_year.csv", index=False)
    plot_rain(climate["metrics"])

    print("SIATA fixture (schema only; not used as a predictor)...", flush=True)
    fixture_out = OUT / "siata_territorio_anual_fixture.csv"
    siata.main(["--mode", "fixture", "--out", str(fixture_out)])
    hist_map, hist_status = load_historical_allocation(
        ROOT / "data/external/inversion_historica_por_territorio.csv"
    )

    print("ABM ranking, saturation, and dispersion...", flush=True)
    urban = _urban()
    predicted, observed, shap = load_initial_vectors()
    abm_table, paths = run_abm_grid(urban, predicted, observed, shap)
    abm_table.to_csv(OUT / "table_abm_dispersion.csv", index=False)
    paths.to_csv(OUT / "table_abm_cv_path.csv", index=False)
    plot_abm(paths)

    print(f"Sobol (N={args.sobol_n}) on three scenarios, outputs mean IVC / CV / Gini...", flush=True)
    sens = _sensitivity()
    sobol_idx, sobol_detail = sens.run_sobol_scenarios(
        predicted, shap, n_sobol=args.sobol_n, seed=42,
        saturation_active=False, ranking_rule="dynamic_ivc", reparto_top="igual",
    )
    sobol_idx.to_csv(OUT / "table_sobol_indices.csv", index=False)
    sobol_detail.to_csv(OUT / "table_sobol_samples.csv", index=False)
    plot_sobol(sobol_idx)

    legacy = None
    if not args.skip_legacy:
        print("Legacy same-year XGBoost grid on this install...", flush=True)
        legacy = legacy_contemporaneous_cv(panel)
        # params are not all JSON-native once they leave sklearn; coerce via default=str
        (OUT / "table_legacy_cv.json").write_text(
            json.dumps(legacy, indent=2, default=str), encoding="utf-8"
        )
        pd.DataFrame(legacy["folds"]).drop(columns=["best_params"]).to_csv(
            OUT / "table_legacy_cv_folds.csv", index=False
        )

    import sklearn
    import xgboost as xgb

    summary = {
        "elapsed_s": round(time.time() - started, 1),
        "python": sys.version.split()[0],
        "xgboost": xgb.__version__,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "published_mean_R2": PUBLISHED_R2,
        "published_cv_t0": PUBLISHED_CV0,
        "published_cv_t10": PUBLISHED_CV10,
        "published_cv_reduction_pct": PUBLISHED_REDUCTION,
        "hypothesis": checks,
        "contemporaneous_linear": circ,
        "forecast_pooled": suite["temporal_pooled"].to_dict(orient="records"),
        "loto_pooled": suite["loto_pooled"].to_dict(orient="records"),
        "selected_params_by_fold": suite["selected_params_by_fold"],
        "abm": abm_table.to_dict(orient="records"),
        "historical_investment": hist_status,
        "historical_rows_loaded": len(hist_map),
        "legacy_mean_R2": None if legacy is None else legacy["mean_R2"],
        "legacy_best_test_fold": None if legacy is None else legacy["best_test_fold"],
        "legacy_hyperparameters_from_best_test_fold": None if legacy is None else legacy["hyperparameters_from_best_test_fold"],
        "sobol_n": args.sobol_n,
        "predictor_columns": suite["predictor_columns"],
        "clima_uso": "ninguno",
        "rain_n_pairs": climate["n_pairs"],
        "rain_target_years": climate["target_years"],
        "rain_forecast": climate["metrics"].to_dict(orient="records"),
        "rain_qc_by_year": qc.to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    export_manuscript_assets()
    print(f"Done in {summary['elapsed_s']}s. Outputs in {OUT} and {ASSETS}")


FIGURES = [
    ("1", "fig_01_territories", DOUBLE_MM, "plot_territory_map",
     "The 21 study territories of Medellín. Every territory is numbered in the key. Numerals are drawn on the map only where the polygon is large enough to hold 7 pt type. Longitude is negative west of Greenwich."),
    ("2", "fig_02_forecast_r2", DOUBLE_MM, "plot_forecast",
     "Out-of-sample R² for IVC at t+1. Solid blue bars are the temporal walk-forward test. Hatched orange bars are leave-one-territory-out. R² is dimensionless."),
    ("3", "fig_03_forecast_by_year", SINGLE_MM, "plot_forecast",
     "Walk-forward R² by target year for IVC at t+1. The horizontal line is R² = 0."),
    ("4", "fig_04_abm_cv", DOUBLE_MM, "plot_abm",
     "Coefficient of variation of IVC across the 21 territories over a 10-year simulation. Saturation is off and the noise term is zero."),
    ("5", "fig_05_rain_forecast", DOUBLE_MM, "plot_rain",
     "Walk-forward R² with rainfall unused (black), used as a predictor (hatched blue) or placed inside IVC (dotted orange). (a) Level of IVC at t+1. (b) Change in IVC from t to t+1. R² is dimensionless."),
    ("6", "fig_06_sobol_cv", DOUBLE_MM, "plot_sobol",
     "Total-order Sobol indices for the coefficient of variation of IVC at year 10. (a) Focused allocation. (b) Equal allocation. (c) Adaptive allocation. Whiskers are the SALib confidence intervals. The index is dimensionless."),
]

TABLES = [
    ("1", "table_forecast_metrics.csv", "table_01_forecast_metrics",
     "Walk-forward forecast of IVC at t+1 from subindices at t. R², RMSE and MAE are computed on the pooled out-of-sample rows.",
     "ml.lagged_forecast.run_lagged_suite"),
    ("2", "table_loto_metrics.csv", "table_02_loto_metrics",
     "Leave-one-territory-out R², RMSE and MAE for the same lagged IVC task. This is a spatial check, not a temporal forecast.",
     "ml.lagged_forecast.leave_one_territory_out"),
    ("3", "table_forecast_by_year.csv", "table_03_forecast_by_year",
     "Walk-forward R², RMSE and MAE by target year.",
     "ml.lagged_forecast.walk_forward"),
    ("4", "table_rain_forecast.csv", "table_04_rain_forecast",
     "Walk-forward comparison of no rainfall, rainfall as a predictor, and rainfall inside IVC, for the level of IVC and for the annual change.",
     "ml.lagged_forecast.run_climate_comparison"),
    ("5", "table_abm_dispersion.csv", "table_05_abm_dispersion",
     "Change in the coefficient of variation and in mean IVC after 10 simulated years under each allocation rule.",
     "scripts/regenerate_article.py::run_abm_grid"),
    ("6", "table_abm_monte_carlo.csv", "table_06_abm_monte_carlo",
     "Monte Carlo distribution of the percentage change in the coefficient of variation of IVC. Thirty replications, noise standard deviation 0.05.",
     "scripts/regenerate_article.py::run_abm_grid"),
    ("7", "table_sobol_indices.csv", "table_07_sobol_indices",
     "First-order and total-order Sobol indices for mean IVC, the coefficient of variation and the Gini coefficient.",
     "abm/05_sensitivity_analysis.py::run_sobol_scenarios"),
    ("8", "table_rain_qc_by_year.csv", "table_08_rain_qc",
     "Rainfall coverage and annual totals by calendar year after the station-level 80 percent rule.",
     "scripts/regenerate_article.py::rain_qc_table"),
    ("S1", "table_loto_by_territory.csv", "table_s1_loto_by_territory",
     "Leave-one-territory-out metrics for each held-out territory.",
     "ml.lagged_forecast.leave_one_territory_out"),
    ("S2", "table_legacy_cv_folds.csv", "table_s2_legacy_cv",
     "Re-run of the legacy same-year XGBoost grid on the pinned package versions.",
     "ml.lagged_forecast.legacy_contemporaneous_cv"),
    ("S3", "table_circularity_linear.csv", "table_s3_circularity",
     "In-sample linear reconstruction of IVC from the six same-year subindices.",
     "ml.lagged_forecast.contemporaneous_linear_r2"),
]


def export_manuscript_assets() -> None:
    tables = ASSETS / "tables"
    figures = ASSETS / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Manuscript assets",
        "",
        "Files for Socio-Economic Planning Sciences (Elsevier).",
        "Figure files contain no titles; the caption in this list is the figure caption.",
        "CSV files are the unrounded code output. DOCX files are three-line Word tables",
        "with R², RMSE and MAE rounded to 3 decimal places and percentages to 2.",
        "Vector fonts are embedded (PDF/EPS font type 42; SVG text stored as paths).",
        "PNG and TIFF are 600 dpi. TIFF uses LZW compression.",
        "One-column width is 90 mm. Two-column width is 190 mm.",
        "The smallest type is 7 pt DejaVu Sans.",
        "",
        "## Figures",
        "",
        "| Number | File stem | Width (mm) | Generator | Caption |",
        "|---|---|---:|---|---|",
    ]
    checks = []
    for number, stem, width, generator, caption in FIGURES:
        lines.append(
            f"| {number} | `{stem}` | {width:.0f} | `scripts/regenerate_article.py::{generator}` | {caption} |"
        )
        info = verify_figure(figures, stem, width)
        info["number"] = number
        checks.append(info)
        if abs(info["pdf_width_mm"] - width) > 1.0:
            raise RuntimeError(f"{stem} PDF width {info['pdf_width_mm']} mm is not {width} mm")
        if info["png_dpi"] != "600" or info["tif_dpi"] != "600":
            raise RuntimeError(f"{stem} raster DPI is png={info['png_dpi']} tif={info['tif_dpi']}")
    pd.DataFrame(checks).to_csv(ASSETS / "figure_verification.csv", index=False)

    lines += [
        "",
        "## Tables",
        "",
        "| Number | CSV | DOCX | Generator | Title |",
        "|---|---|---|---|---|",
    ]
    for number, source_name, stem, title, generator in TABLES:
        source = OUT / source_name
        export_table(source, tables / f"{stem}.csv", tables / f"{stem}.docx", f"Table {number}. {title}")
        lines.append(
            f"| {number} | `{stem}.csv` | `{stem}.docx` | `{generator}` | {title} |"
        )
    text = "\n".join(lines) + "\n"
    (ASSETS / "README.md").write_text(text, encoding="utf-8")
    print(f"Manuscript assets written to {ASSETS}")


if __name__ == "__main__":
    main()
