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
from ml.lagged_forecast import legacy_contemporaneous_cv, run_climate_comparison, run_lagged_suite

OUT = ROOT / "article_outputs"
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


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def apply_style() -> None:
    import logging
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


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
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(order))
    width = 0.36
    temp = pooled.set_index("model").loc[order, "R2"]
    spat = loto.set_index("model").loc[order, "R2"]
    ax.bar(x - width / 2, temp, width, label="Temporal walk-forward", color="#1f4e79")
    ax.bar(x + width / 2, spat, width, label="Leave-one-territory-out", color="#e07a3d")
    ax.axhline(0, color="#444444", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in order], rotation=15, ha="right")
    ax.set_ylabel("Out-of-sample R²")
    ax.set_title("IVC at t+1 predicted from subindices at t")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    fig.tight_layout()
    save_figure(fig, "fig_forecast_r2")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for model, color in zip(order, ("#4d4d4d", "#1f4e79", "#2a9d8f", "#e07a3d")):
        g = by_year[by_year["model"] == model].sort_values("anio_objetivo")
        ax.plot(g["anio_objetivo"], g["R2"], marker="o", label=MODEL_LABELS[model], color=color)
    ax.axhline(0, color="#444444", linewidth=0.6)
    ax.set_xlabel("Target year (IVC at t+1)")
    ax.set_ylabel("R² in that target year")
    ax.set_title("Walk-forward R² by target year")
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, "fig_forecast_by_year")


def plot_abm(paths: pd.DataFrame) -> None:
    show = [
        ("focused_yearly_ivc", "Focused, yearly IVC rank", "#1f4e79"),
        ("focused_fixed_shap", "Focused, fixed SHAP rank", "#b00020"),
        ("equal", "Equal allocation", "#4d4d4d"),
        ("adaptive", "Adaptive allocation", "#2a9d8f"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for key, label, color in show:
        g = paths[paths["specification"] == key]
        ax.plot(g["year"], g["cv"], marker="o", label=label, color=color)
    ax.set_xlabel("Simulation year")
    ax.set_ylabel("Coefficient of variation of IVC")
    ax.set_title("Dispersion of IVC across 21 territories")
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, "fig_abm_cv_paths")


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
    colors = {"ninguno": "#4d4d4d", "predictor": "#1f4e79", "subindice": "#e07a3d"}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), sharey=False)
    for ax, target, title, column in (
        (axes[0], "level", "IVC level at t+1", "R2"),
        (axes[1], "change", "Change in IVC (t+1 minus t)", "R2"),
    ):
        sub = metrics[metrics["target"] == target]
        x = np.arange(len(order))
        width = 0.25
        for i, uso in enumerate(usos):
            vals = [
                float(sub[(sub["clima_uso"] == uso) & (sub["model"] == m)][column].iloc[0])
                for m in order
            ]
            ax.bar(x + (i - 1) * width, vals, width, label=uso_labels[uso], color=colors[uso])
        ax.axhline(0, color="#444444", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in order], rotation=15, ha="right")
        ax.set_ylabel("Walk-forward R²")
        ax.set_title(title)
    axes[1].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=1)
    fig.suptitle("Does rainfall improve the IVC forecast?", y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig_rain_forecast_r2")


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
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 5.4), sharey=True)
    sub = indices[indices["output"] == "cv"].copy()
    order = list(SOBOL_LABELS)
    y = np.arange(len(order))
    height = 0.36
    hi = float(np.nanmax(sub[["S1_ci_high", "ST_ci_high"]].to_numpy()))
    for ax, scenario in zip(axes, ("focalizado", "distribuido", "adaptativo")):
        g = sub[sub["scenario"] == scenario].set_index("parameter").loc[order]
        s1_err = np.vstack([
            np.clip(g["S1"] - g["S1_ci_low"], 0, None),
            np.clip(g["S1_ci_high"] - g["S1"], 0, None),
        ])
        st_err = np.vstack([
            np.clip(g["ST"] - g["ST_ci_low"], 0, None),
            np.clip(g["ST_ci_high"] - g["ST"], 0, None),
        ])
        ax.barh(y + height / 2, g["S1"], height=height, xerr=s1_err, color="#5b7c99",
                label="First order (S1)", capsize=2, error_kw={"elinewidth": 0.6})
        ax.barh(y - height / 2, g["ST"], height=height, xerr=st_err, color="#1f4e79",
                label="Total order (ST)", capsize=2, error_kw={"elinewidth": 0.6})
        ax.axvline(0, color="#444444", linewidth=0.6)
        ax.axvline(1, color="#888888", linewidth=0.5, linestyle=":")
        ax.set_title(SCENARIO_LABELS[scenario])
        ax.set_xlabel("Sobol index")
        ax.set_yticks(y)
        ax.set_yticklabels(list(SOBOL_LABELS.values()))
        ax.set_xlim(-0.2, max(1.08, hi + 0.05))
    axes[0].set_ylabel("Parameter")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Sobol indices for the CV of IVC (95% bootstrap interval)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_sobol_cv.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "fig_sobol_cv.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "fig_sobol_cv.tiff", dpi=600, bbox_inches="tight", facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate revised article outputs")
    parser.add_argument("--skip-legacy", action="store_true")
    parser.add_argument("--sobol-n", type=int, default=1024)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    apply_style()
    started = time.time()
    panel = pd.read_csv(PANEL, encoding="utf-8-sig")

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
        "(coverage ≥ 80%, 2012 and 2026 excluded; station 352 awaits a re-run)...",
        flush=True,
    )
    rain_raw = pd.read_csv(ROOT / "data/processed/siata_territorio_anual.csv")
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
    print(f"Done in {summary['elapsed_s']}s. Outputs in {OUT}")


if __name__ == "__main__":
    main()
