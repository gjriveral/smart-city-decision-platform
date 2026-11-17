# © 2026 Gabriel Jaime Rivera León. Todos los derechos reservados.
# El diseño arquitectónico, modelado lógico, simulaciones y estructura de este
# software son obra original del autor. La lógica computacional implementada
# comprende pipelines ETL, índices de vulnerabilidad, ensamble XGBoost con
# validación temporal y simulación basada en agentes (ABM).

"""
05_sensitivity_analysis.py
==========================
Análisis de sensibilidad de los coeficientes estructurales del ABM
(β_u, γ_u, δ_u, θ_u) frente a la crítica del jurado de que estaban
fijados a mano sin exploración paramétrica.

Métodos implementados (ambos se ejecutan):
  1. Barrido OAT (One-At-a-Time) ±20 % → diagrama de tornado.
  2. Muestreo Saltelli / índices de Sobol (SALib) si la librería está
     disponible; si no, grilla factorial 3^4 (±20 %, base) como fallback.

Salidas (carpeta data/processed/sensitivity/):
  - sensitivity_results.csv   — todas las corridas
  - sobol_indices.csv         — S1 / ST por parámetro (si SALib)
  - fig_tornado_ivc.png       — tornado OAT
  - fig_boxplot_ivc.png       — boxplots de IVC medio t=10
  - fig_sobol_indices.png     — barras S1/ST (si SALib)

Uso:
    py abm/05_sensitivity_analysis.py
"""
from __future__ import annotations

import importlib.util
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

# ── Importar el ABM real (archivo con dígito inicial) ─────────────────────────
_spec = importlib.util.spec_from_file_location(
    "urban_model", ROOT / "abm" / "04_urban_model.py"
)
abm = importlib.util.module_from_spec(_spec)
sys.modules["urban_model"] = abm
_spec.loader.exec_module(abm)

# ── Valores base (POM documentados en evidencia/calibracion_abm_coeficientes.md)
BASE = {
    "beta": 0.15,   # β_u — mejora wellbeing
    "gamma": 0.05,  # γ_u — reducción IVC
    "delta": 0.03,  # δ_u — depreciación infraestructura
    "theta": 0.10,  # θ_u — recuperación infraestructura
}
PARAM_LABELS = {
    "beta": r"$\beta_u$ (bienestar)",
    "gamma": r"$\gamma_u$ (IVC)",
    "delta": r"$\delta_u$ (depreciación)",
    "theta": r"$\theta_u$ (infraestructura)",
}
PCT = 0.20          # ±20 %
SCENARIO = "focalizado"  # escenario de referencia para la crítica doctoral
SEED = 42
# ρ / horizonte / presupuesto desde config.yaml vía abm module
RHO = float(abm.RHO_FOCALIZADO_DEFAULT)
N_SOBOL = 64        # Saltelli: N*(2D+2) = 64*10 = 640 evaluaciones

OUT_DIR = ROOT / "data" / "processed" / "sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CM = 1 / 2.54
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def load_inputs():
    csv_ivc = Path(abm._abs.datos_sociales) / "processed" / "indices_comunas.csv"
    csv_shap = Path(abm._abs.modelos) / "shap_values.csv"
    ivc = abm.load_ivc_from_csv(str(csv_ivc))
    shap = abm.load_shap_priority(str(csv_shap))
    return ivc, shap


def run_once(ivc_data, shap_priority, beta, gamma, delta, theta, seed=SEED):
    """Una corrida del ABM; retorna IVC medio territorial en t = N_YEARS."""
    # Silenciar logs verbosos del ABM durante el barrido
    prev = abm.logger.level
    abm.logger.setLevel(40)  # ERROR
    try:
        model = abm.UrbanModel(
            strategy=SCENARIO,
            ivc_data=ivc_data,
            seed=seed,
            rho=RHO,
            shap_priority=shap_priority,
            beta=beta,
            gamma=gamma,
            delta=delta,
            theta=theta,
        )
        df = model.run_simulation()
        year_n = df[df["year"] == abm.N_YEARS]
        return float(year_n["ivc_current"].mean())
    finally:
        abm.logger.setLevel(prev)


# ═══════════════════════════════════════════════════════════════════════════
# 1. OAT ±20 % → tornado
# ═══════════════════════════════════════════════════════════════════════════
def run_oat(ivc_data, shap_priority) -> pd.DataFrame:
    rows = []
    ivc_base = run_once(ivc_data, shap_priority, **BASE)
    rows.append({
        "method": "OAT", "param": "baseline", "multiplier": 1.0,
        **BASE, "ivc_medio_t10": ivc_base,
        "delta_vs_base": 0.0,
    })
    print(f"  Baseline IVC_t10 = {ivc_base:.4f}")

    for name, base_val in BASE.items():
        for mult in (1.0 - PCT, 1.0 + PCT):
            params = dict(BASE)
            params[name] = base_val * mult
            ivc = run_once(ivc_data, shap_priority, **params)
            rows.append({
                "method": "OAT",
                "param": name,
                "multiplier": mult,
                **params,
                "ivc_medio_t10": ivc,
                "delta_vs_base": ivc - ivc_base,
            })
            print(f"  OAT {name}×{mult:.2f} → IVC_t10={ivc:.4f} (Δ={ivc - ivc_base:+.4f})")
    return pd.DataFrame(rows)


def plot_tornado(oat_df: pd.DataFrame, out_path: Path, ivc_base: float):
    """Tornado: rango de IVC_t10 al variar cada parámetro ±20 %."""
    fig, ax = plt.subplots(figsize=(16 * CM, 10 * CM))
    params = list(BASE.keys())
    lows, highs, labels = [], [], []
    for p in params:
        sub = oat_df[(oat_df["param"] == p) & (oat_df["method"] == "OAT")]
        lo = float(sub.loc[sub["multiplier"] < 1, "ivc_medio_t10"].iloc[0])
        hi = float(sub.loc[sub["multiplier"] > 1, "ivc_medio_t10"].iloc[0])
        lows.append(min(lo, hi))
        highs.append(max(lo, hi))
        labels.append(PARAM_LABELS[p])

    # Ordenar por amplitud (efecto total) descendente
    amp = [h - l for l, h in zip(lows, highs)]
    order = np.argsort(amp)[::-1]
    lows = [lows[i] for i in order]
    highs = [highs[i] for i in order]
    labels = [labels[i] for i in order]
    y = np.arange(len(labels))

    ax.barh(y, [h - l for l, h in zip(lows, highs)], left=lows,
            color="#3949ab", alpha=0.85, height=0.55)
    ax.axvline(ivc_base, color="#d62728", linestyle="--", linewidth=1.4,
               label=f"Línea base (IVC={ivc_base:.4f})")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("IVC medio territorial en t = 10")
    ax.legend(loc="best", frameon=False)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Figura tornado → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Grilla factorial 3^4 (±20 %, base) → boxplots
# ═══════════════════════════════════════════════════════════════════════════
def run_grid(ivc_data, shap_priority) -> pd.DataFrame:
    levels = (1.0 - PCT, 1.0, 1.0 + PCT)
    rows = []
    names = list(BASE.keys())
    total = 3 ** 4
    k = 0
    t0 = time.time()
    for mb in levels:
        for mg in levels:
            for md in levels:
                for mt in levels:
                    k += 1
                    params = {
                        "beta": BASE["beta"] * mb,
                        "gamma": BASE["gamma"] * mg,
                        "delta": BASE["delta"] * md,
                        "theta": BASE["theta"] * mt,
                    }
                    ivc = run_once(ivc_data, shap_priority, **params)
                    rows.append({
                        "method": "grid",
                        "param": "all",
                        "multiplier": np.nan,
                        **params,
                        "ivc_medio_t10": ivc,
                        "mult_beta": mb, "mult_gamma": mg,
                        "mult_delta": md, "mult_theta": mt,
                    })
                    if k % 27 == 0 or k == total:
                        print(f"  Grid {k}/{total} ({time.time() - t0:.1f}s)")
    return pd.DataFrame(rows)


def plot_boxplots(grid_df: pd.DataFrame, oat_df: pd.DataFrame, out_path: Path):
    """
    Boxplots: para cada parámetro, distribución de IVC_t10 cuando ese
    parámetro toma los niveles −20/0/+20 % (resto libre en la grilla).
    """
    fig, axes = plt.subplots(1, 4, figsize=(16 * CM, 8 * CM), sharey=True)
    for ax, name in zip(axes, BASE.keys()):
        # Filtrar filas de la grilla agrupadas por el multiplicador del param
        col_mult = f"mult_{name}"
        data, tick_labels = [], []
        for mult, lab in [(1.0 - PCT, "−20 %"), (1.0, "base"), (1.0 + PCT, "+20 %")]:
            vals = grid_df.loc[
                np.isclose(grid_df[col_mult], mult), "ivc_medio_t10"
            ].values
            data.append(vals)
            tick_labels.append(lab)
        bp = ax.boxplot(data, patch_artist=True, widths=0.55)
        for box in bp["boxes"]:
            box.set_facecolor("#9fa8da")
            box.set_edgecolor("#3949ab")
        for med in bp["medians"]:
            med.set_color("#d62728")
            med.set_linewidth(1.5)
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_title(PARAM_LABELS[name], fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("IVC medio t = 10")
    fig.suptitle(
        f"Sensibilidad ±20 % — escenario «{SCENARIO}» (grilla 3⁴)",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Figura boxplots → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Sobol (SALib) — opcional
# ═══════════════════════════════════════════════════════════════════════════
def run_sobol(ivc_data, shap_priority):
    try:
        from SALib.analyze import sobol as sobol_analyze
        from SALib.sample import sobol as sobol_sample
    except ImportError:
        print("  SALib no instalada — se omite el análisis de Sobol "
              "(instalar con: py -m pip install SALib).")
        return None, None

    problem = {
        "num_vars": 4,
        "names": list(BASE.keys()),
        "bounds": [
            [BASE[n] * (1 - PCT), BASE[n] * (1 + PCT)] for n in BASE
        ],
    }
    param_values = sobol_sample.sample(problem, N_SOBOL, calc_second_order=False)
    print(f"  Sobol: {len(param_values)} evaluaciones...")
    Y = np.empty(len(param_values))
    t0 = time.time()
    rows = []
    for i, x in enumerate(param_values):
        params = dict(zip(BASE.keys(), x))
        Y[i] = run_once(ivc_data, shap_priority, **params)
        rows.append({
            "method": "sobol",
            "param": "all",
            "multiplier": np.nan,
            **params,
            "ivc_medio_t10": Y[i],
        })
        if (i + 1) % 100 == 0 or i + 1 == len(param_values):
            print(f"    {i + 1}/{len(param_values)} ({time.time() - t0:.1f}s)")

    Si = sobol_analyze.analyze(problem, Y, calc_second_order=False, print_to_console=False)
    idx_df = pd.DataFrame({
        "parametro": problem["names"],
        "S1": Si["S1"],
        "S1_conf": Si["S1_conf"],
        "ST": Si["ST"],
        "ST_conf": Si["ST_conf"],
    })
    return pd.DataFrame(rows), idx_df


def plot_sobol(idx_df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(16 * CM, 9 * CM))
    x = np.arange(len(idx_df))
    w = 0.35
    ax.bar(x - w / 2, idx_df["S1"], w, yerr=idx_df["S1_conf"],
           label="S1 (efecto de primer orden)", color="#3949ab",
           capsize=3, alpha=0.9)
    ax.bar(x + w / 2, idx_df["ST"], w, yerr=idx_df["ST_conf"],
           label="ST (efecto total)", color="#ef6c00",
           capsize=3, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([PARAM_LABELS[p] for p in idx_df["parametro"]], fontsize=9)
    ax.set_ylabel("Índice de Sobol")
    ax.set_ylim(0, max(1.05, float(idx_df[["S1", "ST"]].max().max()) * 1.15))
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Figura Sobol → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("ANÁLISIS DE SENSIBILIDAD ABM — β, γ, δ, θ (±20 %)")
    print(f"Escenario de referencia: {SCENARIO}")
    print("=" * 70)

    ivc_data, shap_priority = load_inputs()

    print("\n[1/3] Barrido OAT ±20 %...")
    oat_df = run_oat(ivc_data, shap_priority)
    ivc_base = float(oat_df.loc[oat_df["param"] == "baseline", "ivc_medio_t10"].iloc[0])
    plot_tornado(oat_df, OUT_DIR / "fig_tornado_ivc.png", ivc_base)

    print("\n[2/3] Grilla factorial 3⁴ (±20 %, base)...")
    grid_df = run_grid(ivc_data, shap_priority)
    plot_boxplots(grid_df, oat_df, OUT_DIR / "fig_boxplot_ivc.png")

    print("\n[3/3] Análisis de Sobol (SALib)...")
    sobol_df, idx_df = run_sobol(ivc_data, shap_priority)

    # Ensamblar CSV maestro
    frames = [oat_df, grid_df]
    if sobol_df is not None:
        frames.append(sobol_df)
        idx_path = OUT_DIR / "sobol_indices.csv"
        idx_df.to_csv(idx_path, index=False, encoding="utf-8")
        print(f"  Índices Sobol → {idx_path}")
        plot_sobol(idx_df, OUT_DIR / "fig_sobol_indices.png")

    results = pd.concat(frames, ignore_index=True, sort=False)
    csv_path = OUT_DIR / "sensitivity_results.csv"
    results.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\nResultados → {csv_path}  ({len(results)} filas)")
    print(f"Figuras    → {OUT_DIR}")
    print("=" * 70)
    return results


def _unpack_sobol_row(values, n_territories: int) -> dict:
    names = ["beta", "gamma", "delta", "theta", "rho", "n_prioritarios", "noise_sd"]
    raw = dict(zip(names, values))
    n_priority = int(np.clip(np.rint(raw["n_prioritarios"]), 1, max(1, n_territories - 1)))
    return {
        "beta": float(raw["beta"]),
        "gamma": float(raw["gamma"]),
        "delta": float(raw["delta"]),
        "theta": float(raw["theta"]),
        "rho": float(raw["rho"]),
        "n_priority": n_priority,
        "noise_sd": float(raw["noise_sd"]),
    }


def run_sobol_scenarios(
    ivc_data: dict,
    shap_priority: dict,
    n_sobol: int = 16,
    seed: int = 42,
    saturation_active: bool | None = None,
    ranking_rule: str | None = None,
    reparto_top: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Saltelli sample over structural coefficients, rho, the number of
    prioritized territories, and the noise standard deviation.

    Evaluated on all three allocation rules. Outputs are mean IVC, the
    coefficient of variation, and the Gini coefficient at the horizon.
    """
    from SALib.analyze import sobol as sobol_analyze
    from SALib.sample import sobol as sobol_sample

    from abm.dispersion import coefficient_of_variation, gini

    if saturation_active is None:
        saturation_active = bool(abm.SATURACION_ACTIVA_DEFAULT)
    if ranking_rule is None:
        ranking_rule = abm.RANKING_REGLA_DEFAULT
    if reparto_top is None:
        reparto_top = abm.REPARTO_TOP_DEFAULT

    problem = {
        "num_vars": 7,
        "names": ["beta", "gamma", "delta", "theta", "rho", "n_prioritarios", "noise_sd"],
        "bounds": [
            [BASE["beta"] * (1 - PCT), BASE["beta"] * (1 + PCT)],
            [BASE["gamma"] * (1 - PCT), BASE["gamma"] * (1 + PCT)],
            [BASE["delta"] * (1 - PCT), BASE["delta"] * (1 + PCT)],
            [BASE["theta"] * (1 - PCT), BASE["theta"] * (1 + PCT)],
            [0.50, 0.95],
            [1.0, 10.0],
            [0.0, 0.10],
        ],
    }
    sample = sobol_sample.sample(problem, n_sobol, calc_second_order=False)
    scenarios = ["focalizado", "distribuido", "adaptativo"]
    outputs = ("mean_ivc", "cv", "gini")
    store = {
        sc: {out: np.empty(len(sample)) for out in outputs} for sc in scenarios
    }
    detail_rows = []
    prev = abm.logger.level
    abm.logger.setLevel(40)
    try:
        for i, values in enumerate(sample):
            params = _unpack_sobol_row(values, len(ivc_data))
            for scenario in scenarios:
                model = abm.UrbanModel(
                    strategy=scenario,
                    ivc_data=ivc_data,
                    seed=seed,
                    rho=params["rho"],
                    shap_priority=shap_priority,
                    beta=params["beta"],
                    gamma=params["gamma"],
                    delta=params["delta"],
                    theta=params["theta"],
                    ranking_rule=ranking_rule,
                    n_priority=params["n_priority"],
                    reparto_top=reparto_top,
                    saturation_active=saturation_active,
                    noise_sd=params["noise_sd"],
                )
                df = model.run_simulation()
                terminal = df[df["year"] == abm.N_YEARS]["ivc_current"].to_numpy(dtype=float)
                metrics = {
                    "mean_ivc": float(np.mean(terminal)),
                    "cv": float(coefficient_of_variation(terminal)),
                    "gini": float(gini(terminal)),
                }
                for out, value in metrics.items():
                    store[scenario][out][i] = value
                detail_rows.append({
                    "sample": i,
                    "scenario": scenario,
                    **params,
                    **metrics,
                })
            if (i + 1) % 50 == 0 or i + 1 == len(sample):
                print(f"  Sobol sample {i + 1}/{len(sample)}", flush=True)
    finally:
        abm.logger.setLevel(prev)

    index_rows = []
    for scenario in scenarios:
        for out in outputs:
            analyzed = sobol_analyze.analyze(
                problem, store[scenario][out], calc_second_order=False, print_to_console=False
            )
            for j, name in enumerate(problem["names"]):
                index_rows.append({
                    "scenario": scenario,
                    "output": out,
                    "parameter": name,
                    "S1": float(analyzed["S1"][j]),
                    "S1_conf": float(analyzed["S1_conf"][j]),
                    "ST": float(analyzed["ST"][j]),
                    "ST_conf": float(analyzed["ST_conf"][j]),
                    "n_sobol": int(n_sobol),
                    "n_evaluations": int(len(sample)),
                    "ranking_rule": ranking_rule,
                    "reparto_top": reparto_top,
                    "saturation_active": bool(saturation_active),
                })
    return pd.DataFrame(index_rows), pd.DataFrame(detail_rows)


if __name__ == "__main__":
    main()
