"""
sensibilidad_coeficientes.py
============================
Barrido OAT (One-At-a-Time) de los coeficientes estructurales del ABM
(β_u, γ_u, δ_u, θ_u), análogo al barrido de ρ que alimenta Tabla 15 /
Figura 7 (entregables_tesis / export_tesis_resultados.py).

Diseño (mismo patrón que el barrido de rho):
  - Escenario: focalizado
  - rho fijo: 0.80
  - Semilla: 42 (reproducible)
  - Un coeficiente a la vez; los otros tres en su valor por defecto
  - Rango: ±20 % alrededor del valor base, 5 puntos por coeficiente
    (multiplicadores 0.80, 0.90, 1.00, 1.10, 1.20)
  - Métricas terminales (t = N_YEARS): ivc_medio, gini, maximin

Salidas:
  - models/sensibilidad_coeficientes_abm.csv
  - models/sensibilidad_coeficientes_abm.png  (1 subfigura por coeficiente)

Uso:
    py abm/sensibilidad_coeficientes.py
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "urban_model_sens", ROOT / "abm" / "04_urban_model.py"
)
abm = importlib.util.module_from_spec(_spec)
sys.modules["urban_model_sens"] = abm
_spec.loader.exec_module(abm)

# Silenciar logs verbosos del ABM durante el barrido
logging.getLogger(abm.__name__ if hasattr(abm, "__name__") else "urban_model").setLevel(
    logging.WARNING
)
for name in list(logging.Logger.manager.loggerDict):
    if "urban" in name.lower() or name.endswith("04_urban_model"):
        logging.getLogger(name).setLevel(logging.WARNING)
abm.logger.setLevel(logging.WARNING)

# Valores base = defaults de UrbanModel / agentes (reproducen caso actual)
BASE = {
    "beta": 0.15,
    "gamma": 0.05,
    "delta": 0.03,
    "theta": 0.10,
}
PARAM_ORDER = ["beta", "gamma", "delta", "theta"]
PARAM_LABELS = {
    "beta": r"$\beta_u$ (bienestar)",
    "gamma": r"$\gamma_u$ (IVC)",
    "delta": r"$\delta_u$ (depreciación)",
    "theta": r"$\theta_u$ (infraestructura)",
}
# ±20 %, ≥5 puntos (incluye el caso base = 1.00)
MULTIPLIERS = [0.80, 0.90, 1.00, 1.10, 1.20]
SCENARIO = "focalizado"
# ρ desde config.yaml → simulacion.escenarios[focalizado].peso
RHO = float(abm.RHO_FOCALIZADO_DEFAULT)
SEED = 42

OUT_CSV = Path(abm._abs.modelos) / "sensibilidad_coeficientes_abm.csv"
OUT_FIG = Path(abm._abs.modelos) / "sensibilidad_coeficientes_abm.png"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

CM = 1 / 2.54
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def gini(values) -> float:
    """Misma definición que entregables_tesis/_pipeline_real.py (barrido rho)."""
    v = np.sort(np.asarray(list(values), dtype=float))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    cum = np.cumsum(v)
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)


def terminal_metrics(df: pd.DataFrame, n_years: int) -> dict[str, float]:
    sub = df[df["year"] == n_years]
    ivc_vals = sub["ivc_current"].to_numpy(dtype=float)
    return {
        "ivc_medio_t10": float(np.mean(ivc_vals)),
        "gini_t10": gini(ivc_vals),
        "maximin_t10": float(np.max(ivc_vals)),
    }


def run_once(ivc_data, shap_priority, **coefs) -> dict[str, float]:
    model = abm.UrbanModel(
        strategy=SCENARIO,
        ivc_data=ivc_data,
        seed=SEED,
        rho=RHO,
        shap_priority=shap_priority,
        beta=coefs["beta"],
        gamma=coefs["gamma"],
        delta=coefs["delta"],
        theta=coefs["theta"],
    )
    df = model.run_simulation()
    return terminal_metrics(df, abm.N_YEARS)


def main():
    print("=" * 60)
    print("Barrido OAT de coeficientes estructurales ABM")
    print(f"  escenario={SCENARIO}  rho={RHO}  seed={SEED}")
    print(f"  multiplicadores={MULTIPLIERS}  (±20 %)")
    print("=" * 60)

    csv_ivc = Path(abm._abs.datos_sociales) / "processed" / "indices_comunas.csv"
    csv_shap = Path(abm._abs.modelos) / "shap_values.csv"
    ivc_data = abm.load_ivc_from_csv(str(csv_ivc))
    # Preferir IVC predicho OOS si existe (FUENTE_IVC default del ABM)
    pred_path = Path(abm._abs.modelos) / "ivc_predictions_oos.csv"
    if pred_path.is_file():
        ivc_data = abm.load_ivc_from_predictions(
            str(pred_path), historico_fallback=ivc_data
        )
        print(f"  IVC inicial: predicho OOS ({pred_path.name})")
    else:
        print(f"  IVC inicial: histórico ({csv_ivc.name})")
    shap_priority = abm.load_shap_priority(str(csv_shap))

    rows = []
    for param in PARAM_ORDER:
        base_val = BASE[param]
        print(f"\n--- OAT: {param} (base={base_val}) ---")
        for mult in MULTIPLIERS:
            coefs = dict(BASE)
            coefs[param] = base_val * mult
            metrics = run_once(ivc_data, shap_priority, **coefs)
            row = {
                "parametro": param,
                "valor_base": base_val,
                "multiplicador": mult,
                "valor": coefs[param],
                "es_caso_base": mult == 1.0,
                **metrics,
            }
            rows.append(row)
            print(
                f"  {param}={coefs[param]:.4f} (×{mult:.2f})  "
                f"ivc={metrics['ivc_medio_t10']:.4f}  "
                f"gini={metrics['gini_t10']:.4f}  "
                f"maximin={metrics['maximin_t10']:.4f}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nCSV → {OUT_CSV}")

    # Figura análoga a fig07_sensibilidad_rho: una subfigura por coeficiente,
    # tres series (ivc_medio, gini, maximin) vs valor del parámetro.
    fig, axes = plt.subplots(2, 2, figsize=(16 * CM, 12 * CM), sharey=False)
    axes = axes.ravel()
    metric_specs = [
        ("ivc_medio_t10", "IVC medio t=10", "#1f77b4"),
        ("gini_t10", "Gini(IVC) t=10", "#d62728"),
        ("maximin_t10", "Maximin t=10", "#2ca02c"),
    ]

    for ax, param in zip(axes, PARAM_ORDER):
        sub = df[df["parametro"] == param].sort_values("valor")
        x = sub["valor"].to_numpy()
        for col, label, color in metric_specs:
            ax.plot(
                x, sub[col], color=color, linewidth=1.6, marker="o",
                markersize=4, label=label,
            )
        # Marcar caso base (×1.00), como el diamante de ρ=0.80 en fig07
        base_row = sub[sub["es_caso_base"]]
        if not base_row.empty:
            xb = float(base_row.iloc[0]["valor"])
            ax.axvline(xb, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
            ax.scatter(
                [xb], [float(base_row.iloc[0]["gini_t10"])],
                color="black", zorder=5, s=40, marker="D",
                label=f"base ({param}={BASE[param]})",
            )
        ax.set_title(PARAM_LABELS[param], fontsize=10)
        ax.set_xlabel(f"valor de {param}")
        ax.grid(alpha=0.25, linewidth=0.5)
        if param == PARAM_ORDER[0]:
            ax.legend(loc="best", fontsize=7, frameon=False)

    fig.suptitle(
        f"Sensibilidad OAT ±20 % — escenario {SCENARIO}, ρ={RHO}\n"
        f"(análogo al barrido de ρ; un coeficiente a la vez)",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura → {OUT_FIG}")
    print("DONE")


if __name__ == "__main__":
    main()
