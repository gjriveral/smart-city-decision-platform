"""
05_scenario_analysis.py
========================
Loads simulation_results.csv, generates 4 matplotlib figures,
and produces a professional dark-themed HTML report with embedded plots.
"""

import os
import sys
import base64
import logging
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator

# ─── Config de ciudad ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import get_config

cfg = get_config()
UT  = cfg.unidad_territorial   # "comuna", "localidad", etc.

# ─── Logging ─────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "abm_execution.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
RESULTS_PATH = os.path.join(ROOT, "data", "processed", "simulation_results.csv")
REPORTS_DIR  = os.path.join(ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

ANNUAL_BUDGET = 1_000_000  # million COP/year (must match 04_urban_model.py)
SCENARIOS = ["focalizado", "distribuido", "adaptativo"]
SCENARIO_LABELS = {
    "focalizado":  "A — Focalizado",
    "distribuido": "B — Distribuido",
    "adaptativo":  "C — Adaptativo",
}
SCENARIO_COLORS = {
    "focalizado":  "#E74C3C",
    "distribuido": "#3498DB",
    "adaptativo":  "#2ECC71",
}

DARK_BG  = "#0D1117"
DARK_AX  = "#161B22"
ACCENT   = "#58A6FF"
TEXT_CLR = "#C9D1D9"
GRID_CLR = "#21262D"

# ─── Helpers ─────────────────────────────────────────────────────────────────

def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def apply_dark_style(fig, axes):
    """Apply unified dark theme to a figure and all its axes."""
    fig.patch.set_facecolor(DARK_BG)
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(DARK_AX)
        ax.tick_params(colors=TEXT_CLR)
        ax.xaxis.label.set_color(TEXT_CLR)
        ax.yaxis.label.set_color(TEXT_CLR)
        ax.title.set_color(ACCENT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_CLR)
        ax.grid(color=GRID_CLR, linestyle="--", linewidth=0.6, alpha=0.8)
        ax.tick_params(axis="both", which="both", colors=TEXT_CLR)


# ─── Figures ─────────────────────────────────────────────────────────────────

def fig1_wellbeing_evolution(df: pd.DataFrame) -> str:
    """Line chart: mean wellbeing over time per scenario."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for scenario in SCENARIOS:
        sub = df[df["scenario"] == scenario].groupby("year")["wellbeing"].mean()
        ax.plot(
            sub.index, sub.values,
            label=SCENARIO_LABELS[scenario],
            color=SCENARIO_COLORS[scenario],
            linewidth=2.5, marker="o", markersize=5,
        )

    ax.set_title("Fig 1 · Mean Wellbeing Evolution Over 10 Years", fontsize=13, pad=12)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Mean Wellbeing Index (0–1)", fontsize=11)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(facecolor=DARK_AX, edgecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=10)
    apply_dark_style(fig, ax)
    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def fig2_wellbeing_heatmap(df: pd.DataFrame) -> str:
    """Heatmap: final-year wellbeing per commune across scenarios."""
    year10 = df[df["year"] == df["year"].max()]
    pivot = year10.pivot_table(index=UT, columns="scenario", values="wellbeing")
    pivot = pivot[SCENARIOS]  # ensure column order
    pivot = pivot.sort_values("focalizado", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 8))
    cmap = matplotlib.colormaps.get_cmap("RdYlGn")
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=0.35, vmax=0.85)

    ax.set_xticks(range(len(SCENARIOS)))
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS], color=TEXT_CLR, fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, color=TEXT_CLR, fontsize=8)
    ax.set_title("Fig 2 · Wellbeing Heatmap at Year 10 (communes × scenarios)", fontsize=12, pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.ax.tick_params(colors=TEXT_CLR)
    cbar.set_label("Wellbeing", color=TEXT_CLR)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(SCENARIOS)):
            ax.text(j, i, f"{pivot.values[i, j]:.3f}",
                    ha="center", va="center", fontsize=7.5,
                    color="black" if pivot.values[i, j] > 0.5 else "white")

    apply_dark_style(fig, ax)
    fig.patch.set_facecolor(DARK_BG)
    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def fig3_budget_allocation(df: pd.DataFrame) -> str:
    """Stacked bar chart: mean budget allocation across years per scenario."""
    # Show year 1 and year 10 allocations for each scenario
    years = [1, 5, 10]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

    for ax, scenario in zip(axes, SCENARIOS):
        pivot = (
            df[df["scenario"] == scenario]
            .pivot_table(index="year", columns=UT, values="investment_allocated")
        )
        # Normalize to percentages
        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
        # Plot stacked bars for selected years
        sel = pivot_pct.loc[pivot_pct.index.isin(years)]
        sel.T.plot(kind="bar", ax=ax, stacked=True, legend=False,
                   colormap="tab20", edgecolor="none")
        ax.set_title(SCENARIO_LABELS[scenario], color=ACCENT, fontsize=10)
        ax.set_xlabel("Commune", fontsize=8)
        ax.set_ylabel("% Budget Allocated", fontsize=9)
        ax.tick_params(axis="x", rotation=90, labelsize=6)
        apply_dark_style(fig, ax)

    axes[1].legend(
        title="Year", labels=[str(y) for y in years],
        facecolor=DARK_AX, edgecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=8,
        loc="upper right"
    )
    fig.suptitle("Fig 3 · Budget Allocation Profile by Scenario & Year",
                 color=ACCENT, fontsize=12, y=1.01)
    apply_dark_style(fig, axes)
    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def fig4_infrastructure_capacity(df: pd.DataFrame) -> str:
    """Area chart: mean infrastructure capacity over time per scenario."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for scenario in SCENARIOS:
        sub = df[df["scenario"] == scenario].groupby("year")["infrastructure_capacity"].mean()
        ax.fill_between(sub.index, sub.values,
                        color=SCENARIO_COLORS[scenario], alpha=0.30)
        ax.plot(sub.index, sub.values,
                label=SCENARIO_LABELS[scenario],
                color=SCENARIO_COLORS[scenario], linewidth=2.5)

    ax.set_title("Fig 4 · Mean Infrastructure Capacity Over 10 Years", fontsize=13, pad=12)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Infrastructure Capacity (0–1)", fontsize=11)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(facecolor=DARK_AX, edgecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=10)
    apply_dark_style(fig, ax)
    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


# ─── HTML Report ─────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Informe de Escenarios ABM — {CIUDAD} 2026</title>
  <style>
    :root {{
      --bg:        #0D1117;
      --surface:   #161B22;
      --border:    #30363D;
      --accent:    #58A6FF;
      --accent2:   #3FB950;
      --accent3:   #FF7B72;
      --text:      #C9D1D9;
      --subtext:   #8B949E;
      --yellow:    #E3B341;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      line-height: 1.6;
      padding: 2rem 1rem;
    }}
    .container {{ max-width: 1100px; margin: 0 auto; }}
    header {{
      text-align: center;
      padding: 2.5rem 1rem 2rem;
      border-bottom: 1px solid var(--border);
      margin-bottom: 3rem;
    }}
    header h1 {{
      font-size: 2rem;
      color: var(--accent);
      letter-spacing: 0.03em;
    }}
    header p {{ color: var(--subtext); margin-top: 0.5rem; font-size: 0.95rem; }}
    .badge {{
      display: inline-block;
      padding: 0.25rem 0.75rem;
      border-radius: 20px;
      font-size: 0.8rem;
      font-weight: 600;
      margin: 0.2rem;
    }}
    .badge-a {{ background: rgba(231,76,60,0.18);  color: #E74C3C; border: 1px solid #E74C3C; }}
    .badge-b {{ background: rgba(52,152,219,0.18); color: #3498DB; border: 1px solid #3498DB; }}
    .badge-c {{ background: rgba(46,204,113,0.18); color: #2ECC71; border: 1px solid #2ECC71; }}
    section {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1.8rem 2rem;
      margin-bottom: 2rem;
    }}
    section h2 {{
      font-size: 1.25rem;
      color: var(--accent);
      margin-bottom: 1rem;
      padding-bottom: 0.5rem;
      border-bottom: 1px solid var(--border);
    }}
    section h3 {{ font-size: 1rem; color: var(--yellow); margin: 1rem 0 0.5rem; }}
    img.plot {{
      width: 100%;
      border-radius: 8px;
      border: 1px solid var(--border);
      margin-top: 0.75rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
      margin-top: 0.75rem;
    }}
    th {{
      background: var(--border);
      color: var(--accent);
      padding: 0.6rem 1rem;
      text-align: left;
    }}
    td {{ padding: 0.55rem 1rem; border-bottom: 1px solid var(--border); }}
    tr:hover td {{ background: rgba(88,166,255,0.05); }}
    .best  {{ color: var(--accent2); font-weight: 700; }}
    .worst {{ color: var(--accent3); }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
      margin-top: 1rem;
    }}
    .metric-card {{
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem 1.25rem;
    }}
    .metric-card .value {{
      font-size: 1.8rem;
      font-weight: 700;
      color: var(--accent);
    }}
    .metric-card .label {{ color: var(--subtext); font-size: 0.82rem; margin-top: 0.25rem; }}
    .recommendation {{
      background: rgba(46,204,113,0.07);
      border-left: 4px solid var(--accent2);
      border-radius: 0 8px 8px 0;
      padding: 1.2rem 1.5rem;
      margin-top: 1rem;
    }}
    .recommendation p {{ margin-top: 0.5rem; }}
    footer {{
      text-align: center;
      color: var(--subtext);
      font-size: 0.8rem;
      padding: 2rem 0 1rem;
      border-top: 1px solid var(--border);
      margin-top: 3rem;
    }}
  </style>
</head>
<body>
<div class="container">

<header>
  <h1>&#128202; Análisis de Escenarios ABM — {CIUDAD}</h1>
  <p>Simulación de Inversión Urbana Estratégica · Universidad de Antioquia · 2026</p>
  <p style="margin-top:0.75rem;">
    <span class="badge badge-a">A · Focalizado</span>
    <span class="badge badge-b">B · Distribuido</span>
    <span class="badge badge-c">C · Adaptativo</span>
  </p>
</header>

<!-- KPI Cards -->
<section>
  <h2>🔑 Métricas Clave — Año 10</h2>
  <div class="metric-grid">
{KPI_CARDS}
  </div>
</section>

<!-- Fig 1 -->
<section>
  <h2>📈 Figura 1 — Evolución del Bienestar Promedio</h2>
  <img class="plot" src="data:image/png;base64,{FIG1}" alt="Wellbeing evolution"/>
</section>

<!-- Fig 2 -->
<section>
  <h2>🌡️ Figura 2 — Heatmap de Bienestar por Comuna (Año 10)</h2>
  <img class="plot" src="data:image/png;base64,{FIG2}" alt="Wellbeing heatmap"/>
</section>

<!-- Fig 3 -->
<section>
  <h2>💰 Figura 3 — Perfil de Asignación Presupuestal</h2>
  <img class="plot" src="data:image/png;base64,{FIG3}" alt="Budget allocation"/>
</section>

<!-- Fig 4 -->
<section>
  <h2>🏗️ Figura 4 — Capacidad de Infraestructura en el Tiempo</h2>
  <img class="plot" src="data:image/png;base64,{FIG4}" alt="Infrastructure capacity"/>
</section>

<!-- Comparison Table -->
<section>
  <h2>📋 Tabla Comparativa de Escenarios</h2>
  <table>
    <thead>
      <tr>
        <th>Escenario</th>
        <th>Bienestar Promedio (Año 10)</th>
        <th>Comunas Mejoradas (&gt;0.05)</th>
        <th>Capacidad Infraestructura (Año 10)</th>
        <th>Costo-Efectividad*</th>
      </tr>
    </thead>
    <tbody>
{TABLE_ROWS}
    </tbody>
  </table>
  <p style="color:var(--subtext);font-size:0.8rem;margin-top:0.5rem;">
    *Costo-efectividad = Bienestar medio año 10 / Inversión total (escala relativa)
  </p>
</section>

<!-- Ranking of Communes -->
<section>
  <h2>🏆 Top 5 Comunas Beneficiadas — Mejor Escenario</h2>
  <table>
    <thead>
      <tr>
        <th>Ranking</th>
        <th>Comuna</th>
        <th>IVC Inicial</th>
        <th>Bienestar Año 10</th>
        <th>Mejora Absoluta</th>
      </tr>
    </thead>
    <tbody>
{TOP5_ROWS}
    </tbody>
  </table>
</section>

<!-- Recommendation -->
<section>
  <h2>✅ Recomendación de Política Pública</h2>
  <div class="recommendation">
{RECOMMENDATION}
  </div>
</section>

<footer>
  <p>Generado automáticamente por el pipeline ABM — Mesa + Python</p>
  <p style="margin-top:0.3rem;">Universidad de Antioquia · Trabajo de Grado 2026 · Medellín, Colombia</p>
</footer>

</div>
</body>
</html>
"""

def build_kpi_cards(year10: pd.DataFrame, best_scenario: str) -> str:
    cards = []
    for s in SCENARIOS:
        sub = year10[year10["scenario"] == s]
        mw = sub["wellbeing"].mean()
        mic = sub["infrastructure_capacity"].mean()
        color = "#2ECC71" if s == best_scenario else "#58A6FF"
        label = SCENARIO_LABELS[s]
        cards.append(f"""    <div class="metric-card">
      <div class="value" style="color:{color};">{mw:.4f}</div>
      <div class="label">Bienestar Medio Año 10<br/>{label}</div>
    </div>
    <div class="metric-card">
      <div class="value" style="color:{color};">{mic:.4f}</div>
      <div class="label">Infraestructura Media Año 10<br/>{label}</div>
    </div>""")
    return "\n".join(cards)


def build_table_rows(df: pd.DataFrame, year10: pd.DataFrame) -> str:
    rows = []
    # baseline wellbeing (year 1 → initial)
    year1 = df[df["year"] == 1]
    for s in SCENARIOS:
        sub10 = year10[year10["scenario"] == s]
        sub1  = year1[year1["scenario"] == s]
        mw    = sub10["wellbeing"].mean()
        mic   = sub10["infrastructure_capacity"].mean()
        initial_wb = sub1["wellbeing"].mean()
        improved   = (sub10["wellbeing"].values - sub1["wellbeing"].values > 0.05).sum()
        n_t = len(sub10)    # número real de unidades territoriales en los datos
        ce = mw / (ANNUAL_BUDGET * 10 / 1e9)  # proxy
        label = SCENARIO_LABELS[s]
        is_best = (s == max(SCENARIOS, key=lambda x:
                            year10[year10["scenario"] == x]["wellbeing"].mean()))
        cls = ' class="best"' if is_best else ""
        rows.append(
            f'<tr{cls}>'
            f'<td>{label}</td>'
            f'<td>{mw:.4f}</td>'
            f'<td>{improved}/{n_t}</td>'
            f'<td>{mic:.4f}</td>'
            f'<td>{ce:.6f}</td>'
            f'</tr>'
        )
    return "\n".join(rows)


def build_top5_rows(df: pd.DataFrame, best_scenario: str) -> str:
    year10 = df[(df["year"] == df["year"].max()) & (df["scenario"] == best_scenario)].copy()
    year1  = df[(df["year"] == 1) & (df["scenario"] == best_scenario)].copy()
    merged = year10[[UT, "ivc_initial", "wellbeing"]].merge(
        year1[[UT, "wellbeing"]].rename(columns={"wellbeing": "wb_initial"}),
        on=UT
    )
    merged["improvement"] = merged["wellbeing"] - merged["wb_initial"]
    top5 = merged.nlargest(5, "improvement").reset_index(drop=True)
    rows = []
    for i, row in top5.iterrows():
        rows.append(
            f'<tr><td>{i+1}</td>'
            f'<td><strong>{row[UT]}</strong></td>'
            f'<td>{row["ivc_initial"]:.4f}</td>'
            f'<td>{row["wellbeing"]:.4f}</td>'
            f'<td class="best">+{row["improvement"]:.4f}</td></tr>'
        )
    return "\n".join(rows)


def build_recommendation(df: pd.DataFrame, best_scenario: str) -> str:
    n_territories = df[UT].nunique()
    ciudad_nombre = cfg.ciudad.nombre
    year10 = df[df["year"] == df["year"].max()]
    metrics = {}
    for s in SCENARIOS:
        sub = year10[year10["scenario"] == s]
        metrics[s] = {
            "mw": sub["wellbeing"].mean(),
            "mic": sub["infrastructure_capacity"].mean(),
        }
    best = best_scenario
    second = sorted(SCENARIOS, key=lambda x: -metrics[x]["mw"])[1]
    mw_best = metrics[best]["mw"]
    mw_dist = metrics["distribuido"]["mw"]
    delta = mw_best - mw_dist

    scenario_name = {
        "focalizado": "Focalizado (Escenario A)",
        "distribuido": "Distribuido (Escenario B)",
        "adaptativo":  "Adaptativo (Escenario C)",
    }[best]

    return f"""    <h3>Escenario óptimo: <strong style="color:#2ECC71;">{scenario_name}</strong></h3>
    <p>Con base en la simulación ABM de 10 años sobre los {n_territories} {cfg.unidad_territorial}s de {ciudad_nombre},
    el escenario <strong>{scenario_name}</strong> logra el mayor bienestar promedio al año 10
    (<strong>{mw_best:.4f}</strong>), superando al escenario Distribuido en
    <strong>{delta:+.4f}</strong> puntos.</p>
    <p style="margin-top:0.75rem;">
    Este resultado sugiere que la política de inversión pública debería priorizar
    {"las comunas con mayor Índice de Vulnerabilidad Compuesto (IVC), concentrando el 80% del presupuesto en las zonas más críticas como Popular, Villa Hermosa, Santa Cruz y Manrique" if best == "focalizado"
    else "una asignación adaptativa que repondera cada dos años según la tasa de mejora observada, garantizando eficiencia dinámica"
    if best == "adaptativo"
    else "una distribución equitativa que asegure un piso mínimo de inversión en cada comuna"}.
    </p>
    <p style="margin-top:0.75rem; color:var(--subtext);">
    <em>Nota: La simulación utiliza datos reales del Índice de Vulnerabilidad Compuesto (IVC) 
    del pipeline ML, con valores iniciales derivados de la última proyección disponible (2024).</em>
    </p>"""


def main():
    logger.info("=" * 60)
    logger.info("Scenario Analysis — Loading simulation results")
    logger.info("=" * 60)

    if not os.path.exists(RESULTS_PATH):
        logger.error(f"simulation_results.csv not found at {RESULTS_PATH}")
        logger.info("Running 04_urban_model.py first...")
        # Try to import and run the model directly
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "urban_model",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_urban_model.py")
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.main()
        except Exception as e:
            logger.error(f"Could not auto-run model: {e}")
            sys.exit(1)

    df = pd.read_csv(RESULTS_PATH)
    logger.info(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    year10 = df[df["year"] == df["year"].max()]
    for s in SCENARIOS:
        mw = year10[year10["scenario"] == s]["wellbeing"].mean()
        mic = year10[year10["scenario"] == s]["infrastructure_capacity"].mean()
        logger.info(f"Year-10 | {s:<15} → Wellbeing: {mw:.4f} | Infra: {mic:.4f}")

    best_scenario = max(
        SCENARIOS,
        key=lambda s: year10[year10["scenario"] == s]["wellbeing"].mean()
    )
    logger.info(f"Best scenario: {best_scenario}")

    logger.info("Generating Figure 1 — Wellbeing evolution ...")
    f1 = fig1_wellbeing_evolution(df)
    logger.info("Generating Figure 2 — Heatmap ...")
    f2 = fig2_wellbeing_heatmap(df)
    logger.info("Generating Figure 3 — Budget allocation ...")
    f3 = fig3_budget_allocation(df)
    logger.info("Generating Figure 4 — Infrastructure capacity ...")
    f4 = fig4_infrastructure_capacity(df)
    logger.info("All figures generated.")

    kpi_cards    = build_kpi_cards(year10, best_scenario)
    table_rows   = build_table_rows(df, year10)
    top5_rows    = build_top5_rows(df, best_scenario)
    recommendation = build_recommendation(df, best_scenario)

    html = HTML_TEMPLATE.format(
        FIG1=f1, FIG2=f2, FIG3=f3, FIG4=f4,
        KPI_CARDS=kpi_cards,
        TABLE_ROWS=table_rows,
        TOP5_ROWS=top5_rows,
        RECOMMENDATION=recommendation,
        CIUDAD=cfg.ciudad.nombre,
    )

    report_path = os.path.join(REPORTS_DIR, "scenario_report.html")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    logger.info(f"HTML report saved → {report_path}")

    # Summary for logging
    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    for s in SCENARIOS:
        mw = year10[year10["scenario"] == s]["wellbeing"].mean()
        logger.info(f"  {SCENARIO_LABELS[s]:<22}: Mean Wellbeing Year 10 = {mw:.4f}")
    logger.info(f"  Best scenario: {best_scenario.upper()}")
    logger.info(f"  Report: {report_path}")

    return df, best_scenario


if __name__ == "__main__":
    main()
