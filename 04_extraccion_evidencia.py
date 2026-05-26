"""
04_extraccion_evidencia.py
==========================
Extracción automática de evidencia gráfica y tabular para el Capítulo 4.

Tareas:
  1. fig_01_feature_importance.png  — Feature Importance XGBoost (300 DPI)
  2. resultados_simulacion_escenarios.csv — Simulación ABM (3 escenarios × 10 años)
  3. mapa_escenario_base.png        — Mapa coroplérico / burbujas, Año 10
  4. mapa_escenario_estrategico.png
  5. mapa_escenario_alternativo.png

Todos los archivos se guardan en la carpeta 'evidencia/' en la raíz del proyecto.
"""

import os
import sys
import pickle
import logging
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ── Directorio raíz y rutas ───────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config_loader import get_config

cfg    = get_config()
_abs   = cfg.rutas_absolutas()
UT     = cfg.unidad_territorial          # "comuna"
UT_COL = UT.title()                      # "Comuna"
CIUDAD = cfg.ciudad.nombre               # "Medellín"

OUT_DIR = ROOT / "evidencia"
OUT_DIR.mkdir(exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("evidencia")

log.info("=" * 65)
log.info(f"04_extraccion_evidencia.py  —  {CIUDAD}")
log.info("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# TAREA 1 — Feature Importance (300 DPI)
# ─────────────────────────────────────────────────────────────────────────────

log.info("\n[TAREA 1] Feature Importance XGBoost")

# Etiquetas académicas en español
FEATURE_LABELS = {
    "idx_desempleo":     "Tensión de Desempleo\n(IML)",
    "idx_habitat":       "Precariedad del Hábitat\n(ECV + IPM)",
    "idx_educacion":     "Vulnerabilidad Educativa\n(ECV + IPM)",
    "idx_riesgo_clima":  "Riesgo Climático\n(SIATA / IoT)",
    "idx_pobreza":       "Pobreza Multidimensional\n(IPM)",
    "idx_tejido_social": "Fragilidad del Tejido Social\n(ECV)",
    "año_normalizado":   "Tendencia Temporal\n(normalizada)",
}

# Paleta: índices sociales vs ambiental
SOCIAL_IDX = {"idx_desempleo", "idx_habitat", "idx_educacion",
              "idx_pobreza", "idx_tejido_social"}
CLIMA_IDX  = {"idx_riesgo_clima"}
TIME_IDX   = {"año_normalizado"}

model_path = Path(_abs.modelos) / "vulnerability_model.pkl"
if not model_path.exists():
    log.error(f"Modelo no encontrado: {model_path}")
    sys.exit(1)

with open(model_path, "rb") as fh:
    model = pickle.load(fh)

# Leer métricas para obtener lista de features en el orden exacto del entrenamiento
metrics_path = Path(_abs.modelos) / "cv_metrics.json"
with open(metrics_path, encoding="utf-8") as fh:
    metrics = json.load(fh)
feature_cols = metrics["features"]  # orden canónico

fi = model.feature_importances_
fi_df = (
    pd.DataFrame({"feature": feature_cols, "importance": fi})
    .sort_values("importance", ascending=True)
    .reset_index(drop=True)
)

# Colores: social=azul oscuro, clima=naranja, tiempo=gris
def _bar_color(feat):
    if feat in CLIMA_IDX:
        return "#d95f02"       # naranja
    if feat in TIME_IDX:
        return "#7f7f7f"       # gris
    return "#1a4f8a"           # azul académico

colors = [_bar_color(f) for f in fi_df["feature"]]
labels = [FEATURE_LABELS.get(f, f) for f in fi_df["feature"]]
values = fi_df["importance"].values

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.65)

# Anotaciones de valor
for bar, val in zip(bars, values):
    ax.text(
        val + 0.002, bar.get_y() + bar.get_height() / 2,
        f"{val:.3f}",
        va="center", ha="left", fontsize=9, color="#333333",
    )

# Línea de media
mean_fi = values.mean()
ax.axvline(mean_fi, color="#e74c3c", linestyle="--", linewidth=1.3,
           label=f"Media ({mean_fi:.3f})")

# Leyenda de categorías
legend_patches = [
    mpatches.Patch(color="#1a4f8a", label="Índice social (ECV / IPM / IML)"),
    mpatches.Patch(color="#d95f02", label="Riesgo climático (SIATA / IoT)"),
    mpatches.Patch(color="#7f7f7f", label="Tendencia temporal"),
    plt.Line2D([0], [0], color="#e74c3c", linestyle="--", linewidth=1.3,
               label=f"Media global ({mean_fi:.3f})"),
]
ax.legend(handles=legend_patches, loc="lower right", fontsize=8.5,
          framealpha=0.92)

ax.set_xlabel("XGBoost Gain Importance (mayor valor = mayor influencia sobre el IVC)",
              fontsize=10)
ax.set_title(
    f"Importancia de Variables — Modelo XGBoost de Vulnerabilidad Urbana\n"
    f"{CIUDAD}, {cfg.ciudad.departamento} · Validación cruzada temporal 2007–2024",
    fontsize=11, fontweight="bold", pad=12,
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlim(0, values.max() * 1.18)
fig.tight_layout()

fi_out = OUT_DIR / "fig_01_feature_importance.png"
fig.savefig(fi_out, dpi=300, bbox_inches="tight")
plt.close(fig)
log.info(f"  -> Guardado: {fi_out}")


# ─────────────────────────────────────────────────────────────────────────────
# TAREA 2 — Simulación ABM (3 escenarios × 10 años)
# ─────────────────────────────────────────────────────────────────────────────

log.info("\n[TAREA 2] Simulación ABM — 3 escenarios")

# 04_urban_model.py empieza con dígito → import con importlib
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "urban_model", str(ROOT / "abm" / "04_urban_model.py")
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

UrbanModel       = _mod.UrbanModel
load_ivc_from_csv = _mod.load_ivc_from_csv
SCENARIOS        = _mod.SCENARIOS
N_YEARS          = _mod.N_YEARS

CSV_PATH = os.path.join(_abs.datos_sociales, "processed", "indices_comunas.csv")
ivc_data = load_ivc_from_csv(CSV_PATH)

all_dfs = []
for scenario in SCENARIOS:
    log.info(f"  Ejecutando escenario: {scenario} ...")
    m   = UrbanModel(strategy=scenario, ivc_data=ivc_data, seed=42)
    df  = m.run_simulation()
    all_dfs.append(df)

sim_results = pd.concat(all_dfs, ignore_index=True)

sim_out = OUT_DIR / "resultados_simulacion_escenarios.csv"
sim_results.to_csv(sim_out, index=False, encoding="utf-8-sig")
log.info(f"  -> Guardado: {sim_out}  ({len(sim_results):,} filas)")


# ─────────────────────────────────────────────────────────────────────────────
# TAREA 3 — Mapas coropléticos (Año 10 de cada escenario)
# ─────────────────────────────────────────────────────────────────────────────

log.info("\n[TAREA 3] Mapas coropléticos — Año 10")

# ── Intentar cargar GeoJSON real ──────────────────────────────────────────────
geo_dir = Path(_abs.geo)
geo_dir.mkdir(parents=True, exist_ok=True)

GEO_CANDIDATES = [
    geo_dir / f"{UT}s.geojson",
    geo_dir / f"{UT}s_{CIUDAD.lower()}.geojson",
    geo_dir / "comunas.geojson",
]

gdf = None
geo_mode = "bubble"

try:
    import geopandas as gpd
    for geofile in GEO_CANDIDATES:
        if geofile.exists():
            gdf = gpd.read_file(str(geofile))
            log.info(f"  GeoJSON cargado: {geofile}  ({len(gdf)} polígonos)")
            geo_mode = "choropleth"
            break

    if gdf is None:
        # Intentar descarga pública (Medellín — DANE / CartoDB)
        PUBLIC_URLS = [
            "https://raw.githubusercontent.com/guillermobustosm/colombia-municipios/main/comunas_medellin.geojson",
            "https://opendata.arcgis.com/datasets/a3e5f6a5b1b24a0f9e5bcd7e8f5a9a41_0.geojson",
        ]
        for url in PUBLIC_URLS:
            try:
                log.info(f"  Intentando descarga: {url}")
                gdf = gpd.read_file(url)
                local = geo_dir / f"{UT}s.geojson"
                gdf.to_file(str(local), driver="GeoJSON")
                log.info(f"  GeoJSON descargado y guardado en {local}")
                geo_mode = "choropleth"
                break
            except Exception as e:
                log.warning(f"  No se pudo descargar ({e})")

except ImportError:
    log.warning("  geopandas no disponible — usando mapa de burbujas")

# ── Tabla de centroides desde config.yaml ─────────────────────────────────────
centroides = {
    t.nombre: (t.lon, t.lat)
    for t in cfg.territorios
}
# También agregar desde vulnerability_ranking
ranking_path = Path(_abs.modelos) / "vulnerability_ranking.csv"
rank_df = pd.read_csv(ranking_path)
# Normalizar nombre de columna
col_ut = UT_COL if UT_COL in rank_df.columns else rank_df.columns[0]
rank_df = rank_df.rename(columns={col_ut: "territorio", "mean_ivc": "ivc_base"})

# ── Extraer datos Año 10 por escenario ────────────────────────────────────────
SCENARIO_META = {
    "focalizado":  ("Escenario A — Focalizado",    "#c0392b", "mapa_escenario_base.png"),
    "distribuido": ("Escenario B — Distribuido",   "#27ae60", "mapa_escenario_estrategico.png"),
    "adaptativo":  ("Escenario C — Adaptativo",    "#2980b9", "mapa_escenario_alternativo.png"),
}

year10 = sim_results[sim_results["year"] == N_YEARS].copy()

# Determinar nombre de columna de territorio en sim_results
ut_sim_col = UT if UT in year10.columns else "comuna"

CMAP_NAME = "RdYlGn_r"   # rojo=alta vulnerabilidad, verde=baja

for scenario, (title, accent_color, fname) in SCENARIO_META.items():
    df_s = year10[year10["scenario"] == scenario].copy()
    df_s = df_s.rename(columns={ut_sim_col: "territorio"})

    vmin = df_s["ivc_current"].min()
    vmax = df_s["ivc_current"].max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(CMAP_NAME)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_facecolor("#f0f4f8")
    fig.patch.set_facecolor("white")

    if geo_mode == "choropleth" and gdf is not None:
        # ── MAPA COROPLÉRICO REAL ─────────────────────────────────────────
        # Buscar columna de nombre en GDF
        name_col = None
        for cand in ["NOMBRE", "nombre", "NOMB_COM", "COM_NOMBRE", "name", "Name"]:
            if cand in gdf.columns:
                name_col = cand
                break
        if name_col is None:
            name_col = gdf.columns[0]

        merged = gdf.merge(
            df_s[["territorio", "ivc_current", "wellbeing"]],
            left_on=name_col, right_on="territorio", how="left",
        )
        merged["ivc_current"] = merged["ivc_current"].fillna(merged["ivc_current"].mean())

        merged.plot(
            column="ivc_current",
            cmap=CMAP_NAME,
            linewidth=0.5,
            edgecolor="#ffffff",
            ax=ax,
            legend=False,
            vmin=vmin,
            vmax=vmax,
        )
        # Etiquetas
        for _, row in merged.iterrows():
            if row.geometry is not None:
                cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
                label = str(row.get(name_col, ""))[:8]
                ax.annotate(label, (cx, cy), fontsize=5.5, ha="center",
                            color="white", fontweight="bold")

    else:
        # ── MAPA DE BURBUJAS (fallback) ────────────────────────────────────
        df_plot = df_s.merge(
            pd.DataFrame(
                [(k, v[0], v[1]) for k, v in centroides.items()],
                columns=["territorio", "lon", "lat"],
            ),
            on="territorio", how="left",
        ).dropna(subset=["lon", "lat"])

        # Fondo: polígono aprox de Medellín con scatter de relleno
        sc = ax.scatter(
            df_plot["lon"], df_plot["lat"],
            c=df_plot["ivc_current"],
            s=(df_plot["ivc_current"] * 1800 + 300),
            cmap=CMAP_NAME, norm=norm,
            edgecolors="#555555", linewidths=0.7, alpha=0.88, zorder=5,
        )

        # Etiquetas de cada territorio
        for _, row in df_plot.iterrows():
            ivc_val = row["ivc_current"]
            font_color = "white" if ivc_val > (vmin + (vmax - vmin) * 0.4) else "#222"
            ax.annotate(
                row["territorio"],
                (row["lon"], row["lat"]),
                fontsize=6.8, ha="center", va="center",
                color=font_color, fontweight="bold", zorder=6,
            )

        # Ajustar límites con margen
        lons = df_plot["lon"].values
        lats = df_plot["lat"].values
        margin_x = (lons.max() - lons.min()) * 0.12 + 0.01
        margin_y = (lats.max() - lats.min()) * 0.12 + 0.01
        ax.set_xlim(lons.min() - margin_x, lons.max() + margin_x)
        ax.set_ylim(lats.min() - margin_y, lats.max() + margin_y)
        ax.set_xlabel("Longitud", fontsize=9)
        ax.set_ylabel("Latitud", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # ── Colorbar ───────────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, shrink=0.75)
    cbar.set_label("IVC — Índice de Vulnerabilidad Compuesto (Año 10)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # ── Tabla resumen (Top 5 más vulnerables) ────────────────────────────
    top5 = df_s.nlargest(5, "ivc_current")[["territorio", "ivc_current", "wellbeing"]]
    top5 = top5.reset_index(drop=True)
    table_text = "Top 5 más vulnerables\n"
    table_text += "-" * 35 + "\n"
    for _, r in top5.iterrows():
        table_text += f"{r['territorio']:<20}  IVC={r['ivc_current']:.3f}\n"
    ax.text(
        0.01, 0.01, table_text,
        transform=ax.transAxes, fontsize=7.5, verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85),
        fontfamily="monospace",
    )

    # ── Estadísticas en la esquina superior ────────────────────────────────
    ivc_mean = df_s["ivc_current"].mean()
    wb_mean  = df_s["wellbeing"].mean()
    stats_text = (
        f"IVC medio: {ivc_mean:.3f}\n"
        f"Bienestar medio: {wb_mean:.3f}\n"
        f"N {UT}s: {len(df_s)}"
    )
    ax.text(
        0.99, 0.99, stats_text,
        transform=ax.transAxes, fontsize=8.5, verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=accent_color,
                  alpha=0.82, edgecolor="white"),
        color="white", fontweight="bold",
    )

    # ── Títulos ────────────────────────────────────────────────────────────
    ax.set_title(
        f"{title}\nVulnerabilidad por {UT_COL} — {CIUDAD} · Proyección Año 10",
        fontsize=12, fontweight="bold", pad=10,
    )

    fig.tight_layout()
    out_path = OUT_DIR / fname
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  -> Guardado: {out_path}")

# ── Resumen final ─────────────────────────────────────────────────────────────
log.info("\n" + "=" * 65)
log.info("RESUMEN DE ARCHIVOS GENERADOS EN 'evidencia/'")
log.info("=" * 65)
for f in sorted(OUT_DIR.iterdir()):
    size_kb = f.stat().st_size / 1024
    log.info(f"  {f.name:<48}  {size_kb:7.1f} KB")
log.info("=" * 65)
log.info("04_extraccion_evidencia.py  —  COMPLETADO")
