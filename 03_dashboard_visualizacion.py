"""
03_dashboard_visualizacion.py
Dashboard de Ciudades Inteligentes — Plataforma Multi-Ciudad
Ejecutar con:  py -m streamlit run 03_dashboard_visualizacion.py

Funcionalidades:
  · Selector de perfil de ciudad (Medellín / Bogotá / personalizado)
  · Carga de config.yaml personalizado vía sidebar
  · Todos los textos, etiquetas y popups son dinámicos (unidad territorial del config)
  · Botón "Ejecutar Pipeline Automático" que reentrena el modelo XGBoost
"""

import io
import os
import sys
import json
import pickle
import shutil
import warnings
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── Detección de entorno cloud ─────────────────────────────────────────────────
# En Streamlit Community Cloud la carpeta de telemetría cruda nunca existe.
# Usamos esta flag para deshabilitar el botón de pipeline y evitar errores.
_IS_CLOUD = not (ROOT / "pluviometrica").exists()

# ── Perfiles de ciudad predefinidos ───────────────────────────────────────────
CITY_PROFILES = {
    "Medellin": {
        "label":  "Medellin (predeterminado)",
        "config": str(ROOT / "config.yaml"),
        "icon":   "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/"
                  "Escudo_de_Medell%C3%ADn.svg/200px-Escudo_de_Medell%C3%ADn.svg.png",
    },
    "Bogota": {
        "label":  "Bogota (demo)",
        "config": str(ROOT / "config_bogota.yaml"),   # creado si no existe
        "icon":   "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/"
                  "Escudo_de_Bogot%C3%A1.svg/200px-Escudo_de_Bogot%C3%A1.svg.png",
    },
    "Personalizado": {
        "label":  "Ciudad personalizada (subir config.yaml)",
        "config": None,
        "icon":   None,
    },
}

# ── Página ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plataforma Ciudades Inteligentes",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-header{font-size:2rem;font-weight:800;color:#1a237e;
  border-bottom:3px solid #3949ab;padding-bottom:.4rem;margin-bottom:.2rem;}
.sub-header{font-size:.95rem;color:#546e7a;margin-bottom:1.2rem;}
.kpi-card{background:linear-gradient(135deg,#1a237e 0%,#3949ab 100%);
  border-radius:12px;padding:1.2rem 1.4rem;color:white;
  box-shadow:0 4px 14px rgba(26,35,126,.2);}
.kpi-label{font-size:.78rem;opacity:.85;text-transform:uppercase;letter-spacing:.05em;}
.kpi-value{font-size:2rem;font-weight:700;line-height:1.2;}
.kpi-sub{font-size:.75rem;opacity:.75;margin-top:.2rem;}
.section-title{font-size:1.1rem;font-weight:700;color:#1a237e;
  margin-top:.5rem;margin-bottom:.3rem;}
.pipeline-box{background:#e8f5e9;border-left:4px solid #388e3c;
  border-radius:0 8px 8px 0;padding:1rem 1.2rem;font-size:.85rem;}
div[data-testid="stTabs"] button{font-size:1rem;font-weight:600;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — creación de config demo para Bogotá
# ══════════════════════════════════════════════════════════════════════════════

def _create_bogota_demo(path: Path):
    """Genera un config.yaml de demo para Bogotá si no existe."""
    if path.exists():
        return
    content = """\
ciudad:
  nombre: "Bogota"
  departamento: "Cundinamarca"
  codigo_dane: "11001"
  unidad_territorial: "localidad"
  crs: "EPSG:4326"
  centro:
    lat: 4.7110
    lon: -74.0721
rutas:
  datos_climaticos: "pluviometrica"
  datos_sociales: "ModeloDatos/ModeloDatos/data"
  procesados: "data/processed"
  modelos: "models"
  logs: "logs"
  geo: "data/geo"
territorios:
  - {id: 1,  nombre: "Usaquen",          lat: 4.7016, lon: -74.0317}
  - {id: 2,  nombre: "Chapinero",        lat: 4.6464, lon: -74.0603}
  - {id: 3,  nombre: "Santa Fe",         lat: 4.5980, lon: -74.0752}
  - {id: 4,  nombre: "San Cristobal",    lat: 4.5643, lon: -74.0768}
  - {id: 5,  nombre: "Usme",             lat: 4.4795, lon: -74.1278}
  - {id: 6,  nombre: "Tunjuelito",       lat: 4.5680, lon: -74.1280}
  - {id: 7,  nombre: "Bosa",             lat: 4.6197, lon: -74.1960}
  - {id: 8,  nombre: "Kennedy",          lat: 4.6280, lon: -74.1536}
  - {id: 9,  nombre: "Fontibon",         lat: 4.6747, lon: -74.1470}
  - {id: 10, nombre: "Engativa",         lat: 4.7010, lon: -74.1140}
  - {id: 11, nombre: "Suba",             lat: 4.7590, lon: -74.0840}
  - {id: 12, nombre: "Barrios Unidos",   lat: 4.6696, lon: -74.0800}
  - {id: 13, nombre: "Teusaquillo",      lat: 4.6513, lon: -74.0830}
  - {id: 14, nombre: "Martires",         lat: 4.6080, lon: -74.0951}
  - {id: 15, nombre: "Antonio Narino",   lat: 4.5958, lon: -74.1039}
  - {id: 16, nombre: "Puente Aranda",    lat: 4.6272, lon: -74.1110}
  - {id: 17, nombre: "Candelaria",       lat: 4.5966, lon: -74.0737}
  - {id: 18, nombre: "Rafael Uribe",     lat: 4.5639, lon: -74.1091}
  - {id: 19, nombre: "Ciudad Bolivar",   lat: 4.5200, lon: -74.1560}
  - {id: 20, nombre: "Sumapaz",          lat: 4.1933, lon: -74.3594}
catalogo_estaciones: {}
"""
    path.write_text(content, encoding="utf-8")


_create_bogota_demo(Path(CITY_PROFILES["Bogota"]["config"]))


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — selección de ciudad y pipeline
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## Configuracion de Ciudad")
    st.markdown("---")

    # ── Selector de perfil ────────────────────────────────────────────────────
    profile_keys   = list(CITY_PROFILES.keys())
    profile_labels = [CITY_PROFILES[k]["label"] for k in profile_keys]
    selected_idx   = st.selectbox(
        "Perfil de ciudad",
        range(len(profile_keys)),
        format_func=lambda i: profile_labels[i],
        key="city_profile_idx",
    )
    selected_profile = profile_keys[selected_idx]

    # ── Carga personalizada ───────────────────────────────────────────────────
    custom_config_path = None
    if selected_profile == "Personalizado":
        uploaded = st.file_uploader(
            "Subir config.yaml",
            type=["yaml", "yml"],
            help="El archivo debe seguir la estructura de config.yaml del proyecto.",
        )
        if uploaded is not None:
            tmp_dir  = Path(tempfile.mkdtemp())
            tmp_cfg  = tmp_dir / "config.yaml"
            tmp_cfg.write_bytes(uploaded.read())
            custom_config_path = str(tmp_cfg)
            st.success(f"config.yaml cargado correctamente.")
        else:
            st.info("Sube un archivo config.yaml para continuar.")

    # ── Determinar ruta de config activa ─────────────────────────────────────
    if selected_profile == "Personalizado":
        active_config_path = custom_config_path or str(ROOT / "config.yaml")
    else:
        active_config_path = CITY_PROFILES[selected_profile]["config"]

    # ── Cargar config ─────────────────────────────────────────────────────────
    try:
        from config_loader import get_config
        cfg    = get_config(active_config_path)
        UT     = cfg.unidad_territorial                   # "comuna" / "localidad"
        UT_COL = UT.title()                               # "Comuna" / "Localidad"
        UT_PLU = UT_COL + "s"                             # "Comunas" / "Localidades"
        CIUDAD = cfg.ciudad.nombre
        _abs   = cfg.rutas_absolutas()
        config_ok = True
    except Exception as cfg_err:
        st.error(f"Error en config.yaml: {cfg_err}")
        UT, UT_COL, UT_PLU, CIUDAD = "comuna", "Comuna", "Comunas", "Ciudad"
        _abs       = None
        config_ok  = False

    # ── Logo de ciudad ────────────────────────────────────────────────────────
    icon_url = CITY_PROFILES[selected_profile].get("icon")
    if icon_url:
        st.image(icon_url, width=64)

    st.markdown(f"**Ciudad activa:** {CIUDAD}")
    st.markdown(f"**Unidad territorial:** `{UT}`")
    st.markdown("---")

    # ── Filtros ───────────────────────────────────────────────────────────────
    st.markdown("### Filtros de visualizacion")

    # (se rellenan después de cargar los datos)
    _filter_placeholder = st.empty()

    st.markdown("---")

    # ── BOTON PIPELINE ────────────────────────────────────────────────────────
    st.markdown("### Automatizacion")
    if _IS_CLOUD:
        # En Streamlit Community Cloud no hay datos de telemetría cruda.
        # El modelo ya viene pre-entrenado en el repositorio.
        run_pipeline = False
        st.info(
            "**Modo nube activo.**\n\n"
            "El reentrenamiento requiere datos de telemetría local (20 GB) "
            "no disponibles en este entorno. El modelo pre-entrenado "
            "ya está cargado y listo para visualización.",
            icon="☁️",
        )
    else:
        run_pipeline = st.button(
            "Ejecutar Pipeline Automatico",
            type="primary",
            use_container_width=True,
            help=(
                "Ejecuta en secuencia:\n"
                "1. Procesador SIATA (bigdata)\n"
                "2. Indices de vulnerabilidad (ML)\n"
                "3. Reentrenamiento XGBoost\n\n"
                "Al finalizar, el dashboard se recarga con los nuevos datos."
            ),
        )

    st.markdown("---")
    st.caption("Trabajo de Grado 2026 · UdeA")


# ══════════════════════════════════════════════════════════════════════════════
# EJECUTAR PIPELINE (si el botón fue presionado)
# ══════════════════════════════════════════════════════════════════════════════

if run_pipeline:
    pipeline_steps = [
        ("Procesador SIATA",           str(ROOT / "bigdata" / "01_siata_processor.py")),
        ("Indices de Vulnerabilidad",  str(ROOT / "ml"      / "02_vulnerability_indices.py")),
        ("Reentrenamiento XGBoost",    str(ROOT / "ml"      / "03_xgboost_model.py")),
    ]

    st.markdown("---")
    st.markdown("### Pipeline en ejecucion")
    overall_ok = True

    for step_name, script_path in pipeline_steps:
        with st.status(f"Ejecutando: {step_name}...", expanded=True) as status:
            try:
                proc = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=str(ROOT),
                    env={**os.environ, "PYTHONPATH": str(ROOT)},
                    timeout=600,
                )
                stdout = proc.stdout or ""
                stderr = proc.stderr or ""

                if proc.returncode == 0:
                    status.update(label=f"✅ {step_name}", state="complete")
                    if stdout.strip():
                        with st.expander("Salida del script", expanded=False):
                            st.code(stdout[-4000:], language="text")
                else:
                    status.update(label=f"❌ {step_name}", state="error")
                    st.error(f"El script terminó con código {proc.returncode}")
                    if stderr.strip():
                        st.code(stderr[-3000:], language="text")
                    overall_ok = False
                    break

            except subprocess.TimeoutExpired:
                status.update(label=f"⏱️ {step_name} (timeout)", state="error")
                st.error("El script tardó más de 10 minutos. Cancela y verifica los datos.")
                overall_ok = False
                break
            except Exception as ex:
                status.update(label=f"❌ {step_name}", state="error")
                st.error(f"Error inesperado: {ex}")
                overall_ok = False
                break

    if overall_ok:
        st.success(
            "Pipeline completado. Los modelos han sido reentrenados con los datos "
            f"de {CIUDAD}. Limpiando cache y recargando..."
        )
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    else:
        st.warning(
            "El pipeline se detuvo con errores. Revisa los mensajes anteriores "
            "y verifica que los datos de entrada existen en las rutas del config."
        )
    st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# RUTAS DE ARTEFACTOS (dinámicas desde config)
# ══════════════════════════════════════════════════════════════════════════════

if config_ok:
    MODELS_DIR   = Path(_abs.modelos)
    DATA_PROC    = Path(_abs.datos_sociales) / "processed"
    PROC_DIR     = Path(_abs.procesados)
    GEO_DIR      = Path(_abs.geo)
else:
    MODELS_DIR = ROOT / "models"
    DATA_PROC  = ROOT / "ModeloDatos" / "ModeloDatos" / "data" / "processed"
    PROC_DIR   = ROOT / "data" / "processed"
    GEO_DIR    = ROOT / "data" / "geo"

MODEL_PATH   = MODELS_DIR / "vulnerability_model.pkl"
METRICS_PATH = MODELS_DIR / "cv_metrics.json"
RANKING_PATH = MODELS_DIR / "vulnerability_ranking.csv"
SHAP_PATH    = MODELS_DIR / "shap_values.csv"
INDICES_PATH = DATA_PROC  / "indices_comunas.csv"

GEO_CANDIDATES = [
    GEO_DIR / f"{UT}s.geojson",
    GEO_DIR / f"{UT}s_{CIUDAD.lower()}.geojson",
    GEO_DIR / f"{UT}s.gpkg",
    ROOT    / "data" / "comunas.geojson",
    ROOT    / "data" / "comunas_medellin.geojson",
]


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS (con cache por ruta de config)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=f"Cargando datos...", ttl=300)
def load_indices(path: str, ut_col: str) -> pd.DataFrame | None:
    df = pd.read_csv(path, encoding="utf-8-sig")
    # Normaliza el nombre de la columna de UT al esperado
    col_found = ut_col if ut_col in df.columns else (
        next((c for c in df.columns if c.lower() in
              ("comuna", "localidad", "corregimiento", "upz")), None)
    )
    if col_found and col_found != ut_col:
        df = df.rename(columns={col_found: ut_col})
    years = df["Año"].values.astype(float)
    df["año_normalizado"] = (years - years.min()) / max(years.max() - years.min(), 1)
    return df


@st.cache_data(show_spinner=False, ttl=300)
def load_ranking(path: str, ut_col: str) -> pd.DataFrame | None:
    df = pd.read_csv(path, encoding="utf-8-sig")
    col_found = next(
        (c for c in df.columns if c.lower() in ("comuna", "localidad", ut_col.lower())),
        df.columns[0],
    )
    if col_found != ut_col:
        df = df.rename(columns={col_found: ut_col})
    return df


@st.cache_data(show_spinner=False, ttl=300)
def load_shap(path: str) -> pd.DataFrame | None:
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_data(show_spinner=False, ttl=300)
def load_metrics(path: str) -> dict | None:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False, ttl=300)
def load_geodata(candidates: list[str]) -> tuple:
    try:
        import geopandas as gpd
        for path in candidates:
            if Path(path).exists():
                return gpd.read_file(path), "local"
    except ImportError:
        pass
    return None, "none"


# ── Cargar artefactos ─────────────────────────────────────────────────────────
errors: list[str] = []

df_indices = df_ranking = df_shap = metrics = model = None

try:
    df_indices = load_indices(str(INDICES_PATH), UT_COL)
except Exception as e:
    errors.append(f"indices_comunas.csv: {e}")

try:
    df_ranking = load_ranking(str(RANKING_PATH), UT_COL)
except Exception as e:
    errors.append(f"vulnerability_ranking.csv: {e}")

try:
    df_shap = load_shap(str(SHAP_PATH))
except Exception as e:
    errors.append(f"shap_values.csv: {e}")

try:
    metrics = load_metrics(str(METRICS_PATH))
except Exception as e:
    errors.append(f"cv_metrics.json: {e}")

try:
    model = load_model(str(MODEL_PATH))
except Exception as e:
    errors.append(f"vulnerability_model.pkl: {e}")

gdf, geo_source = load_geodata([str(c) for c in GEO_CANDIDATES])

# ── Constantes de features ─────────────────────────────────────────────────────
FEATURE_COLS = [
    "idx_desempleo", "idx_habitat", "idx_educacion",
    "idx_riesgo_clima", "idx_pobreza", "idx_tejido_social", "año_normalizado",
]
FEATURE_LABELS = {
    "idx_desempleo":     "Desempleo",
    "idx_habitat":       "Habitat / Vivienda",
    "idx_educacion":     "Educacion",
    "idx_riesgo_clima":  "Riesgo Climatico (SIATA)",
    "idx_pobreza":       "Pobreza",
    "idx_tejido_social": "Tejido Social",
    "año_normalizado":   "Tendencia Temporal",
}
PRIORITY_COLORS = {"Alta": "#d32f2f", "Media": "#f57c00", "Baja": "#388e3c"}

# ── Predicciones ──────────────────────────────────────────────────────────────
if model is not None and df_indices is not None:
    latest_year = int(df_indices["Año"].max())
    df_latest   = df_indices[df_indices["Año"] == latest_year].copy()
    missing     = [c for c in FEATURE_COLS if c not in df_latest.columns]
    df_latest["ivc_pred"] = (
        model.predict(df_latest[FEATURE_COLS].values)
        if not missing
        else df_latest.get("ivc", np.nan)
    )
else:
    latest_year = None
    df_latest   = pd.DataFrame()

if not df_latest.empty and "ivc_pred" in df_latest.columns:
    q33 = df_latest["ivc_pred"].quantile(0.33)
    q66 = df_latest["ivc_pred"].quantile(0.66)
    df_latest["prioridad"] = pd.cut(
        df_latest["ivc_pred"],
        bins=[-np.inf, q33, q66, np.inf],
        labels=["Baja", "Media", "Alta"],
    )
else:
    q33 = q66 = 0.0


# ── Rellenar filtros en sidebar ────────────────────────────────────────────────
with _filter_placeholder.container():
    if df_indices is not None:
        all_years    = sorted(df_indices["Año"].unique(), reverse=True)
        selected_year = st.selectbox("Año de analisis", all_years, index=0)
    else:
        selected_year = latest_year

    if df_ranking is not None:
        n_top = st.slider(f"Top {UT_PLU} vulnerables", 5, max(5, len(df_ranking)), 10)
    else:
        n_top = 10

    if metrics:
        st.markdown("**Metricas del modelo**")
        st.metric("R² CV promedio",      f"{metrics['mean_R2']:.4f}")
        st.metric("RMSE CV promedio",    f"{metrics['mean_RMSE']:.4f}")
        st.metric("R² datos completos",  f"{metrics['full_data_R2']:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# ENCABEZADO PRINCIPAL (dinámico)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    f'<p class="main-header">Plataforma de Ciudades Inteligentes — {CIUDAD}</p>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<p class="sub-header">Modelo predictivo de vulnerabilidad y priorizacion estrategica '
    f'de inversion por {UT_PLU.lower()} · XGBoost + SIATA · {CIUDAD}</p>',
    unsafe_allow_html=True,
)

if errors:
    with st.expander("Advertencias de carga de archivos", expanded=False):
        for e in errors:
            st.warning(e)


# ══════════════════════════════════════════════════════════════════════════════
# PESTAÑAS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    f"Vision General",
    f"Mapa Predictivo",
    f"Interpretabilidad del Modelo",
])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — VISIÓN GENERAL
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown(
        f'<p class="section-title">Indicadores Clave — {CIUDAD}</p>',
        unsafe_allow_html=True,
    )

    if df_ranking is not None and not df_latest.empty:
        mean_ivc   = df_latest["ivc_pred"].mean() if "ivc_pred" in df_latest.columns else df_ranking["mean_ivc"].mean()
        max_ivc    = df_ranking["mean_ivc"].max()
        min_ivc    = df_ranking["mean_ivc"].min()
        n_terr     = df_ranking[UT_COL].nunique() if UT_COL in df_ranking.columns else len(df_ranking)
        alta_count = int((df_latest["prioridad"] == "Alta").sum()) if "prioridad" in df_latest.columns else "—"
        worst      = df_ranking.iloc[0][UT_COL] if UT_COL in df_ranking.columns else "—"

        col1, col2, col3, col4 = st.columns(4)
        kpis = [
            ("IVC Promedio Ciudad",          f"{mean_ivc:.3f}", "Indice de Vulnerabilidad Compuesto"),
            ("IVC Maximo",                   f"{max_ivc:.3f}",  f"{UT_COL} mas vulnerable: {worst}"),
            (f"{UT_PLU} Prioritarias",       str(alta_count),   "Prioridad Alta (cuartil superior)"),
            (f"Total {UT_PLU} Analizadas",   str(n_terr),       f"IVC min: {min_ivc:.3f}"),
        ]
        for col, (label, value, sub) in zip([col1, col2, col3, col4], kpis):
            with col:
                st.markdown(
                    f'<div class="kpi-card">'
                    f'<div class="kpi-label">{label}</div>'
                    f'<div class="kpi-value">{value}</div>'
                    f'<div class="kpi-sub">{sub}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown(
                f'<p class="section-title">Ranking de Vulnerabilidad por {UT_COL}</p>',
                unsafe_allow_html=True,
            )
            top_df = df_ranking.head(n_top).copy()
            ut_vals = top_df[UT_COL] if UT_COL in top_df.columns else top_df.iloc[:, 0]
            top_df["color"] = top_df["mean_ivc"].apply(
                lambda v: "#d32f2f" if v >= q66 else ("#f57c00" if v >= q33 else "#388e3c")
            )
            fig_rank = go.Figure(go.Bar(
                y=ut_vals[::-1],
                x=top_df["mean_ivc"][::-1],
                orientation="h",
                marker_color=top_df["color"][::-1].tolist(),
                text=[f"{v:.3f}" for v in top_df["mean_ivc"][::-1]],
                textposition="outside",
                hovertemplate=f"<b>%{{y}}</b><br>IVC medio: %{{x:.4f}}<extra></extra>",
            ))
            fig_rank.update_layout(
                xaxis_title="IVC Medio Historico",
                yaxis_title="",
                height=420,
                margin=dict(l=10, r=60, t=10, b=40),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(showgrid=True, gridcolor="#e0e0e0"),
            )
            st.plotly_chart(fig_rank, use_container_width=True)

        with col_right:
            st.markdown(
                '<p class="section-title">Distribucion de Prioridades</p>',
                unsafe_allow_html=True,
            )
            if "prioridad" in df_latest.columns:
                prio_counts = df_latest["prioridad"].value_counts().reset_index()
                prio_counts.columns = ["Prioridad", UT_PLU]
                fig_pie = px.pie(
                    prio_counts,
                    names="Prioridad",
                    values=UT_PLU,
                    color="Prioridad",
                    color_discrete_map=PRIORITY_COLORS,
                    hole=0.45,
                    title=f"Prioridad por {UT_COL}",
                )
                fig_pie.update_layout(
                    height=220, margin=dict(l=0, r=0, t=30, b=0),
                    showlegend=True, legend=dict(orientation="h", y=-0.15),
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown(
                '<p class="section-title">Evolucion Temporal del IVC</p>',
                unsafe_allow_html=True,
            )
            if df_indices is not None and UT_COL in df_indices.columns:
                top5 = (df_ranking.head(5)[UT_COL].tolist()
                        if UT_COL in df_ranking.columns else [])
                df_evo = df_indices[df_indices[UT_COL].isin(top5)]
                fig_evo = px.line(
                    df_evo, x="Año", y="ivc", color=UT_COL, markers=True,
                    labels={"ivc": "IVC", "Año": "Año", UT_COL: UT_COL},
                )
                fig_evo.update_layout(
                    height=200, margin=dict(l=0, r=0, t=0, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    legend=dict(font=dict(size=9), y=1.0),
                    yaxis=dict(showgrid=True, gridcolor="#e0e0e0"),
                )
                st.plotly_chart(fig_evo, use_container_width=True)

        st.markdown(
            f'<p class="section-title">Detalle por Indice Social — {UT_COL} (ultimo año disponible)</p>',
            unsafe_allow_html=True,
        )
        if not df_latest.empty and UT_COL in df_latest.columns:
            cols_show = (
                [UT_COL]
                + [c for c in FEATURE_COLS[:-1] if c in df_latest.columns]
                + ["ivc_pred", "prioridad"]
            )
            df_show = (
                df_latest[cols_show]
                .rename(columns={**FEATURE_LABELS, "ivc_pred": "IVC Predicho", "prioridad": "Prioridad"})
                .sort_values("IVC Predicho", ascending=False)
                .reset_index(drop=True)
            )

            def color_priority(val):
                return {
                    "Alta":  "background-color:#ffcdd2",
                    "Media": "background-color:#ffe0b2",
                    "Baja":  "background-color:#c8e6c9",
                }.get(val, "")

            st.dataframe(
                df_show.style
                       .map(color_priority, subset=["Prioridad"])
                       .format({c: "{:.3f}" for c in df_show.select_dtypes("float").columns}),
                use_container_width=True,
                height=300,
            )
    else:
        st.warning(
            f"No se cargaron datos de vulnerabilidad. "
            f"Verifica que `vulnerability_ranking.csv` e `indices_comunas.csv` "
            f"existen en `{PROC_DIR}` y `{DATA_PROC}`."
        )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — MAPA PREDICTIVO
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown(
        f'<p class="section-title">Mapa Corepletico — Priorizacion por {UT_COL} · {CIUDAD}</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Predicciones XGBoost para {latest_year} · "
        f"Coloreado por IVC predicho · {UT_COL}: unidad territorial activa"
    )

    if gdf is not None and not df_latest.empty:
        try:
            from streamlit_folium import st_folium
            import folium

            name_col = next(
                (c for c in ["NOMBRE", "nombre", "NOMBRE_COM", "NOMB_COM",
                              UT_COL, UT_COL.upper(), "Comuna", "COMUNA",
                              "Localidad", "LOCALIDAD"]
                 if c in gdf.columns),
                None,
            )
            if name_col:
                gdf_m = gdf.merge(
                    df_latest[[UT_COL, "ivc_pred", "prioridad"]],
                    left_on=name_col, right_on=UT_COL, how="left",
                ).to_crs(epsg=4326)

                center = [cfg.ciudad.centro.lat, cfg.ciudad.centro.lon] if config_ok else [6.2442, -75.5812]
                m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

                chor = folium.Choropleth(
                    geo_data=gdf_m.__geo_interface__,
                    data=df_latest,
                    columns=[UT_COL, "ivc_pred"],
                    key_on=f"feature.properties.{name_col}",
                    fill_color="RdYlGn_r",
                    fill_opacity=0.75,
                    line_opacity=0.5,
                    legend_name=f"IVC Predicho por {UT_COL} (XGBoost)",
                    nan_fill_color="#cccccc",
                ).add_to(m)

                chor.geojson.add_child(folium.features.GeoJsonTooltip(
                    fields=[name_col, "ivc_pred", "prioridad"],
                    aliases=[f"{UT_COL}:", "IVC Predicho:", "Prioridad:"],
                    localize=True,
                    sticky=True,
                ))
                folium.LayerControl().add_to(m)
                st_folium(m, width="100%", height=560)
            else:
                st.info(
                    f"Archivo geoespacial encontrado pero no se identifico la columna "
                    f"de {UT}. Columnas: {gdf.columns.tolist()}"
                )
                gdf = None
        except Exception as e:
            st.error(f"Error al renderizar el mapa Folium: {e}")
            gdf = None

    if gdf is None and not df_latest.empty and "ivc_pred" in df_latest.columns:
        st.info(
            f"No se encontro archivo geoespacial de {UT_PLU.lower()} en `{GEO_DIR}`. "
            f"Coloca el archivo como `{UT}s.geojson` para activar el mapa Folium. "
            f"Se muestra un mapa de burbujas con centroides del config."
        )

        # Centroides desde config.yaml (dinámicos)
        centroids: dict[str, tuple[float, float]] = {
            t.nombre: (t.lat, t.lon) for t in cfg.territorios
        } if config_ok else {}

        df_map = df_latest.copy()
        df_map["lat"] = df_map[UT_COL].map(lambda c: centroids.get(c, (np.nan, np.nan))[0])
        df_map["lon"] = df_map[UT_COL].map(lambda c: centroids.get(c, (np.nan, np.nan))[1])
        df_map = df_map.dropna(subset=["lat", "lon"])

        if df_map.empty:
            st.warning(
                f"No se encontraron coordenadas para las {UT_PLU.lower()} del dataset. "
                f"Verifica que los nombres en `indices_comunas.csv` coinciden con "
                f"los nombres en `config.yaml > territorios`."
            )
        else:
            center = [cfg.ciudad.centro.lat, cfg.ciudad.centro.lon] if config_ok else [6.2442, -75.5812]
            fig_map = px.scatter_mapbox(
                df_map,
                lat="lat", lon="lon",
                color="ivc_pred",
                size="ivc_pred",
                hover_name=UT_COL,
                hover_data={"ivc_pred": ":.4f", "prioridad": True, "lat": False, "lon": False},
                color_continuous_scale="RdYlGn_r",
                size_max=35,
                zoom=11,
                center={"lat": center[0], "lon": center[1]},
                mapbox_style="carto-positron",
                labels={"ivc_pred": "IVC Predicho", UT_COL: UT_COL},
                title=f"Priorizacion por {UT_COL} — {CIUDAD} (centroides del config)",
            )
            fig_map.update_layout(height=560, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_map, use_container_width=True)

            st.dataframe(
                df_map[[UT_COL, "ivc_pred", "prioridad"]]
                .sort_values("ivc_pred", ascending=False)
                .rename(columns={"ivc_pred": "IVC Predicho", "prioridad": "Prioridad"})
                .reset_index(drop=True),
                use_container_width=True,
                height=260,
            )

    elif not run_pipeline and df_latest.empty:
        st.warning("No hay datos de prediccion disponibles. Ejecuta el Pipeline Automatico.")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — INTERPRETABILIDAD
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown(
        '<p class="section-title">Importancia de Variables — Modelo XGBoost</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Variables sociales y climaticas que mas influyen en el IVC predicho "
        f"para las {UT_PLU.lower()} de {CIUDAD}."
    )

    col_fi, col_shap = st.columns(2)

    with col_fi:
        st.markdown("**XGBoost Gain Importance**")
        st.caption("Contribucion de cada variable al criterio de division de los arboles.")
        if model is not None:
            try:
                fi = model.feature_importances_
                fi_df = pd.DataFrame({"feature": FEATURE_COLS, "importance": fi})
                fi_df = fi_df.sort_values("importance", ascending=True)
                fi_df["feature_label"] = fi_df["feature"].map(FEATURE_LABELS)
                max_fi = fi_df["importance"].max()
                fi_df["color"] = fi_df["importance"].apply(
                    lambda v: "#d32f2f" if v == max_fi else "#3949ab"
                )
                fig_fi = go.Figure(go.Bar(
                    y=fi_df["feature_label"],
                    x=fi_df["importance"],
                    orientation="h",
                    marker_color=fi_df["color"].tolist(),
                    text=[f"{v:.4f}" for v in fi_df["importance"]],
                    textposition="outside",
                    hovertemplate=f"<b>%{{y}}</b><br>Importancia: %{{x:.5f}}<extra></extra>",
                ))
                fig_fi.add_vline(
                    x=fi_df["importance"].mean(),
                    line_dash="dash", line_color="gray",
                    annotation_text="Media", annotation_position="top right",
                )
                fig_fi.update_layout(
                    xaxis_title="Importancia (Gain)",
                    yaxis_title="",
                    title=f"Gain Importance — {CIUDAD}",
                    height=380,
                    margin=dict(l=10, r=70, t=40, b=40),
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(showgrid=True, gridcolor="#e0e0e0"),
                )
                st.plotly_chart(fig_fi, use_container_width=True)
                best_feat = fi_df.iloc[-1]["feature_label"]
                st.success(f"**Variable mas influyente:** {best_feat} (Gain: {max_fi:.4f})")
            except Exception as e:
                st.error(f"No se pudo calcular Feature Importance: {e}")
        else:
            st.warning("Modelo no disponible. Ejecuta el Pipeline Automatico.")

    with col_shap:
        st.markdown("**SHAP — Importancia Media |SHAP|**")
        st.caption("Impacto promedio de cada variable sobre la prediccion final.")
        if df_shap is not None:
            try:
                shap_feats = [c for c in FEATURE_COLS if c in df_shap.columns]
                mean_shap  = df_shap[shap_feats].abs().mean().reset_index()
                mean_shap.columns = ["feature", "mean_abs_shap"]
                mean_shap = mean_shap.sort_values("mean_abs_shap", ascending=True)
                mean_shap["feature_label"] = mean_shap["feature"].map(FEATURE_LABELS)
                max_s = mean_shap["mean_abs_shap"].max()
                mean_shap["color"] = mean_shap["mean_abs_shap"].apply(
                    lambda v: "#2e7d32" if v == max_s else "#1b5e20" if v > max_s * 0.5 else "#81c784"
                )
                fig_shap = go.Figure(go.Bar(
                    y=mean_shap["feature_label"],
                    x=mean_shap["mean_abs_shap"],
                    orientation="h",
                    marker_color=mean_shap["color"].tolist(),
                    text=[f"{v:.4f}" for v in mean_shap["mean_abs_shap"]],
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>|SHAP| medio: %{x:.5f}<extra></extra>",
                ))
                fig_shap.update_layout(
                    xaxis_title="Mean |SHAP value|",
                    yaxis_title="",
                    title=f"SHAP Importance — {CIUDAD}",
                    height=380,
                    margin=dict(l=10, r=70, t=40, b=40),
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(showgrid=True, gridcolor="#e0e0e0"),
                )
                st.plotly_chart(fig_shap, use_container_width=True)
                best_shap = mean_shap.iloc[-1]["feature_label"]
                st.success(
                    f"**Variable con mayor impacto SHAP:** {best_shap} "
                    f"(|SHAP| medio: {max_s:.4f})"
                )
            except Exception as e:
                st.error(f"Error al graficar SHAP: {e}")
        else:
            st.info(
                "shap_values.csv no encontrado. "
                "Ejecuta el Pipeline Automatico para generarlo."
            )

    st.markdown("---")
    st.markdown(
        '<p class="section-title">Metricas de Validacion Cruzada Temporal (Walk-Forward)</p>',
        unsafe_allow_html=True,
    )
    if metrics:
        cv_df = pd.DataFrame(metrics["cv_folds"])[
            ["fold", "train_years", "test_years", "RMSE", "MAE", "R2"]
        ]
        cv_df.columns = ["Fold", "Años Entrenamiento", "Años Prueba", "RMSE", "MAE", "R²"]

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("R² CV Promedio",      f"{metrics['mean_R2']:.4f}")
        col_m2.metric("RMSE CV Promedio",    f"{metrics['mean_RMSE']:.5f}")
        col_m3.metric("R² Datos Completos",  f"{metrics['full_data_R2']:.4f}")

        fig_cv = go.Figure()
        fig_cv.add_trace(go.Scatter(
            x=cv_df["Fold"], y=cv_df["R²"],
            mode="lines+markers+text",
            text=[f"{v:.3f}" for v in cv_df["R²"]],
            textposition="top center",
            name="R² por Fold",
            line=dict(color="#3949ab", width=2),
            marker=dict(size=9),
        ))
        fig_cv.add_hline(
            y=metrics["mean_R2"], line_dash="dot", line_color="#d32f2f",
            annotation_text=f"R² promedio = {metrics['mean_R2']:.4f}",
            annotation_position="bottom right",
        )
        fig_cv.update_layout(
            xaxis_title="Fold Temporal",
            yaxis_title="R²",
            title=f"Walk-Forward CV — {CIUDAD}",
            height=280,
            margin=dict(l=0, r=20, t=40, b=40),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(range=[0.85, 1.0], showgrid=True, gridcolor="#e0e0e0"),
            xaxis=dict(tickmode="linear", dtick=1),
        )
        st.plotly_chart(fig_cv, use_container_width=True)

        st.dataframe(
            cv_df.style.format({"RMSE": "{:.5f}", "MAE": "{:.5f}", "R²": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("Hiperparametros del modelo final"):
            st.json(metrics.get("best_params", {}))
    else:
        st.warning(
            "No se cargaron metricas. Ejecuta el Pipeline Automatico para entrenar el modelo."
        )
