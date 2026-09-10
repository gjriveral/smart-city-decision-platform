"""
generar_datos_bogota.py
-----------------------
Genera datos SINTÉTICOS (mock) de Bogotá para pruebas de portabilidad /
escalabilidad del pipeline doctoral (ABM + XGBoost + Streamlit).

Unidades territoriales:
  · 20 localidades oficiales
  · Barrios representativos asociados a cada localidad
    (mock de nombres conocidos; NO es el inventario oficial completo)

IMPORTANTE: estos archivos NO son datos oficiales del DANE ni de la Alcaldía.
Sirven únicamente para verificar que la plataforma puede cargar una segunda
ciudad con localidades + barrios sin romper el esquema de columnas.

Uso (desde la raíz del repositorio):
    py -X utf8 generar_datos_bogota.py

Salida:
    Datos_Bogotá_Beta/
        indices_comunas.csv
        shap_values.csv
        bogota_dummy.geojson
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "Datos_Bogotá_Beta"
SEED = 42
YEARS = list(range(2010, 2025))  # 15 años: 2010–2024
YEAR_SHAP = 2024

# Localidad → barrios representativos (sintéticos / conocidos; no exhaustivo)
BARRIOS_POR_LOCALIDAD: dict[str, list[str]] = {
    "Usaquén": [
        "Usaquén Centro", "Cedritos", "Santa Bárbara", "Toberín", "Verbenal",
    ],
    "Chapinero": [
        "Chapinero Central", "El Chicó", "Rosales", "La Cabrera", "Pardo Rubio",
    ],
    "Santa Fe": [
        "Las Cruces", "La Perseverancia", "San Diego", "Egipto",
    ],
    "San Cristóbal": [
        "San Blas", "La Gloria", "Sosiego", "Quinta Mentira",
    ],
    "Usme": [
        "Usme Centro", "Alfonso López", "Yomasa", "La Fiscala",
    ],
    "Tunjuelito": [
        "Venecia", "Tunjuelito Centro", "San Vicente Ferrer",
    ],
    "Bosa": [
        "Bosa Centro", "El Recreo", "La Estación", "Apogeo",
    ],
    "Kennedy": [
        "Kennedy Central", "Castilla", "Timiza", "Patio Bonito", "Carvajal",
    ],
    "Fontibón": [
        "Fontibón Centro", "Modelia", "Zona Franca", "Capellanía",
    ],
    "Engativá": [
        "Engativá Centro", "Bonanza", "Las Ferias", "Normandía", "El Minuto de Dios",
    ],
    "Suba": [
        "Suba Centro", "Niza", "Britalia", "La Gaitana", "Tibabuyes",
    ],
    "Barrios Unidos": [
        "Doce de Octubre", "Los Andes", "Entrerríos", "La Castellana",
    ],
    "Teusaquillo": [
        "Teusaquillo Centro", "Galarias", "La Soledad", "Palermo",
    ],
    "Los Mártires": [
        "Santa Isabel", "La Sabana", "Eduardo Santos",
    ],
    "Antonio Nariño": [
        "Restrepo", "Ciudad Jardín", "Villa Mayor",
    ],
    "Puente Aranda": [
        "Puente Aranda Centro", "Mundo Aventura", "La Trinidad", "Santa Rita",
    ],
    "La Candelaria": [
        "La Catedral", "Centro Histórico", "Egipto Alto",
    ],
    "Rafael Uribe Uribe": [
        "Quiroga", "Diana Turbay", "Marco Fidel Suárez", "Inglés",
    ],
    "Ciudad Bolívar": [
        "Lucero", "Meissen", "Ismael Perdomo", "Arborizadora Alta", "Quiba",
    ],
    "Sumapaz": [
        "Nazareth", "San Juan", "Betania",
    ],
}

LOCALIDADES = list(BARRIOS_POR_LOCALIDAD.keys())

IDX_COLS = [
    "idx_desempleo",
    "idx_habitat",
    "idx_educacion",
    "idx_riesgo_clima",
    "idx_pobreza",
    "idx_tejido_social",
]

SHAP_COLS = [
    "shap_desempleo",
    "shap_habitat",
    "shap_educacion",
    "shap_riesgo_clima",
    "shap_pobreza",
    "shap_tejido_social",
]


def unidades_territoriales() -> list[dict]:
    """
    Lista plana de UTs: primero las 20 localidades, luego cada barrio
    con referencia a su localidad padre.
    """
    units: list[dict] = []
    for loc in LOCALIDADES:
        units.append(
            {
                "Unidad_Territorial": loc,
                "tipo": "localidad",
                "localidad_padre": loc,
            }
        )
        for barrio in BARRIOS_POR_LOCALIDAD[loc]:
            # Prefijo evita colisión de nombres entre localidad y barrio homónimo
            nombre = barrio if barrio != loc else f"Barrio {barrio}"
            units.append(
                {
                    "Unidad_Territorial": nombre,
                    "tipo": "barrio",
                    "localidad_padre": loc,
                }
            )
    return units


def generar_indices(rng: np.random.Generator, units: list[dict]) -> pd.DataFrame:
    """Panel histórico sintético: localidades + barrios × 15 años."""
    rows = []
    base = {
        u["Unidad_Territorial"]: rng.uniform(0.20, 0.80, size=len(IDX_COLS))
        for u in units
    }
    # Barrios: correlacionar levemente con la base de su localidad padre
    for u in units:
        if u["tipo"] != "barrio":
            continue
        padre = base[u["localidad_padre"]]
        ruido = rng.normal(0, 0.06, size=len(IDX_COLS))
        base[u["Unidad_Territorial"]] = np.clip(padre + ruido, 0.0, 1.0)

    for year in YEARS:
        drift = 0.01 * (year - YEARS[0]) / max(len(YEARS) - 1, 1)
        for u in units:
            name = u["Unidad_Territorial"]
            vals = np.clip(
                base[name] + drift + rng.normal(0, 0.035, size=len(IDX_COLS)),
                0.0,
                1.0,
            )
            vals[IDX_COLS.index("idx_riesgo_clima")] = float(
                np.clip(0.5 + rng.normal(0, 0.02), 0.0, 1.0)
            )
            rec = {
                "Año": year,
                "Unidad_Territorial": name,
                "tipo": u["tipo"],
                "localidad_padre": u["localidad_padre"],
            }
            for c, v in zip(IDX_COLS, vals):
                rec[c] = float(v)
            rec["IVC"] = float(np.mean(vals))
            rows.append(rec)

    df = pd.DataFrame(rows)
    col_order = [
        "Año",
        "Unidad_Territorial",
        "tipo",
        "localidad_padre",
        *IDX_COLS,
        "IVC",
    ]
    return df[col_order]


def generar_shap(rng: np.random.Generator, units: list[dict]) -> pd.DataFrame:
    """SHAP sintético 2024 por localidad y por barrio."""
    rows = []
    for u in units:
        raw = rng.normal(0.0, 0.02, size=len(SHAP_COLS))
        raw = raw - raw.mean()
        rec = {
            "Unidad_Territorial": u["Unidad_Territorial"],
            "tipo": u["tipo"],
            "localidad_padre": u["localidad_padre"],
        }
        for c, v in zip(SHAP_COLS, raw):
            rec[c] = float(v)
        rows.append(rec)
    return pd.DataFrame(rows)[
        ["Unidad_Territorial", "tipo", "localidad_padre", *SHAP_COLS]
    ]


def generar_geojson(units: list[dict]) -> dict:
    """
    GeoJSON mock:
      · Localidades = celdas grandes en grilla 5×4
      · Barrios = subcuadrados dentro de la celda de su localidad
    """
    lon0, lat0 = -74.22, 4.45
    step = 0.06
    ncols = 5

    # Centroide / SW corner por localidad
    loc_origin: dict[str, tuple[float, float]] = {}
    for i, loc in enumerate(LOCALIDADES):
        row, col = divmod(i, ncols)
        loc_origin[loc] = (lon0 + col * step, lat0 + row * step)

    features = []
    barrio_counters: dict[str, int] = {loc: 0 for loc in LOCALIDADES}

    for u in units:
        loc = u["localidad_padre"]
        lon_sw, lat_sw = loc_origin[loc]

        if u["tipo"] == "localidad":
            # Cuadrado grande de la localidad
            ring = [
                [lon_sw, lat_sw],
                [lon_sw + step * 0.92, lat_sw],
                [lon_sw + step * 0.92, lat_sw + step * 0.92],
                [lon_sw, lat_sw + step * 0.92],
                [lon_sw, lat_sw],
            ]
        else:
            # Subcuadrícula 2×N dentro de la localidad
            k = barrio_counters[loc]
            barrio_counters[loc] = k + 1
            sub = step * 0.40
            br, bc = divmod(k, 2)
            lon = lon_sw + 0.04 + bc * (sub + 0.02)
            lat = lat_sw + 0.04 + br * (sub + 0.02)
            ring = [
                [lon, lat],
                [lon + sub, lat],
                [lon + sub, lat + sub],
                [lon, lat + sub],
                [lon, lat],
            ]

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "nombre": u["Unidad_Territorial"],
                    "Unidad_Territorial": u["Unidad_Territorial"],
                    "tipo": u["tipo"],
                    "localidad_padre": u["localidad_padre"],
                    "nota": "geometría sintética — no es cartografía oficial",
                },
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )

    return {
        "type": "FeatureCollection",
        "name": "bogota_localidades_y_barrios_dummy",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": features,
    }


def _centroid(ring: list[list[float]]) -> tuple[float, float]:
    xs = [p[0] for p in ring[:-1]]
    ys = [p[1] for p in ring[:-1]]
    return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))


def generar_abm_sintetico(
    rng: np.random.Generator,
    df_idx: pd.DataFrame,
    units: list[dict],
) -> pd.DataFrame:
    """
    Trayectorias ABM sintéticas (10 años × 3 escenarios) para que el
    dashboard de Bogotá no reutilice la simulación de Medellín.
    """
    latest = int(df_idx["Año"].max())
    base = (
        df_idx[df_idx["Año"] == latest]
        .set_index("Unidad_Territorial")["IVC"]
        .to_dict()
    )
    scenarios = {
        "focalizado": 0.035,
        "adaptativo": 0.022,
        "distribuido": 0.012,
    }
    rows = []
    for sc, decay in scenarios.items():
        for year in range(1, 11):
            for u in units:
                name = u["Unidad_Territorial"]
                iv0 = float(base.get(name, 0.5))
                noise = float(rng.normal(0, 0.003))
                # Focalizado reduce más las UTs con IVC alto
                boost = decay * (0.5 + iv0) if sc == "focalizado" else decay
                iv = float(np.clip(iv0 - boost * year + noise, 0.05, 0.95))
                wb = float(np.clip(0.45 + (iv0 - iv) * 1.2, 0.05, 0.95))
                rows.append(
                    {
                        "scenario": sc,
                        "year": year,
                        "comuna": name,  # el ABM/dashboard usa 'comuna' como clave genérica
                        "ivc_initial": iv0,
                        "investment_allocated": float(1_000_000 / max(len(units), 1)),
                        "wellbeing": wb,
                        "ivc_current": iv,
                        "infrastructure_capacity": float(
                            np.clip(0.4 + year * 0.02 + rng.normal(0, 0.01), 0.1, 1.0)
                        ),
                        "cumulative_investment": float(
                            year * 1_000_000 / max(len(units), 1)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def exportar_para_plataforma(
    df_idx: pd.DataFrame,
    df_shap: pd.DataFrame,
    geo: dict,
    units: list[dict],
    rng: np.random.Generator,
) -> None:
    """
    Materializa la estructura de rutas que lee Streamlit vía config_bogota.yaml
    y escribe el config apuntando a Datos_Bogotá_Beta.
    """
    import shutil

    processed = OUT_DIR / "processed"
    models = OUT_DIR / "models"
    geo_dir = OUT_DIR / "geo"
    evidencia = OUT_DIR / "evidencia"
    logs = OUT_DIR / "logs"
    for d in (processed, models, geo_dir, evidencia, logs):
        d.mkdir(parents=True, exist_ok=True)

    # ── Índices en esquema del pipeline (columna = Localidad) ───────────────
    pipe_idx = df_idx.rename(
        columns={"Unidad_Territorial": "Localidad", "IVC": "ivc"}
    )
    pipe_idx.to_csv(processed / "indices_comunas.csv", index=False, encoding="utf-8-sig")
    print(f"OK  processed/indices_comunas.csv ({len(pipe_idx)} filas)")

    # ── Ranking medio histórico ─────────────────────────────────────────────
    ranking = (
        pipe_idx.groupby("Localidad", as_index=False)["ivc"]
        .mean()
        .rename(columns={"ivc": "mean_ivc"})
        .sort_values("mean_ivc", ascending=False)
    )
    ranking.to_csv(models / "vulnerability_ranking.csv", index=False, encoding="utf-8-sig")
    print(f"OK  models/vulnerability_ranking.csv ({len(ranking)} UTs)")

    # ── SHAP en esquema del dashboard (idx_* + Localidad + Año) ─────────────
    shap_pipe = df_shap.rename(
        columns={
            "Unidad_Territorial": "Localidad",
            "shap_desempleo": "idx_desempleo",
            "shap_habitat": "idx_habitat",
            "shap_educacion": "idx_educacion",
            "shap_riesgo_clima": "idx_riesgo_clima",
            "shap_pobreza": "idx_pobreza",
            "shap_tejido_social": "idx_tejido_social",
        }
    )
    shap_pipe["año_normalizado"] = 1.0
    shap_pipe["Año"] = YEAR_SHAP
    shap_pipe.to_csv(models / "shap_values.csv", index=False, encoding="utf-8-sig")

    # Reutilizar métricas/modelo/estabilidad de Medellín (mismo esquema de features)
    # para poder predecir IVC en el dashboard sin reentrenar.
    src_models = ROOT / "models"
    for fname in (
        "vulnerability_model.pkl",
        "cv_metrics.json",
        "shap_importance_cv_stability.csv",
        "feature_importance.png",
        "shap_importance.png",
        "shap_importance_cv_stability.png",
    ):
        src = src_models / fname
        if src.exists():
            shutil.copy2(src, models / fname)
            print(f"OK  models/{fname} (copiado desde Medellín — demo portabilidad)")

    # ── GeoJSON canónico ────────────────────────────────────────────────────
    # Si ya existe cartografía oficial IDECA, no la sobrescribe con el dummy.
    oficial = OUT_DIR / "bogota_localidades.geojson"
    geo_path = geo_dir / "localidades.geojson"
    if oficial.exists() and oficial.stat().st_size > 100_000:
        print(
            "OK  geo/localidades.geojson — se conserva cartografía oficial "
            f"({oficial.name}). Ejecuta descargar_geojson_bogota.py para actualizarla."
        )
        # Asegura que la plataforma tenga la copia normalizada
        try:
            from descargar_geojson_bogota import instalar_en_plataforma
            raw = json.loads(oficial.read_text(encoding="utf-8"))
            instalar_en_plataforma(raw)
            print(f"OK  Reinstalado {geo_path.name} desde oficial")
        except Exception as exc:
            print(f"AVISO: no se pudo reinstalar oficial → {exc}")
    else:
        for ft in geo["features"]:
            ft["properties"]["Localidad"] = ft["properties"]["nombre"]
        geo_path.write_text(json.dumps(geo, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.copy2(geo_path, OUT_DIR / "bogota_dummy.geojson")
        print(f"OK  geo/localidades.geojson ({len(geo['features'])} features DUMMY)")

    # ── ABM sintético ───────────────────────────────────────────────────────
    df_abm = generar_abm_sintetico(rng, df_idx, units)
    abm_path = evidencia / "resultados_simulacion_escenarios.csv"
    df_abm.to_csv(abm_path, index=False, encoding="utf-8-sig")
    print(f"OK  evidencia/resultados_simulacion_escenarios.csv ({len(df_abm)} filas)")

    # ── config_bogota.yaml ──────────────────────────────────────────────────
    centros = {}
    for ft in geo["features"]:
        name = ft["properties"]["nombre"]
        lon, lat = _centroid(ft["geometry"]["coordinates"][0])
        centros[name] = (lat, lon)

    terr_lines = []
    for i, u in enumerate(units, start=1):
        name = u["Unidad_Territorial"]
        lat, lon = centros.get(name, (4.7110, -74.0721))
        # YAML: entrecomillar nombres con espacios/tildes
        terr_lines.append(
            f'  - {{id: {i}, nombre: "{name}", lat: {lat:.6f}, lon: {lon:.6f}}}'
        )

    cfg = f"""# config_bogota.yaml — Bogotá beta (localidades + barrios sintéticos)
# Datos de portabilidad (índices sintéticos; cartografía según fuente).
ciudad:
  nombre: "Bogotá"
  departamento: "Cundinamarca"
  codigo_dane: "11001"
  unidad_territorial: "localidad"
  crs: "EPSG:4326"
  centro:
    lat: 4.7110
    lon: -74.0721

rutas:
  datos_climaticos: "Datos_Bogotá_Beta"
  datos_sociales: "Datos_Bogotá_Beta"
  procesados: "Datos_Bogotá_Beta/processed"
  modelos: "Datos_Bogotá_Beta/models"
  logs: "Datos_Bogotá_Beta/logs"
  geo: "Datos_Bogotá_Beta/geo"

territorios:
{chr(10).join(terr_lines)}

simulacion:
  horizonte_anios: 10
  presupuesto_anual: 1000000
  n_replicas_montecarlo: 1
  escenarios:
    - id: focalizado
      nombre: "Focalizado — Optimización Algorítmica"
      peso: 0.80
    - id: distribuido
      nombre: "Distribuido — Igualitario"
      peso: 0.0
    - id: adaptativo
      nombre: "Adaptativo — Híbrido"
      peso: 0.0

catalogo_estaciones: {{}}
"""
    cfg_path = ROOT / "config_bogota.yaml"
    cfg_path.write_text(cfg, encoding="utf-8")
    print(f"OK  config_bogota.yaml ({len(units)} territorios)")


def main() -> None:
    print("=== Datos sintéticos Bogotá — localidades + barrios ===")
    print("ADVERTENCIA: mock data — no usar para inferencia empírica.\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    units = unidades_territoriales()
    n_loc = sum(1 for u in units if u["tipo"] == "localidad")
    n_bar = sum(1 for u in units if u["tipo"] == "barrio")
    print(f"UTs: {len(units)}  ({n_loc} localidades + {n_bar} barrios)")

    df_idx = generar_indices(rng, units)
    path_idx = OUT_DIR / "indices_comunas.csv"
    df_idx.to_csv(path_idx, index=False, encoding="utf-8-sig")
    print(
        f"OK  {path_idx.name}: {len(df_idx)} filas "
        f"({df_idx['Unidad_Territorial'].nunique()} UTs × "
        f"{df_idx['Año'].nunique()} años)"
    )

    df_shap = generar_shap(rng, units)
    path_shap = OUT_DIR / "shap_values.csv"
    df_shap.to_csv(path_shap, index=False, encoding="utf-8-sig")
    print(f"OK  {path_shap.name}: {len(df_shap)} filas (año {YEAR_SHAP})")

    geo = generar_geojson(units)
    path_geo = OUT_DIR / "bogota_dummy.geojson"
    path_geo.write_text(json.dumps(geo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK  {path_geo.name}: {len(geo['features'])} polígonos mock")

    print("\n--- Exportación para la plataforma Streamlit ---")
    exportar_para_plataforma(df_idx, df_shap, geo, units, rng)

    print(f"\nCarpeta lista: {OUT_DIR}")
    print("En el dashboard: Perfil de ciudad → Bogotá (demo)")


if __name__ == "__main__":
    main()
