"""
generar_datos_ciudades.py
-------------------------
Genera bloques de datos de portabilidad / escalabilidad para cinco ciudades
colombianas (Cali, Barranquilla, Bucaramanga, Pereira, Manizales), con la
estructura que consume el pipeline doctoral (índices → XGBoost/SHAP → ABM →
Streamlit/Folium).

Por cada ciudad crea:

    Datos_<Ciudad>_Beta/
        indices_comunas.csv          # panel histórico 2010–2024 (Unidad_Territorial)
        shap_values.csv              # SHAP sintético 2024
        <ciudad>_oficial.geojson     # comunas/localidades (+ barrios si la API los da)
        processed/indices_comunas.csv
        models/shap_values.csv
        models/vulnerability_ranking.csv
        geo/localidades.geojson
        evidencia/resultados_simulacion_escenarios.csv
        logs/

Lógica de portabilidad
----------------------
El esquema de columnas es el mismo que Medellín/Bogotá: índices `idx_*`, IVC
compuesto, SHAP por dimensión y polígonos con propiedad de unión
`Unidad_Territorial`. Cambiar de ciudad = cambiar carpeta + nombres de UT;
el resto del pipeline no se reescribe.

Cartografía
-----------
1. Intenta descargar GeoJSON oficial (WFS IDESC / ArcGIS REST municipal).
2. Si falla (timeout, SSL, capa inexistente), genera un mock en grilla y
   imprime el enlace del portal de datos abiertos para descarga manual.

Uso (raíz del repositorio)::

    py -X utf8 generar_datos_ciudades.py
    py -X utf8 generar_datos_ciudades.py --solo cali,pereira
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import requests
import urllib3

# SSL auto-firmado en algunos geoportales municipales (p. ej. Manizales).
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------------------------------------------------------------------
# Constantes globales
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SEED = 42
YEARS = list(range(2010, 2025))  # 15 años
YEAR_SHAP = 2024
HEADERS = {
    "User-Agent": "Prioriza-TDG-UdeA/1.0 (portabilidad; tesis doctoral)",
    "Accept": "application/json, application/geo+json, */*",
}
TIMEOUT_S = 180

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


# ---------------------------------------------------------------------------
# Catálogo de ciudades (nombres oficiales de UT + fuentes geo)
# ---------------------------------------------------------------------------
@dataclass
class CiudadSpec:
    """Metadatos de una ciudad para portabilidad del pipeline."""

    clave: str  # slug: cali, barranquilla, ...
    nombre: str
    departamento: str
    codigo_dane: str
    unidad_label: str  # "comuna" | "localidad"
    folder: str  # Datos_Cali_Beta
    unidades: list[str]  # nombres oficiales canónicos
    centro: tuple[float, float]  # (lat, lon)
    portal_manual: str  # URL si falla la descarga automática
    geo_fetchers: list[Callable[["CiudadSpec"], dict]] = field(default_factory=list)


def _get_json(url: str, *, verify: bool = True, params: dict | None = None) -> dict:
    resp = requests.get(
        url, params=params, headers=HEADERS, timeout=TIMEOUT_S, verify=verify
    )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} → {resp.url}")
    try:
        payload = resp.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Respuesta no-JSON desde {resp.url}") from exc
    return payload


def _es_feature_collection(payload: object, min_feats: int = 1) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "no es dict"
    if payload.get("type") != "FeatureCollection":
        return False, f"type={payload.get('type')!r}"
    feats = payload.get("features")
    if not isinstance(feats, list) or len(feats) < min_feats:
        return False, f"features insuficientes ({0 if not isinstance(feats, list) else len(feats)})"
    return True, f"{len(feats)} features"


def _pick_name(props: dict, candidates: tuple[str, ...]) -> Optional[str]:
    for key in candidates:
        if key in props and props[key] not in (None, ""):
            return str(props[key]).strip()
    # fallback: primera clave string no técnica
    skip = {"OBJECTID", "FID", "GlobalID", "Shape__Area", "Shape__Length", "id"}
    for k, v in props.items():
        if k in skip or v is None:
            continue
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _normalize_feature_names(
    geo: dict,
    name_keys: tuple[str, ...],
    *,
    tipo_default: str = "comuna",
) -> dict:
    """Asegura properties.Unidad_Territorial / nombre en cada feature."""
    out_feats = []
    for ft in geo.get("features", []):
        props = dict(ft.get("properties") or {})
        name = _pick_name(props, name_keys)
        if name is None:
            continue
        props["Unidad_Territorial"] = name
        props["nombre"] = name
        props.setdefault("tipo", tipo_default)
        out_feats.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": ft.get("geometry"),
            }
        )
    return {
        "type": "FeatureCollection",
        "name": geo.get("name", "oficial"),
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": out_feats,
    }


def _merge_feature_collections(*collections: dict) -> dict:
    feats: list[dict] = []
    for col in collections:
        if not col:
            continue
        feats.extend(col.get("features") or [])
    return {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": feats,
    }


# ── Fetchers por ciudad ─────────────────────────────────────────────────────


def fetch_cali_oficial(_city: CiudadSpec) -> dict:
    """IDESC GeoServer WFS — comunas + barrios/sectores (EPSG:4326)."""
    base = "https://ws-idesc.cali.gov.co/geoserver/wfs"
    comunas = _get_json(
        base,
        params={
            "service": "WFS",
            "version": "1.1.0",
            "request": "GetFeature",
            "typeName": "dapm:pdt_dpa_comunas",
            "outputFormat": "application/json",
            "srsName": "EPSG:4326",
        },
    )
    ok, msg = _es_feature_collection(comunas, min_feats=20)
    if not ok:
        raise RuntimeError(f"Comunas Cali inválidas: {msg}")

    barrios: dict = {}
    try:
        barrios = _get_json(
            base,
            params={
                "service": "WFS",
                "version": "1.1.0",
                "request": "GetFeature",
                "typeName": "dapm:pdt_dpa_barrios_sectores",
                "outputFormat": "application/json",
                "srsName": "EPSG:4326",
            },
        )
        ok_b, _ = _es_feature_collection(barrios, min_feats=1)
        if not ok_b:
            barrios = {}
    except Exception as exc:  # noqa: BLE001 — contingencia documentada
        print(f"  AVISO barrios Cali: {exc}")
        barrios = {}

    g_com = _normalize_feature_names(
        comunas, ("comnombre", "COMNOMBRE", "nombre"), tipo_default="comuna"
    )
    for ft in g_com["features"]:
        ft["properties"]["tipo"] = "comuna"

    parts = [g_com]
    if barrios:
        # Campos típicos IDESC barrios
        g_bar = _normalize_feature_names(
            barrios,
            ("barrio", "nombre", "barnombre", "NOMBRE", "sector"),
            tipo_default="barrio",
        )
        for ft in g_bar["features"]:
            ft["properties"]["tipo"] = "barrio"
        parts.append(g_bar)
        print(f"  Cali: {len(g_com['features'])} comunas + {len(g_bar['features'])} barrios")
    else:
        print(f"  Cali: {len(g_com['features'])} comunas (sin barrios)")

    merged = _merge_feature_collections(*parts)
    merged["name"] = "cali_oficial"
    return merged


def fetch_barranquilla_oficial(_city: CiudadSpec) -> dict:
    """ArcGIS Online — localidades (+ barrios si el FeatureServer responde)."""
    org = "https://services3.arcgis.com/oGYAc07w6wsvgUYr/ArcGIS/rest/services"
    loc = _get_json(
        f"{org}/Localidades/FeatureServer/0/query",
        params={
            "where": "1=1",
            "outFields": "*",
            "outSR": "4326",
            "f": "geojson",
        },
    )
    ok, msg = _es_feature_collection(loc, min_feats=5)
    if not ok:
        raise RuntimeError(f"Localidades Barranquilla inválidas: {msg}")

    g_loc = _normalize_feature_names(loc, ("NOMBRE", "Nombre", "nombre"), tipo_default="localidad")
    # Title-case canónico (METROPOLITANA → Metropolitana)
    for ft in g_loc["features"]:
        raw = ft["properties"]["Unidad_Territorial"]
        canon = _title_ut(raw.replace(" - ", "-").replace("–", "-"))
        # Alias oficiales
        aliases = {
            "Norte-Centro Historico": "Norte-Centro Histórico",
            "Norte Centro Historico": "Norte-Centro Histórico",
            "Suroriente": "Suroriente",
            "Suroccidente": "Suroccidente",
            "Metropolitana": "Metropolitana",
            "Riomar": "Riomar",
        }
        ft["properties"]["Unidad_Territorial"] = aliases.get(canon, canon)
        ft["properties"]["nombre"] = ft["properties"]["Unidad_Territorial"]
        ft["properties"]["tipo"] = "localidad"

    parts = [g_loc]
    try:
        bar = _get_json(
            f"{org}/BARRIOS_WFL1/FeatureServer/0/query",
            params={
                "where": "1=1",
                "outFields": "*",
                "outSR": "4326",
                "f": "geojson",
                "resultRecordCount": "2000",
            },
        )
        ok_b, _ = _es_feature_collection(bar, min_feats=1)
        if ok_b:
            g_bar = _normalize_feature_names(
                bar, ("NOMBRE", "Nombre", "BARRIO"), tipo_default="barrio"
            )
            for ft in g_bar["features"]:
                ft["properties"]["tipo"] = "barrio"
                ft["properties"]["Unidad_Territorial"] = _title_ut(
                    ft["properties"]["Unidad_Territorial"]
                )
                ft["properties"]["nombre"] = ft["properties"]["Unidad_Territorial"]
            parts.append(g_bar)
            print(
                f"  Barranquilla: {len(g_loc['features'])} localidades + "
                f"{len(g_bar['features'])} barrios"
            )
        else:
            print(f"  Barranquilla: {len(g_loc['features'])} localidades")
    except Exception as exc:  # noqa: BLE001
        print(f"  AVISO barrios Barranquilla: {exc}")
        print(f"  Barranquilla: {len(g_loc['features'])} localidades")

    merged = _merge_feature_collections(*parts)
    merged["name"] = "barranquilla_oficial"
    return merged


def fetch_bucaramanga_oficial(_city: CiudadSpec) -> dict:
    """ArcGIS municipal — LIMITES_POLITICOS/Comunas."""
    url = (
        "https://vmarcgis01.bucaramanga.gov.co/waserver/rest/services/"
        "LIMITES_POLITICOS/Comunas/MapServer/0/query"
    )
    payload = _get_json(
        url,
        params={
            "where": "1=1",
            "outFields": "*",
            "outSR": "4326",
            "f": "geojson",
            "resultRecordCount": "100",
        },
    )
    ok, msg = _es_feature_collection(payload, min_feats=10)
    if not ok:
        raise RuntimeError(f"Comunas Bucaramanga inválidas: {msg}")

    g = _normalize_feature_names(
        payload, ("NOMBRE_COM", "NOMBRE", "nombre"), tipo_default="comuna"
    )
    for ft in g["features"]:
        name = ft["properties"]["Unidad_Territorial"]
        # "Comuna Norte" → "Norte"
        if name.lower().startswith("comuna "):
            name = name[7:].strip()
        name = _title_ut(name)
        # Acentos / alias canónicos
        aliases = {
            "Garcia Rovira": "García Rovira",
            "Ciudadela Real De Minas": "La Ciudadela",
            "Tejar": "Lagos del Cacique",
        }
        name = aliases.get(name, name)
        ft["properties"]["Unidad_Territorial"] = name
        ft["properties"]["nombre"] = name
        ft["properties"]["tipo"] = "comuna"

    print(f"  Bucaramanga: {len(g['features'])} comunas (capa municipal)")
    g["name"] = "bucaramanga_oficial"
    return g


def fetch_manizales_oficial(_city: CiudadSpec) -> dict:
    """ArcGIS SIG Manizales — comunas + barrios (certificado a menudo auto-firmado)."""
    base = (
        "https://sig.manizales.gov.co:6443/arcgis/rest/services/"
        "TIC/TIC_Delimitacion_Barrios_Comunas/MapServer"
    )
    comunas = _get_json(
        f"{base}/0/query",
        verify=False,
        params={
            "where": "1=1",
            "outFields": "*",
            "outSR": "4326",
            "f": "geojson",
        },
    )
    ok, msg = _es_feature_collection(comunas, min_feats=10)
    if not ok:
        raise RuntimeError(f"Comunas Manizales inválidas: {msg}")

    g_com = _normalize_feature_names(
        comunas, ("NOMBRES_CO", "NOMBRE", "nombre"), tipo_default="comuna"
    )
    cleaned: list[dict] = []
    seen: set[str] = set()
    for ft in g_com["features"]:
        name = ft["properties"]["Unidad_Territorial"]
        if name.lower().startswith("comuna "):
            name = name[7:].strip()
        name = _title_ut(name)
        aliases = {
            "San Jose": "San José",
            "Estacion": "La Estación",
            "La Estacion": "La Estación",
            "Ecoturistico Cerro De Oro": "Ecoturístico Cerro de Oro",
            "La Macarena": "La Macarena",
        }
        # Duplicado cartográfico / no pertenece al listado oficial de 11
        if "nuevo horizonte" in name.lower():
            continue
        name = aliases.get(name, name)
        if name in seen:
            continue
        seen.add(name)
        ft["properties"]["Unidad_Territorial"] = name
        ft["properties"]["nombre"] = name
        ft["properties"]["tipo"] = "comuna"
        cleaned.append(ft)
    g_com["features"] = cleaned

    parts = [g_com]
    try:
        bar = _get_json(
            f"{base}/1/query",
            verify=False,
            params={
                "where": "1=1",
                "outFields": "*",
                "outSR": "4326",
                "f": "geojson",
                "resultRecordCount": "2000",
            },
        )
        ok_b, _ = _es_feature_collection(bar, min_feats=1)
        if ok_b:
            g_bar = _normalize_feature_names(
                bar, ("BARRIOS", "NOMBRE", "nombre"), tipo_default="barrio"
            )
            for ft in g_bar["features"]:
                ft["properties"]["tipo"] = "barrio"
                ft["properties"]["Unidad_Territorial"] = _title_ut(
                    ft["properties"]["Unidad_Territorial"]
                )
                ft["properties"]["nombre"] = ft["properties"]["Unidad_Territorial"]
            parts.append(g_bar)
            print(
                f"  Manizales: {len(g_com['features'])} comunas + "
                f"{len(g_bar['features'])} barrios"
            )
        else:
            print(f"  Manizales: {len(g_com['features'])} comunas")
    except Exception as exc:  # noqa: BLE001
        print(f"  AVISO barrios Manizales: {exc}")
        print(f"  Manizales: {len(g_com['features'])} comunas")

    merged = _merge_feature_collections(*parts)
    merged["name"] = "manizales_oficial"
    return merged


def fetch_pereira_oficial(_city: CiudadSpec) -> dict:
    """
    Cartografía oficial SIGPER / Portal Geográfico de Pereira.

    Fuente primaria: ítem público ArcGIS «Comunas del municipio de Pereira»
    (shapefile ZIP publicado por sigper_arcgis). Se reproyecta EPSG:3115 → 4326.
    """
    import io
    import tempfile
    import zipfile

    import geopandas as gpd

    item_id = "36474e8eba2f4d49a4701d31c782251c"
    url = f"https://www.arcgis.com/sharing/rest/content/items/{item_id}/data"
    print(f"  Descargando shapefile SIGPER Pereira…\n    {url}")
    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT_S)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} al descargar shapefile Pereira")
    if len(resp.content) < 1000:
        raise RuntimeError("Respuesta demasiado corta para ser un shapefile ZIP")

    with tempfile.TemporaryDirectory(prefix="pereira_comunas_") as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            zf.extractall(tmp_path)
        shps = list(tmp_path.rglob("*.shp"))
        if not shps:
            raise RuntimeError("El ZIP no contiene un .shp")
        gdf = gpd.read_file(shps[0])

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=3115)  # Magna-Sirgas / Colombia Bogotá zone
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    # Alias topónimos SIGPER (sin tildes / tipografía antigua) → canónico CSV
    aliases = {
        "Olimpica": "Olímpica",
        "Del Cafe": "Del Café",
        "El Jardin": "El Jardín",
        "San Joaquin": "San Joaquín",
        "San Nicolas": "San Nicolás",
        "Consota": "Consotá",
        "Perla Del Otun": "Perla del Otún",
        "Perla del Otun": "Perla del Otún",
        "El Rocio": "El Rocío",
        "Rio Otun": "Río Otún",
        "Río Otun": "Río Otún",
        "Aeropuerto": "Sector Aeropuerto",
        "Batallon": "Sector Batallón",
        "Batallón": "Sector Batallón",
    }

    features = []
    for _, row in gdf.iterrows():
        raw = str(row.get("nombre") or "").strip()
        if not raw:
            continue
        name = raw
        tipo = "comuna"
        if name.lower().startswith("comuna "):
            name = name[7:].strip()
        elif name.lower().startswith("sector "):
            name = name[7:].strip()
            tipo = "sector"
        name = _title_ut(name)
        name = aliases.get(name, name)
        # Sectores especiales: prefijo canónico
        if tipo == "sector" and not name.lower().startswith("sector "):
            name = f"Sector {name}"
            name = aliases.get(name, name)
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "Unidad_Territorial": name,
                    "nombre": name,
                    "tipo": tipo,
                    "id_comuna": int(row["id_comuna"])
                    if "id_comuna" in row and row["id_comuna"] == row["id_comuna"]
                    else None,
                    "codcom": str(row["codcom"])
                    if "codcom" in row and row["codcom"] == row["codcom"]
                    else None,
                    "fuente": "SIGPER / Portal Geográfico Pereira (ArcGIS item)",
                },
                "geometry": geom.__geo_interface__,
            }
        )

    if len(features) < 10:
        raise RuntimeError(f"Solo {len(features)} polígonos tras normalizar; se esperaban ≥19")

    n_com = sum(1 for f in features if f["properties"]["tipo"] == "comuna")
    n_sec = sum(1 for f in features if f["properties"]["tipo"] == "sector")
    print(f"  Pereira: {n_com} comunas + {n_sec} sectores (SIGPER shapefile → WGS84)")
    return {
        "type": "FeatureCollection",
        "name": "pereira_oficial",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": features,
    }


def _osm_relation_to_multipolygon(el: dict) -> Optional[dict]:
    """Extrae anillo(es) exterior(es) de una relation Overpass `out geom` (fallback)."""
    members = el.get("members") or []
    rings: list[list[list[float]]] = []
    for m in members:
        if m.get("type") != "way" or m.get("role") not in ("outer", "", None):
            continue
        geom = m.get("geometry") or []
        if len(geom) < 3:
            continue
        ring = [[pt["lon"], pt["lat"]] for pt in geom]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        rings.append(ring)
    if not rings:
        g = el.get("geometry") or []
        if len(g) >= 3:
            ring = [[pt["lon"], pt["lat"]] for pt in g]
            if ring[0] != ring[-1]:
                ring.append(ring[0])
            rings = [ring]
    if not rings:
        return None
    if len(rings) == 1:
        return {"type": "Polygon", "coordinates": [rings[0]]}
    return {"type": "MultiPolygon", "coordinates": [[r] for r in rings]}


def _title_ut(s: str) -> str:
    """Title-case conservando conectores frecuentes en toponimia colombiana."""
    small = {"de", "del", "la", "las", "los", "el", "y", "e"}
    parts = str(s).replace("_", " ").strip().split()
    out = []
    for i, w in enumerate(parts):
        lw = w.lower()
        if i > 0 and lw in small:
            out.append(lw)
        else:
            out.append(lw.capitalize() if lw.isascii() else w[:1].upper() + w[1:].lower())
    # Acentos frecuentes
    text = " ".join(out)
    fixes = {
        "Historico": "Histórico",
        "Estacion": "Estación",
        "Jose": "José",
        "Garcia": "García",
        "Ecoturistico": "Ecoturístico",
        "Consota": "Consotá",
        "Rio": "Río",
    }
    for a, b in fixes.items():
        text = text.replace(a, b)
    return text


# ---------------------------------------------------------------------------
# Registro de ciudades
# ---------------------------------------------------------------------------
def build_catalog() -> dict[str, CiudadSpec]:
    cali_uts = [f"Comuna {i:02d}" for i in range(1, 23)]

    barranquilla_uts = [
        "Riomar",
        "Norte-Centro Histórico",
        "Suroccidente",
        "Metropolitana",
        "Suroriente",
    ]

    # Denominación oficial (Acuerdo municipal / POT) — 17 comunas
    bucaramanga_uts = [
        "Norte",
        "Nororiental",
        "San Francisco",
        "Occidental",
        "García Rovira",
        "La Concordia",
        "La Ciudadela",
        "Suroccidente",
        "La Pedregosa",
        "Provenza",
        "Sur",
        "Cabecera del Llano",
        "Oriental",
        "Occidente",
        "Centro",
        "Lagos del Cacique",
        "Mutis",
    ]

    # Anexo Wikipedia / Alcaldía — 19 comunas urbanas
    pereira_uts = [
        "Universidad",
        "Olímpica",
        "San Joaquín",
        "Cuba",
        "Del Café",
        "El Jardín",
        "Boston",
        "Consotá",
        "El Oso",
        "Perla del Otún",
        "El Poblado",
        "El Rocío",
        "Ferrocarril",
        "Centro",
        "Río Otún",
        "Oriente",
        "Villa Santana",
        "San Nicolás",
        "Villavicencio",
        # Sectores urbanos del shapefile SIGPER (rellenan huecos del mapa)
        "Sector Aeropuerto",
        "Sector Batallón",
    ]

    manizales_uts = [
        "Atardeceres",
        "San José",
        "Cumanday",
        "La Estación",
        "Ciudadela del Norte",
        "Ecoturístico Cerro de Oro",
        "Tesorito",
        "Palogrande",
        "Universitaria",
        "La Fuente",
        "La Macarena",
    ]

    cities = [
        CiudadSpec(
            clave="cali",
            nombre="Cali",
            departamento="Valle del Cauca",
            codigo_dane="76001",
            unidad_label="comuna",
            folder="Datos_Cali_Beta",
            unidades=cali_uts,
            centro=(3.4516, -76.5320),
            portal_manual="https://idesc.cali.gov.co/  |  WFS: https://ws-idesc.cali.gov.co/geoserver/wfs  |  Datos.gov: https://www.datos.gov.co/dataset/Comunas-de-Santiago-de-Cali/dx2g-2mhm",
            geo_fetchers=[fetch_cali_oficial],
        ),
        CiudadSpec(
            clave="barranquilla",
            nombre="Barranquilla",
            departamento="Atlántico",
            codigo_dane="08001",
            unidad_label="localidad",
            folder="Datos_Barranquilla_Beta",
            unidades=barranquilla_uts,
            centro=(10.9685, -74.7813),
            portal_manual="https://datos.barranquilla.gov.co/  |  ArcGIS Localidades: https://services3.arcgis.com/oGYAc07w6wsvgUYr/ArcGIS/rest/services/Localidades/FeatureServer",
            geo_fetchers=[fetch_barranquilla_oficial],
        ),
        CiudadSpec(
            clave="bucaramanga",
            nombre="Bucaramanga",
            departamento="Santander",
            codigo_dane="68001",
            unidad_label="comuna",
            folder="Datos_Bucaramanga_Beta",
            unidades=bucaramanga_uts,
            centro=(7.1193, -73.1227),
            portal_manual="https://www.bucaramanga.gov.co/division-politico-urbana/  |  ArcGIS: https://vmarcgis01.bucaramanga.gov.co/waserver/rest/services/LIMITES_POLITICOS/Comunas/MapServer",
            geo_fetchers=[fetch_bucaramanga_oficial],
        ),
        CiudadSpec(
            clave="pereira",
            nombre="Pereira",
            departamento="Risaralda",
            codigo_dane="66001",
            unidad_label="comuna",
            folder="Datos_Pereira_Beta",
            unidades=pereira_uts,
            centro=(4.8143, -75.6946),
            portal_manual="https://hub.arcgis.com/datasets/36474e8eba2f4d49a4701d31c782251c  |  https://mapas-pereira.opendata.arcgis.com/",
            geo_fetchers=[fetch_pereira_oficial],
        ),
        CiudadSpec(
            clave="manizales",
            nombre="Manizales",
            departamento="Caldas",
            codigo_dane="17001",
            unidad_label="comuna",
            folder="Datos_Manizales_Beta",
            unidades=manizales_uts,
            centro=(5.0703, -75.5138),
            portal_manual="https://datos-abiertos-alcaldia-manizales-sigalcmzl.hub.arcgis.com/  |  ArcGIS: https://sig.manizales.gov.co:6443/arcgis/rest/services/TIC/TIC_Delimitacion_Barrios_Comunas/MapServer",
            geo_fetchers=[fetch_manizales_oficial],
        ),
    ]
    return {c.clave: c for c in cities}


# ---------------------------------------------------------------------------
# Generación tabular (índices + SHAP + ABM)
# ---------------------------------------------------------------------------
def generar_indices(rng: np.random.Generator, unidades: list[str]) -> pd.DataFrame:
    """Panel sintético 15 años × UT con IVC = media de los 6 índices."""
    base = {u: rng.uniform(0.20, 0.80, size=len(IDX_COLS)) for u in unidades}
    rows = []
    for year in YEARS:
        drift = 0.01 * (year - YEARS[0]) / max(len(YEARS) - 1, 1)
        for u in unidades:
            vals = np.clip(
                base[u] + drift + rng.normal(0, 0.035, size=len(IDX_COLS)),
                0.0,
                1.0,
            )
            # Riesgo climático: más estacionario (proxy estructural)
            vals[IDX_COLS.index("idx_riesgo_clima")] = float(
                np.clip(0.45 + rng.normal(0, 0.04), 0.0, 1.0)
            )
            rec: dict[str, Any] = {"Año": year, "Unidad_Territorial": u}
            for c, v in zip(IDX_COLS, vals):
                rec[c] = float(v)
            rec["IVC"] = float(np.mean(vals))
            rows.append(rec)
    return pd.DataFrame(rows)[["Año", "Unidad_Territorial", *IDX_COLS, "IVC"]]


def generar_shap(rng: np.random.Generator, unidades: list[str]) -> pd.DataFrame:
    """
    SHAP sintético 2024: valores pequeños centrados (suma ≈ 0 por fila),
    coherente con un TreeExplainer sobre features normalizadas.
    """
    rows = []
    for u in unidades:
        raw = rng.normal(0.0, 0.02, size=len(SHAP_COLS))
        raw = raw - raw.mean()  # contribución neta ~0
        rec: dict[str, Any] = {"Unidad_Territorial": u}
        for c, v in zip(SHAP_COLS, raw):
            rec[c] = float(v)
        rows.append(rec)
    return pd.DataFrame(rows)[["Unidad_Territorial", *SHAP_COLS]]


def generar_abm_sintetico(
    rng: np.random.Generator, df_idx: pd.DataFrame, unidades: list[str]
) -> pd.DataFrame:
    """Trayectorias ABM mock (10 años × 3 escenarios) independientes por ciudad."""
    latest = int(df_idx["Año"].max())
    base = (
        df_idx[df_idx["Año"] == latest]
        .set_index("Unidad_Territorial")["IVC"]
        .to_dict()
    )
    scenarios = {"focalizado": 0.035, "adaptativo": 0.022, "distribuido": 0.012}
    rows = []
    n = max(len(unidades), 1)
    for sc, decay in scenarios.items():
        for year in range(1, 11):
            for u in unidades:
                iv0 = float(base.get(u, 0.5))
                noise = float(rng.normal(0, 0.003))
                boost = decay * (0.5 + iv0) if sc == "focalizado" else decay
                iv = float(np.clip(iv0 - boost * year + noise, 0.05, 0.95))
                wb = float(np.clip(0.45 + (iv0 - iv) * 1.2, 0.05, 0.95))
                rows.append(
                    {
                        "scenario": sc,
                        "year": year,
                        "comuna": u,
                        "ivc_initial": iv0,
                        "investment_allocated": float(1_000_000 / n),
                        "wellbeing": wb,
                        "ivc_current": iv,
                        "infrastructure_capacity": float(
                            np.clip(0.4 + year * 0.02 + rng.normal(0, 0.01), 0.1, 1.0)
                        ),
                        "cumulative_investment": float(year * 1_000_000 / n),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# GeoJSON mock (contingencia)
# ---------------------------------------------------------------------------
def generar_geojson_mock(city: CiudadSpec) -> dict:
    """Grilla de polígonos sintéticos alineados a las UTs oficiales."""
    lat0, lon0 = city.centro
    # Desplazamiento SW para centrar la grilla
    n = len(city.unidades)
    ncols = max(3, int(math.ceil(math.sqrt(n))))
    step = 0.035
    lon_sw = lon0 - (ncols * step) / 2
    lat_sw = lat0 - (math.ceil(n / ncols) * step) / 2

    features = []
    for i, name in enumerate(city.unidades):
        row, col = divmod(i, ncols)
        x = lon_sw + col * step
        y = lat_sw + row * step
        ring = [
            [x, y],
            [x + step * 0.90, y],
            [x + step * 0.90, y + step * 0.90],
            [x, y + step * 0.90],
            [x, y],
        ]
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "Unidad_Territorial": name,
                    "nombre": name,
                    "tipo": city.unidad_label,
                    "nota": "geometría MOCK — reemplazar con cartografía oficial",
                },
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )
    return {
        "type": "FeatureCollection",
        "name": f"{city.clave}_mock",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": features,
    }


def descargar_o_mockear_geo(city: CiudadSpec) -> tuple[dict, bool]:
    """
    Intenta cada fetcher oficial; si todos fallan → mock + aviso con portal.
    Returns: (geojson, es_oficial)
    """
    errors: list[str] = []
    for fetcher in city.geo_fetchers:
        try:
            print(f"  Intentando {fetcher.__name__} …")
            geo = fetcher(city)
            ok, msg = _es_feature_collection(geo, min_feats=1)
            if not ok:
                raise RuntimeError(msg)
            return geo, True
        except Exception as exc:  # noqa: BLE001 — requisito: try/except + mock
            errors.append(f"{fetcher.__name__}: {exc}")
            print(f"  FALLO {fetcher.__name__}: {exc}")

    print("\n  ⚠ Contingencia: se generará GeoJSON MOCK temporal.")
    print(f"  Descargar manualmente desde:\n    {city.portal_manual}\n")
    for e in errors:
        print(f"    · {e}")
    return generar_geojson_mock(city), False


# ---------------------------------------------------------------------------
# Exportación a disco (raíz + estructura pipeline)
# ---------------------------------------------------------------------------
def _centroid_ring(ring: list[list[float]]) -> tuple[float, float]:
    xs = [p[0] for p in ring[:-1]]
    ys = [p[1] for p in ring[:-1]]
    return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))


def _feature_centroid(geom: dict | None) -> Optional[tuple[float, float]]:
    if not geom:
        return None
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    try:
        if gtype == "Polygon":
            return _centroid_ring(coords[0])
        if gtype == "MultiPolygon":
            return _centroid_ring(coords[0][0])
    except (IndexError, TypeError, KeyError):
        return None
    return None


def exportar_ciudad(
    city: CiudadSpec,
    df_idx: pd.DataFrame,
    df_shap: pd.DataFrame,
    geo: dict,
    es_oficial: bool,
    rng: np.random.Generator,
) -> Path:
    """Escribe los 3 artefactos pedidos + espejo compatible con el pipeline."""
    out = ROOT / city.folder
    processed = out / "processed"
    models = out / "models"
    geo_dir = out / "geo"
    evidencia = out / "evidencia"
    logs = out / "logs"
    for d in (out, processed, models, geo_dir, evidencia, logs):
        d.mkdir(parents=True, exist_ok=True)

    # ── Archivo 1: indices_comunas.csv (raíz + processed) ───────────────────
    path_idx = out / "indices_comunas.csv"
    df_idx.to_csv(path_idx, index=False, encoding="utf-8-sig")
    # Esquema pipeline Medellín/Bogotá: ivc en minúsculas
    pipe_idx = df_idx.rename(columns={"IVC": "ivc"})
    pipe_idx.to_csv(processed / "indices_comunas.csv", index=False, encoding="utf-8-sig")
    print(f"  OK indices_comunas.csv ({len(df_idx)} filas)")

    ranking = (
        pipe_idx.groupby("Unidad_Territorial", as_index=False)["ivc"]
        .mean()
        .rename(columns={"ivc": "mean_ivc"})
        .sort_values("mean_ivc", ascending=False)
    )
    ranking.to_csv(models / "vulnerability_ranking.csv", index=False, encoding="utf-8-sig")

    # ── Archivo 2: shap_values.csv ──────────────────────────────────────────
    path_shap = out / "shap_values.csv"
    df_shap.to_csv(path_shap, index=False, encoding="utf-8-sig")
    # Espejo dashboard: columnas idx_* + Año (como Bogotá)
    shap_pipe = df_shap.rename(
        columns={
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
    print(f"  OK shap_values.csv ({len(df_shap)} filas)")

    # Copiar artefactos ML de Medellín (mismo esquema de 6 features) si existen
    src_models = ROOT / "models"
    for fname in (
        "vulnerability_model.pkl",
        "cv_metrics.json",
        "shap_importance_cv_stability.csv",
        "feature_importance.png",
        "shap_importance.png",
    ):
        src = src_models / fname
        if src.exists():
            shutil.copy2(src, models / fname)

    # ── Archivo 3: <ciudad>_oficial.geojson ─────────────────────────────────
    geo_name = f"{city.clave}_oficial.geojson"
    # Anotar procedencia
    for ft in geo.get("features", []):
        ft.setdefault("properties", {})
        ft["properties"].setdefault(
            "fuente_cartografia", "oficial" if es_oficial else "mock"
        )
    path_geo = out / geo_name
    path_geo.write_text(json.dumps(geo, ensure_ascii=False), encoding="utf-8")
    # Copia canónica para Folium: SOLO UTs administrativas (comuna/localidad).
    # Los barrios quedan en <ciudad>_oficial.geojson para inventario, pero no
    # entran al coropleto (el panel de índices es por UT oficial).
    principal = {
        "type": "FeatureCollection",
        "name": f"{city.clave}_uts",
        "crs": geo.get("crs"),
        "features": [
            ft
            for ft in geo.get("features", [])
            if str((ft.get("properties") or {}).get("tipo", "")).lower()
            in ("", city.unidad_label, "comuna", "localidad", "sector")
            or (ft.get("properties") or {}).get("Unidad_Territorial") in city.unidades
        ],
    }
    # Deduplicar por nombre canónico
    seen_names: set[str] = set()
    deduped = []
    for ft in principal["features"]:
        nm = (ft.get("properties") or {}).get("Unidad_Territorial")
        if not nm or nm in seen_names:
            continue
        # Preferir features cuyo nombre esté en el catálogo oficial
        if city.unidades and nm not in city.unidades:
            # Mantener si no hay match exacto pero tipo es UT principal
            tipo = str((ft.get("properties") or {}).get("tipo", "")).lower()
            if tipo not in (city.unidad_label, "comuna", "localidad"):
                continue
        seen_names.add(nm)
        deduped.append(ft)
    principal["features"] = deduped or [
        ft
        for ft in geo.get("features", [])
        if (ft.get("properties") or {}).get("Unidad_Territorial") in city.unidades
    ]
    (geo_dir / "localidades.geojson").write_text(
        json.dumps(principal, ensure_ascii=False), encoding="utf-8"
    )
    n_feat = len(geo.get("features", []))
    n_ut = len(principal["features"])
    tag = "OFICIAL" if es_oficial else "MOCK"
    print(f"  OK {geo_name} ({n_feat} features, {tag}; geo/localidades={n_ut} UTs)")

    # ── ABM sintético ───────────────────────────────────────────────────────
    df_abm = generar_abm_sintetico(rng, df_idx, city.unidades)
    df_abm.to_csv(
        evidencia / "resultados_simulacion_escenarios.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"  OK evidencia/resultados_simulacion_escenarios.csv ({len(df_abm)} filas)")

    # ── config_<ciudad>.yaml (opcional, para cablear Streamlit) ─────────────
    _escribir_config(city, geo)
    return out


def _escribir_config(city: CiudadSpec, geo: dict) -> None:
    centros: dict[str, tuple[float, float]] = {}
    for ft in geo.get("features", []):
        props = ft.get("properties") or {}
        if props.get("tipo") not in (None, city.unidad_label, "comuna", "localidad"):
            # Preferir centroides de UT principales para el config
            if props.get("tipo") == "barrio":
                continue
        name = props.get("Unidad_Territorial") or props.get("nombre")
        c = _feature_centroid(ft.get("geometry"))
        if name and c:
            lon, lat = c
            centros[str(name)] = (lat, lon)

    terr_lines = []
    for i, name in enumerate(city.unidades, start=1):
        lat, lon = centros.get(name, city.centro)
        terr_lines.append(
            f'  - {{id: {i}, nombre: "{name}", lat: {lat:.6f}, lon: {lon:.6f}}}'
        )

    cfg = f"""# config_{city.clave}.yaml — perfil de portabilidad
# Datos de portabilidad / escalabilidad (índices sintéticos; cartografía según fuente).
ciudad:
  nombre: "{city.nombre}"
  departamento: "{city.departamento}"
  codigo_dane: "{city.codigo_dane}"
  unidad_territorial: "{city.unidad_label}"
  crs: "EPSG:4326"
  centro:
    lat: {city.centro[0]}
    lon: {city.centro[1]}

rutas:
  datos_climaticos: "{city.folder}"
  datos_sociales: "{city.folder}"
  procesados: "{city.folder}/processed"
  modelos: "{city.folder}/models"
  logs: "{city.folder}/logs"
  geo: "{city.folder}/geo"

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
    path = ROOT / f"config_{city.clave}.yaml"
    path.write_text(cfg, encoding="utf-8")
    print(f"  OK config_{city.clave}.yaml")


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------
def procesar_ciudad(city: CiudadSpec, rng: np.random.Generator) -> None:
    print("\n" + "=" * 72)
    print(f"{city.nombre.upper()}  ·  {len(city.unidades)} {city.unidad_label}(s)")
    print(f"Carpeta: {city.folder}")
    print("=" * 72)

    df_idx = generar_indices(rng, city.unidades)
    df_shap = generar_shap(rng, city.unidades)
    geo, es_oficial = descargar_o_mockear_geo(city)
    exportar_ciudad(city, df_idx, df_shap, geo, es_oficial, rng)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Genera Datos_*_Beta para Cali, Barranquilla, Bucaramanga, Pereira y Manizales."
    )
    p.add_argument(
        "--solo",
        type=str,
        default="",
        help="Lista separada por comas de claves (ej. cali,pereira). Vacío = todas.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    catalog = build_catalog()
    seleccion = (
        [s.strip().lower() for s in args.solo.split(",") if s.strip()]
        if args.solo
        else list(catalog.keys())
    )
    unknown = [s for s in seleccion if s not in catalog]
    if unknown:
        print(f"Ciudades desconocidas: {unknown}", file=sys.stderr)
        print(f"Válidas: {', '.join(catalog)}", file=sys.stderr)
        return 2

    print("=== Generación multi-ciudad — portabilidad Prioriza / TDG ===")
    print("Índices y SHAP son SINTÉTICOS (demo de escalabilidad).")
    print("La cartografía intenta ser OFICIAL; si falla → mock + enlace portal.\n")

    # Semilla distinta por ciudad (reproducible pero no idéntica entre ciudades)
    for i, clave in enumerate(seleccion):
        city = catalog[clave]
        rng = np.random.default_rng(SEED + i * 17)
        try:
            procesar_ciudad(city, rng)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR FATAL en {city.nombre}: {exc}", file=sys.stderr)
            return 1

    print("\n" + "=" * 72)
    print("Listo. Carpetas generadas:")
    for clave in seleccion:
        print(f"  · {catalog[clave].folder}/")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
