"""
download_geo_real.py
====================
Descarga los límites reales de comunas de Medellín desde el servidor ArcGIS
oficial de la Alcaldía de Medellín y genera data/geo/comunas.geojson
con los nombres normalizados y alineados al modelo predictivo.

Ejecución:
    py download_geo_real.py
"""
import sys, json, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape

# ── URLs del servidor oficial de la Alcaldía de Medellín ──────────────────────
# Layer 2 = "Zonas vigentes de Medellín" (comunas y corregimientos)
# Layer 1 = "Comunas y Corregimientos" (límite catastral)
ARCGIS_URLS = [
    (
        "Alcaldia Medellin — Zonas vigentes (comunas)",
        "https://www.medellin.gov.co/servidormapas/rest/services/"
        "mapas_nacionales/VC_Limite_Politico_Admtivo/MapServer/2/query"
        "?where=1%3D1&outFields=*&f=geojson&returnGeometry=true",
    ),
    (
        "Alcaldia Medellin — Limite catastral comunas",
        "https://www.medellin.gov.co/servidormapas/rest/services/"
        "mapas_nacionales/VC_Limite_Politico_Admtivo/MapServer/1/query"
        "?where=1%3D1&outFields=*&f=geojson&returnGeometry=true",
    ),
    (
        "Alcaldia Medellin — CartografiaBase LimiteComunaCorregimiento",
        "https://www.medellin.gov.co/servidormapas/rest/services/"
        "ServiciosCiudad/CartografiaBase/MapServer/11/query"
        "?where=1%3D1&outFields=*&f=geojson&returnGeometry=true",
    ),
]

# ── Mapeo oficial código DANE → nombre canónico del modelo ────────────────────
# Fuente: Acuerdo 54/1987 + vulnerability_ranking.csv del proyecto
CODIGO_TO_NOMBRE = {
    "01": "Popular",
    "02": "Santa Cruz",
    "03": "Manrique",
    "04": "Aranjuez",
    "05": "Castilla",
    "06": "Doce de Octubre",
    "07": "Robledo",
    "08": "Villa Hermosa",
    "09": "Buenos Aires",
    "10": "La Candelaria",
    "11": "Laureles Estadio",
    "12": "La América",
    "13": "San Javier",
    "14": "El Poblado",
    "15": "Guayabal",
    "16": "Belén",
    # Corregimientos (sin índice en el modelo, se excluyen si no aparecen)
    "50": "Palmitas",
    "60": "San Cristóbal",
    "70": "Altavista",
    "80": "San Antonio de Prado",
    "90": "Santa Elena",
}

NAME_ALIASES = {
    "popular":              "Popular",
    "santa cruz":           "Santa Cruz",
    "manrique":             "Manrique",
    "aranjuez":             "Aranjuez",
    "castilla":             "Castilla",
    "doce de octubre":      "Doce de Octubre",
    "robledo":              "Robledo",
    "villa hermosa":        "Villa Hermosa",
    "villahermosa":         "Villa Hermosa",
    "buenos aires":         "Buenos Aires",
    "la candelaria":        "La Candelaria",
    "candelaria":           "La Candelaria",
    "laureles":             "Laureles Estadio",
    "laureles estadio":     "Laureles Estadio",
    "laureles - estadio":   "Laureles Estadio",
    "la america":           "La América",
    "la america":           "La América",
    "san javier":           "San Javier",
    "el poblado":           "El Poblado",
    "poblado":              "El Poblado",
    "guayabal":             "Guayabal",
    "belen":                "Belén",
    "belen":                "Belén",
}


def _norm(s: str) -> str:
    return (
        unicodedata.normalize("NFD", str(s))
        .encode("ascii", "ignore")
        .decode()
        .lower()
        .strip()
    )


def resolve_name(props: dict) -> str | None:
    """Intenta obtener el nombre canónico a partir de las propiedades del feature."""
    # 1. Por código numérico (más fiable)
    for key in ("CODIGO", "codigo", "COD_COMUNA", "cod_comuna", "NMG", "nmg",
                "OBJECTID", "ID_COMUNA"):
        val = props.get(key)
        if val is not None:
            code = str(val).strip().zfill(2)
            if code in CODIGO_TO_NOMBRE:
                return CODIGO_TO_NOMBRE[code]

    # 2. Por nombre textual
    for key in ("NOMBRE", "nombre", "NOM_COM", "nom_com", "NOMB_COM",
                "nombre_com", "NOMBR", "nombre_comuna"):
        val = props.get(key)
        if val:
            norm = _norm(str(val))
            if norm in NAME_ALIASES:
                return NAME_ALIASES[norm]
            # Coincidencia parcial
            for alias, canonical in NAME_ALIASES.items():
                if alias in norm or norm in alias:
                    return canonical
    return None


def try_download(label: str, url: str) -> gpd.GeoDataFrame | None:
    print(f"\n[DESCARGA] {label}")
    print(f"  URL: {url[:80]}...")
    try:
        r = requests.get(url, timeout=30,
                         headers={"User-Agent": "Mozilla/5.0 (academic research)"})
        if r.status_code != 200:
            print(f"  HTTP {r.status_code} — saltando.")
            return None

        data = r.json()
        if "features" not in data or not data["features"]:
            print("  Sin features — saltando.")
            return None

        print(f"  Features recibidos: {len(data['features'])}")
        # Inspeccionar propiedades del primer feature
        sample_props = data["features"][0].get("properties", {})
        print(f"  Propiedades disponibles: {list(sample_props.keys())}")

        # Construir GeoDataFrame con columna controlada de nombre resuelto
        records = []
        for feat in data["features"]:
            props = feat.get("properties", {})
            geom  = shape(feat["geometry"]) if feat.get("geometry") else None
            if geom is None or geom.is_empty:
                continue
            nombre_resuelto = resolve_name(props)
            # Guardamos solo código y nombre oficial para evitar conflictos
            records.append({
                "codigo_orig": props.get("codigo", ""),
                "nombre_orig": props.get("nombre", ""),
                "_nombre_resuelto": nombre_resuelto,
                "geometry": geom,
            })

        gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
        resueltos = gdf["_nombre_resuelto"].notna().sum()
        print(f"  Nombres resueltos: {resueltos}/{len(gdf)}")
        print(f"  Muestra de nombres originales: {gdf['nombre_orig'].tolist()}")
        return gdf

    except Exception as e:
        print(f"  ERROR: {e}")
        return None


# ── Intentar descarga ─────────────────────────────────────────────────────────
gdf_raw = None
for label, url in ARCGIS_URLS:
    gdf_raw = try_download(label, url)
    if gdf_raw is not None and gdf_raw["_nombre_resuelto"].notna().sum() >= 10:
        break

if gdf_raw is None or gdf_raw["_nombre_resuelto"].notna().sum() < 5:
    print("\n[ERROR] No se pudo descargar el GeoJSON real desde el servidor oficial.")
    print("        Verifica tu conexión a internet o accede manualmente a:")
    print("        https://www.medellin.gov.co/geomedellin/")
    sys.exit(1)

# ── Reproyectar a WGS84 si es necesario ──────────────────────────────────────
print(f"\n[CRS original] {gdf_raw.crs}")
if gdf_raw.crs is None:
    gdf_raw = gdf_raw.set_crs(epsg=4326)
elif str(gdf_raw.crs).upper() != "EPSG:4326":
    print(f"  Reproyectando a EPSG:4326 ...")
    gdf_raw = gdf_raw.to_crs(epsg=4326)
print(f"[CRS final]    {gdf_raw.crs}")

# ── Filtrar comunas urbanas con nombre resuelto ───────────────────────────────
gdf_comunas = gdf_raw[gdf_raw["_nombre_resuelto"].notna()].copy()
gdf_comunas = gdf_comunas.rename(columns={"_nombre_resuelto": "nombre"})

# Asegurar columna 'nombre' limpia y simplificar geometría para performance
gdf_comunas["nombre"]   = gdf_comunas["nombre"].astype(str)
gdf_comunas["geometry"] = gdf_comunas["geometry"].simplify(0.0001, preserve_topology=True)
gdf_comunas = gdf_comunas[["nombre", "geometry"]].reset_index(drop=True)

print(f"\n[COMUNAS IDENTIFICADAS]")
for _, row in gdf_comunas.sort_values("nombre").iterrows():
    print(f"  {row['nombre']}")

# ── Guardar ───────────────────────────────────────────────────────────────────
out_dir = ROOT / "data" / "geo"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "comunas.geojson"
gdf_comunas.to_file(out_path, driver="GeoJSON", encoding="utf-8")
print(f"\n[OK] GeoJSON real guardado: {out_path}")
print(f"     Polígonos: {len(gdf_comunas)}")
print(f"     CRS: {gdf_comunas.crs}")
print(f"     Bounds: {gdf_comunas.total_bounds}")
print("\n[INFO] Recarga el dashboard con Ctrl+R para ver el mapa real.")
