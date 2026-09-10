"""
descargar_geojson_bogota.py
---------------------------
Descarga el GeoJSON oficial de las 20 localidades de Bogotá D.C. desde
una API REST pública (IDECA / Catastro Bogotá — ArcGIS Feature Service)
y lo guarda en Datos_Bogotá_Beta/bogota_localidades.geojson.

Fuente primaria (EPSG:4326, GeoJSON estándar):
  serviciosgis.catastrobogota.gov.co
  .../ordenamientoterritorial/localidad/MapServer/0/query?f=geojson

Uso (desde la raíz del repositorio):
    py -X utf8 descargar_geojson_bogota.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "Datos_Bogotá_Beta"
OUT_FILE = OUT_DIR / "bogota_localidades.geojson"
# Copia normalizada para el dashboard (reemplaza el dummy de portabilidad)
GEO_PLATFORM = OUT_DIR / "geo" / "localidades.geojson"

# Nombres canónicos del CSV beta / config (Title Case con tildes) ↔ LOCNOMBRE IDECA
LOCNOMBRE_A_CANONICO: dict[str, str] = {
    "ANTONIO NARIÑO": "Antonio Nariño",
    "BARRIOS UNIDOS": "Barrios Unidos",
    "BOSA": "Bosa",
    "CANDELARIA": "La Candelaria",
    "CHAPINERO": "Chapinero",
    "CIUDAD BOLIVAR": "Ciudad Bolívar",
    "ENGATIVA": "Engativá",
    "FONTIBON": "Fontibón",
    "KENNEDY": "Kennedy",
    "LOS MARTIRES": "Los Mártires",
    "PUENTE ARANDA": "Puente Aranda",
    "RAFAEL URIBE URIBE": "Rafael Uribe Uribe",
    "SAN CRISTOBAL": "San Cristóbal",
    "SANTA FE": "Santa Fe",
    "SUBA": "Suba",
    "SUMAPAZ": "Sumapaz",
    "TEUSAQUILLO": "Teusaquillo",
    "TUNJUELITO": "Tunjuelito",
    "USAQUEN": "Usaquén",
    "USME": "Usme",
}

# API REST pública de cartografía oficial (SDP / Catastro Bogotá — IDECA)
URL_ARCGIS = (
    "https://serviciosgis.catastrobogota.gov.co/arcgis/rest/services/"
    "ordenamientoterritorial/localidad/MapServer/0/query"
)
PARAMS_ARCGIS = {
    "where": "1=1",
    "outFields": "*",
    "f": "geojson",
    "outSR": "4326",  # WGS84 — compatible con Folium/Streamlit
}

# Fallback: recurso GeoJSON publicado en Datos Abiertos Bogotá (CKAN).
# Nota: a veces entrega envoltorio Esri; se valida y se usa solo si es
# FeatureCollection GeoJSON estándar.
URL_CKAN = (
    "https://datosabiertos.bogota.gov.co/dataset/"
    "856cb657-8ca3-4ee8-857f-37211173b1f8/resource/"
    "497b8756-0927-4aee-8da9-ca4e32ca3a8a/download/localidad_bogota_dc.geojson"
)

HEADERS = {
    "User-Agent": "Prioriza-TDG-UdeA/1.0 (descarga cartografia abierta)",
    "Accept": "application/geo+json, application/json",
}
TIMEOUT_S = 120


def _es_geojson_valido(payload: object) -> tuple[bool, str]:
    """Comprueba que el objeto sea un GeoJSON FeatureCollection usable."""
    if not isinstance(payload, dict):
        return False, "La respuesta no es un objeto JSON (dict)."
    ftype = payload.get("type")
    feats = payload.get("features")
    if ftype != "FeatureCollection":
        return False, f"type={ftype!r}; se esperaba 'FeatureCollection'."
    if not isinstance(feats, list) or len(feats) == 0:
        return False, "FeatureCollection sin features."
    # Localidades oficiales de Bogotá = 20
    if len(feats) < 20:
        return False, f"Solo {len(feats)} features; se esperaban 20 localidades."
    sample = feats[0]
    if not isinstance(sample, dict) or "geometry" not in sample:
        return False, "La primera feature no tiene 'geometry'."
    return True, f"OK — {len(feats)} features"


def _descargar(url: str, params: dict | None = None) -> dict:
    """GET + validación HTTP 200 + parseo JSON."""
    print(f"Descargando:\n  {url}")
    resp = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT_S)
    if resp.status_code != 200:
        raise RuntimeError(
            f"HTTP {resp.status_code} al descargar {resp.url}"
        )
    try:
        payload = resp.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"La respuesta no es JSON válido ({exc}). "
            f"Content-Type={resp.headers.get('Content-Type')!r}"
        ) from exc
    ok, msg = _es_geojson_valido(payload)
    if not ok:
        raise RuntimeError(f"JSON recibido pero no es GeoJSON usable: {msg}")
    print(f"  Validación: {msg}")
    return payload


def instalar_en_plataforma(geo: dict) -> Path:
    """
    Escribe geo/localidades.geojson con propiedades `nombre` y `Localidad`
    alineadas al CSV del beta, para que Folium una 20/20 polígonos.
    """
    GEO_PLATFORM.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "type": "FeatureCollection",
        "name": "bogota_localidades_oficial_ideca",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": [],
    }
    for ft in geo.get("features", []):
        props = dict(ft.get("properties") or {})
        raw = str(props.get("LOCNOMBRE") or props.get("nombre") or "").strip()
        canon = LOCNOMBRE_A_CANONICO.get(raw.upper(), raw.title())
        props["LOCNOMBRE"] = raw
        props["nombre"] = canon
        props["Localidad"] = canon
        props["fuente"] = "IDECA / Catastro Bogotá (ArcGIS REST)"
        out["features"].append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": ft.get("geometry"),
            }
        )
    GEO_PLATFORM.write_text(
        json.dumps(out, ensure_ascii=False),
        encoding="utf-8",
    )
    return GEO_PLATFORM


def main() -> int:
    print("=== Descarga GeoJSON oficial — Localidades de Bogotá ===\n")

    # 1) Carpeta de destino
    if not OUT_DIR.exists():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Carpeta creada: {OUT_DIR}")
    else:
        print(f"Carpeta existente: {OUT_DIR}")

    # 2) Descarga (primaria → fallback)
    geo = None
    errores: list[str] = []
    try:
        geo = _descargar(URL_ARCGIS, PARAMS_ARCGIS)
        print("  Fuente: ArcGIS REST Catastro Bogotá / IDECA")
    except Exception as exc:
        errores.append(f"ArcGIS: {exc}")
        print(f"  Falló fuente primaria: {exc}")
        try:
            geo = _descargar(URL_CKAN)
            print("  Fuente: Datos Abiertos Bogotá (CKAN)")
        except Exception as exc2:
            errores.append(f"CKAN: {exc2}")
            print("ERROR: no se pudo descargar desde ninguna fuente.")
            for e in errores:
                print(f"  - {e}")
            return 1

    # 3) Guardado crudo
    try:
        OUT_FILE.write_text(
            json.dumps(geo, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"ERROR al guardar {OUT_FILE}: {exc}")
        return 1

    # 4) Re-lectura de control
    try:
        recheck = json.loads(OUT_FILE.read_text(encoding="utf-8"))
        ok, msg = _es_geojson_valido(recheck)
        if not ok:
            print(f"ERROR: archivo guardado inválido ({msg})")
            return 1
    except Exception as exc:
        print(f"ERROR al revalidar el archivo: {exc}")
        return 1

    # 5) Instalar en ruta que consume Streamlit (reemplaza dummy)
    try:
        installed = instalar_en_plataforma(recheck)
        print(f"OK  Instalado para el dashboard → {installed}")
    except Exception as exc:
        print(f"ERROR al instalar en geo/localidades.geojson: {exc}")
        return 1

    size_kb = OUT_FILE.stat().st_size / 1024
    print(
        f"\nÉXITO: cartografía oficial lista\n"
        f"  Crudo:      {OUT_FILE} ({size_kb:,.1f} KB)\n"
        f"  Plataforma: {GEO_PLATFORM}\n"
        f"  {msg}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
