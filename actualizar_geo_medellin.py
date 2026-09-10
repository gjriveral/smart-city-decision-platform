"""
actualizar_geo_medellin.py
--------------------------
Descarga la capa oficial «Comunas y Corregimientos» del geoportal de Medellín
(Alcaldía / servidormapas), fusiona polígonos sin nombre (SN01/SN02) con la
comuna vecina, rellena huecos internos residuales y escribe
data/geo/comunas.geojson listo para Folium.

Uso:
    py -X utf8 actualizar_geo_medellin.py
"""

from __future__ import annotations

import io
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "geo" / "comunas.geojson"
BACKUP_DIR = ROOT / "data" / "geo" / "_backup"

URL = (
    "https://www.medellin.gov.co/servidormapas/rest/services/"
    "mapas_nacionales/VC_Limite_Politico_Admtivo/MapServer/1/query"
)
PARAMS = {
    "where": "1=1",
    "outFields": "*",
    "outSR": "4326",
    "f": "geojson",
    "resultRecordCount": "100",
}
HEADERS = {"User-Agent": "Prioriza-TDG-UdeA/1.0 (cartografia comunas Medellin)"}


def _count_holes(geoms) -> int:
    u = unary_union([make_valid(g) for g in geoms if g is not None])
    polys = list(u.geoms) if u.geom_type == "MultiPolygon" else [u]
    return sum(len(list(p.interiors)) for p in polys if hasattr(p, "interiors"))


def _fill_interior_holes(gdf: gpd.GeoDataFrame, name_col: str) -> gpd.GeoDataFrame:
    """Asigna cada hueco interno a la UT con mayor intersección (buffer 120 m)."""
    out = gdf.copy()
    proj = out.to_crs(epsg=3116)
    max_rounds = 5
    for round_i in range(1, max_rounds + 1):
        union = unary_union([make_valid(g) for g in proj.geometry])
        polys = list(union.geoms) if union.geom_type == "MultiPolygon" else [union]
        holes: list[Polygon] = []
        for p in polys:
            for ring in getattr(p, "interiors", []):
                hole = Polygon(ring)
                if hole.area > 0.5:  # m²
                    holes.append(hole)
        if not holes:
            print(f"  Relleno: 0 huecos (ronda {round_i})")
            break
        print(f"  Relleno ronda {round_i}: {len(holes)} huecos")
        geom_list = [make_valid(g) for g in proj.geometry]
        names = out[name_col].tolist()
        for hole in holes:
            best_i, best_a = None, 0.0
            buffered = hole.buffer(120)
            for i, g in enumerate(geom_list):
                a = g.intersection(buffered).area
                if a > best_a:
                    best_a, best_i = a, i
            if best_i is None:
                dists = [g.distance(hole) for g in geom_list]
                best_i = int(min(range(len(dists)), key=lambda i: dists[i]))
            geom_list[best_i] = make_valid(geom_list[best_i].union(hole))
            print(f"    {hole.area:.0f} m² → {names[best_i]}")
        proj = proj.set_geometry(geom_list)

    out = proj.to_crs(epsg=4326)
    return out


def main() -> None:
    print("Descargando capa oficial de Medellín…")
    resp = requests.get(URL, params=PARAMS, headers=HEADERS, timeout=180)
    resp.raise_for_status()
    raw = gpd.read_file(io.BytesIO(resp.content))
    print(f"  Features recibidas: {len(raw)}")

    named = raw[raw["nombre"].notna()].copy()
    orphan = raw[raw["nombre"].isna()].copy()
    print(f"  Con nombre: {len(named)} | sin nombre (SN): {len(orphan)}")
    print(f"  Huecos (solo nombrados): {_count_holes(named.geometry)}")

    # Fusionar SN → comuna/corregimiento vecino (CRS métrico)
    if len(orphan):
        named_m = named.to_crs(epsg=3116)
        orphan_m = orphan.to_crs(epsg=3116)
        geom_map = {
            i: make_valid(g) for i, g in named_m.geometry.items()
        }
        for _, row in orphan_m.iterrows():
            best_i, best_a = None, 0.0
            for i, g in geom_map.items():
                a = row.geometry.intersection(g.buffer(50)).area
                if a > best_a:
                    best_a, best_i = a, i
            if best_i is None or best_a <= 0:
                dists = {
                    i: g.distance(row.geometry) for i, g in geom_map.items()
                }
                best_i = min(dists, key=dists.get)
            target_name = named_m.loc[best_i, "nombre"]
            geom_map[best_i] = make_valid(geom_map[best_i].union(row.geometry))
            print(f"  SN {row.get('codigo')} → {target_name}")
        named_m = named_m.copy()
        named_m["geometry"] = [geom_map[i] for i in named_m.index]
        named = named_m.to_crs(epsg=4326)

    print(f"  Huecos tras fusionar SN: {_count_holes(named.geometry)}")

    # Rellenar huecos internos residuales
    named = _fill_interior_holes(named, "nombre")
    print(f"  Huecos finales: {_count_holes(named.geometry)}")

    # Esquema canónico para el dashboard (sin simplify — destruye topología)
    out = gpd.GeoDataFrame(
        {
            "nombre": named["nombre"].astype(str),
            "Comuna": named["nombre"].astype(str),
            "Unidad_Territorial": named["nombre"].astype(str),
            "codigo": named["codigo"].astype(str),
            "identificacion": named.get(
                "identificacion", pd.Series([""] * len(named))
            ).astype(str),
            "fuente": "Alcaldía de Medellín — VC_Limite_Politico_Admtivo",
        },
        geometry=named.geometry,
        crs="EPSG:4326",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        bak = BACKUP_DIR / f"comunas_backup_{stamp}.geojson"
        shutil.copy2(OUT, bak)
        print(f"  Backup: {bak.name}")

    out.to_file(OUT, driver="GeoJSON")
    size_mb = OUT.stat().st_size / 1e6
    print(f"OK  {OUT}  ({len(out)} UTs, {size_mb:.2f} MB)")

    # Verificación rápida de join con índices
    idx_path = (
        ROOT / "ModeloDatos" / "ModeloDatos" / "data" / "processed" / "indices_comunas.csv"
    )
    if idx_path.exists():
        import unicodedata

        idx = pd.read_csv(idx_path, encoding="utf-8-sig")
        latest = idx[idx["Año"] == idx["Año"].max()]
        aliases = {"san sebastian de palmitas": "palmitas", "palmitas": "palmitas"}

        def norm(s: str) -> str:
            key = (
                unicodedata.normalize("NFD", str(s))
                .encode("ascii", "ignore")
                .decode()
                .lower()
                .strip()
            )
            for p in (
                "corregimiento de ",
                "corregimiento ",
                "comuna ",
                "localidad ",
                "la ",
            ):
                if key.startswith(p):
                    key = key[len(p) :]
                    break
            return aliases.get(key, key)

        gkeys = set(out["nombre"].map(norm))
        ckeys = set(latest["Comuna"].map(norm))
        print(f"  Join con índices: {len(gkeys & ckeys)}/{len(gkeys)} (geo) vs {len(ckeys)} CSV")
        print(f"  Solo geo: {sorted(gkeys - ckeys)}")
        print(f"  Solo CSV: {sorted(ckeys - gkeys)}")
        print(f"  Huecos en archivo final: {_count_holes(out.geometry)}")


if __name__ == "__main__":
    main()
