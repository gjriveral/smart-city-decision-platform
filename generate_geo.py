"""
generate_geo.py
===============
Genera data/geo/comunas.geojson con polígonos de Voronoi anclados a los
centroides definidos en config.yaml. El resultado es compatible con el
mapa Folium del dashboard (columna 'nombre' = nombre exacto de la comuna).

Ejecución:
    py generate_geo.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import json
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, MultiPolygon, mapping, Point
from shapely.ops import unary_union
import geopandas as gpd

from config_loader import get_config

# ── Mapa de corrección de nombres (config.yaml → indices_comunas.csv) ─────────
# Necesario cuando el nombre canónico del config difiere del usado en el CSV de
# índices. El valor aquí es el que aparece en indices_comunas.csv (y en df_latest
# del dashboard), por lo que debe coincidir exactamente para que Folium haga el
# merge correctamente.
NAME_OVERRIDES: dict[str, str] = {
    "Laureles": "Laureles Estadio",
}

# ── Cargar configuración ───────────────────────────────────────────────────────
cfg = get_config()
territorios = cfg.territorios

print(f"[INFO] Territorios cargados: {len(territorios)}")
for t in territorios:
    print(f"  {t.id:>2}. {t.nombre:<30}  lat={t.lat}  lon={t.lon}")

# ── Crear directorio de salida ─────────────────────────────────────────────────
geo_dir = ROOT / "data" / "geo"
geo_dir.mkdir(parents=True, exist_ok=True)
out_path = geo_dir / "comunas.geojson"

# ── Construir polígonos de Voronoi ─────────────────────────────────────────────
# Coordenadas de los centroides [lon, lat]
coords = np.array([[t.lon, t.lat] for t in territorios])

# Bounding box de Medellín con margen (para recortar Voronoi infinito)
MARGIN = 0.04
bbox = Polygon([
    (coords[:, 0].min() - MARGIN, coords[:, 1].min() - MARGIN),
    (coords[:, 0].max() + MARGIN, coords[:, 1].min() - MARGIN),
    (coords[:, 0].max() + MARGIN, coords[:, 1].max() + MARGIN),
    (coords[:, 0].min() - MARGIN, coords[:, 1].max() + MARGIN),
])

# Voronoi necesita al menos 4 puntos; añadimos puntos auxiliares en las esquinas
# del bbox para que ninguna celda sea infinita.
aux_points = np.array([
    [coords[:, 0].min() - MARGIN * 2, coords[:, 1].min() - MARGIN * 2],
    [coords[:, 0].max() + MARGIN * 2, coords[:, 1].min() - MARGIN * 2],
    [coords[:, 0].max() + MARGIN * 2, coords[:, 1].max() + MARGIN * 2],
    [coords[:, 0].min() - MARGIN * 2, coords[:, 1].max() + MARGIN * 2],
])
all_points = np.vstack([coords, aux_points])

vor = Voronoi(all_points)

# Reconstruir polígono de cada región de Voronoi y recortar al bbox
def voronoi_finite_polygon(vor, region_idx, bbox):
    region = vor.regions[region_idx]
    if -1 in region or not region:
        return None
    polygon = Polygon([vor.vertices[i] for i in region])
    return polygon.intersection(bbox)

# Mapear punto_original → región de Voronoi
point_region_map = {}
for point_idx, region_idx in enumerate(vor.point_region):
    point_region_map[point_idx] = region_idx

features = []
for i, t in enumerate(territorios):
    region_idx = point_region_map[i]
    poly = voronoi_finite_polygon(vor, region_idx, bbox)
    if poly is None or poly.is_empty:
        # Fallback: buffer circular de 0.015° (~1.7 km) recortado al bbox
        poly = Point(t.lon, t.lat).buffer(0.015).intersection(bbox)

    # Simplificar ligeramente para reducir tamaño del archivo
    poly = poly.simplify(0.0005, preserve_topology=True)

    # Aplica corrección de nombre si existe (para alinear con indices_comunas.csv)
    nombre_final = NAME_OVERRIDES.get(t.nombre, t.nombre)

    features.append({
        "type": "Feature",
        "properties": {
            "id":     t.id,
            "nombre": nombre_final,
            "Comuna": nombre_final,   # alias adicional para compatibilidad
            "lat":    t.lat,
            "lon":    t.lon,
        },
        "geometry": mapping(poly),
    })

geojson = {
    "type": "FeatureCollection",
    "crs": {
        "type": "name",
        "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
    },
    "features": features,
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(geojson, f, ensure_ascii=False, indent=2)

print(f"\n[OK] GeoJSON generado: {out_path}")
print(f"     Polígonos: {len(features)}")

# ── Verificar con geopandas ────────────────────────────────────────────────────
gdf = gpd.read_file(out_path)
print(f"     CRS: {gdf.crs}")
print(f"     Columnas: {gdf.columns.tolist()}")
print(f"     Área media del polígono (grados²): {gdf.geometry.area.mean():.6f}")
print(f"\n[VERIFICACION] Primeras 5 comunas:")
print(gdf[["nombre", "geometry"]].head().to_string())
print("\n[INFO] El dashboard detectará el archivo en la próxima recarga (Ctrl+R).")
