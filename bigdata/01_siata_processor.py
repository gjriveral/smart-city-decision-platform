"""
01_siata_processor.py
=====================
SIATA Pluviometric Network — Big Data Processing Pipeline
Project: Inversión Estratégica en Ciudades — UdeA TG 2026
Author : pipeline de ingeniería de datos automatizado — UdeA TG 2026
Date   : 2026-05-25

CSV Structure (confirmed by inspection):
    codigo    : int   — station ID (same as in filename)
    fecha_hora: str   — datetime 'YYYY-MM-DD HH:MM:SS'
    p1        : float — precipitation reading 1 (mm), 1-minute accumulation
    p2        : float — precipitation reading 2 (mm), alternative sensor
    calidad   : int   — quality flag (1 = valid)

Pipeline:
    1. Discover CSV files in <rutas.datos_climaticos>/
    2. (SAMPLE_MODE) take first SAMPLE_N files
    3. Read with dask.dataframe (lazy, parallel)
    4. Filter calidad == 1
    5. Parse year from fecha_hora
    6. Map each station to nearest territorial-unit centroid (haversine)
    7. Aggregate per (estacion_id, año, <unidad_territorial>):
         precip_media_mm  = mean(p1)
         precip_max_mm    = max(p1)
         precip_p95_mm    = 95th percentile(p1)
         n_registros      = count
    8. Write <rutas.procesados>/siata_agregado.csv

NOTE on station coordinates:
    The CSV files do not contain explicit lat/lon columns; only station codes.
    The station catalog is loaded from config.yaml (catalogo_estaciones).
    For stations not in the catalog, a deterministic hash of station_id is used
    to assign a pseudo-territory (clearly flagged in output).

Configuración de ciudad:
    Toda referencia a 'Medellín', 'comuna' o rutas hardcoded ha sido
    reemplazada por variables leídas de config.yaml vía config_loader.
    Para replicar en otra ciudad, edita únicamente config.yaml.
"""

import os
import re
import math
import hashlib
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

import pandas as pd

# ─── Config de ciudad (leída desde config.yaml) ───────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config_loader import get_config

cfg         = get_config()
UT          = cfg.unidad_territorial          # "comuna", "localidad", etc.
_abs_rutas  = cfg.rutas_absolutas()

ROOT_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = Path(_abs_rutas.datos_climaticos)
OUT_DIR     = Path(_abs_rutas.procesados)
LOG_DIR     = Path(_abs_rutas.logs)
OUT_FILE    = OUT_DIR / "siata_agregado.csv"
LOG_FILE    = LOG_DIR / "bigdata_execution.log"

SAMPLE_MODE = True    # Set False to process all files
SAMPLE_N    = 10      # Number of files to process in sample mode

# ─── Logging setup ────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ─── Territorios y catálogo de estaciones desde config.yaml ──────────────────
# Lista de unidades territoriales con sus centroides WGS84.
TERRITORIOS = [
    {"id": t.id, "nombre": t.nombre, "lat": t.lat, "lon": t.lon}
    for t in cfg.territorios
]

# Catálogo estación_id → coordenadas (int keys).
STATION_CATALOG = cfg.catalogo_estaciones


# ─── Haversine distance ────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    """Return great-circle distance in km between two WGS84 points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def nearest_territory(lat, lon):
    """Return the territory name whose centroid is closest to (lat, lon)."""
    best_name = TERRITORIOS[0]["nombre"]
    best_dist = float("inf")
    for t in TERRITORIOS:
        d = haversine_km(lat, lon, t["lat"], t["lon"])
        if d < best_dist:
            best_dist = d
            best_name = t["nombre"]
    return best_name


def hash_territory(station_id: int) -> str:
    """Deterministic pseudo-assignment for stations not in the catalog."""
    h = int(hashlib.md5(str(station_id).encode()).hexdigest(), 16)
    idx = h % len(TERRITORIOS)
    return TERRITORIOS[idx]["nombre"]


def get_territory(station_id: int) -> str:
    if station_id in STATION_CATALOG:
        coords = STATION_CATALOG[station_id]
        return nearest_territory(coords["lat"], coords["lon"])
    # Deterministic fallback — flag for manual review
    return hash_territory(station_id) + " [HASH]"


# ─── Station ID extraction from filename ─────────────────────────────────────
_FNAME_RE = re.compile(r"Estacion_pluviometrica_(\d+)_")

def station_id_from_path(p: Path) -> int:
    m = _FNAME_RE.search(p.name)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot extract station ID from {p.name}")


# ─── Main pipeline ─────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    log.info("=" * 65)
    log.info("SIATA Pluviometric Processor — START")
    log.info(f"  Ciudad              : {cfg.ciudad.nombre}")
    log.info(f"  Unidad territorial  : {UT}")
    log.info(f"  CRS                 : {cfg.crs}")
    log.info(f"  Territorios cargados: {len(TERRITORIOS)}")
    log.info(f"  Root dir            : {ROOT_DIR}")
    log.info(f"  Data dir            : {DATA_DIR}")
    log.info(f"  SAMPLE_MODE         : {SAMPLE_MODE}")
    log.info(f"  SAMPLE_N            : {SAMPLE_N if SAMPLE_MODE else 'N/A (full run)'}")
    log.info("=" * 65)

    # 1. Discover files
    all_files = sorted(DATA_DIR.glob("Estacion_pluviometrica_*.csv"))
    log.info(f"Found {len(all_files)} CSV files in {DATA_DIR}")

    if SAMPLE_MODE:
        files = all_files[:SAMPLE_N]
        log.info(f"SAMPLE_MODE active — processing first {len(files)} files")
    else:
        files = all_files
        log.info("Full mode — processing all files")

    for f in files:
        log.info(f"  → {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")

    # 2. Read files with dask (lazy)
    log.info("Loading files with dask.dataframe (lazy reading)…")
    try:
        import dask.dataframe as dd
        USE_DASK = True
    except ImportError:
        log.warning("dask not available — falling back to pandas (slower)")
        USE_DASK = False

    records = []

    if USE_DASK:
        try:
            ddf = dd.read_csv(
                [str(f) for f in files],
                dtype={
                    "codigo":    "int32",
                    "p1":        "float32",
                    "p2":        "float32",
                    "calidad":   "int8",
                    "fecha_hora":"object",
                },
                assume_missing=True,
            )
            log.info(f"Dask graph created — {ddf.npartitions} partitions")

            # 3. Filter valid records
            ddf = ddf[ddf["calidad"] == 1]

            # 4. Compute (triggers actual reading)
            log.info("Computing dask graph (this may take a moment)…")
            df = ddf.compute()
            log.info(f"Loaded {len(df):,} valid records after quality filter")

        except Exception as exc:
            log.error(f"Dask read failed: {exc} — falling back to pandas")
            USE_DASK = False

    if not USE_DASK:
        dfs = []
        for f in files:
            try:
                tmp = pd.read_csv(
                    f,
                    dtype={"codigo": "int32", "p1": "float32",
                           "p2": "float32", "calidad": "int8"},
                )
                tmp = tmp[tmp["calidad"] == 1]
                dfs.append(tmp)
                log.info(f"  pandas read: {f.name} → {len(tmp):,} rows")
            except Exception as e:
                log.warning(f"  SKIP {f.name}: {e}")
        df = pd.concat(dfs, ignore_index=True)
        log.info(f"Total rows (all files, quality==1): {len(df):,}")

    # 5. Parse year
    log.info("Parsing year from fecha_hora…")
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"], errors="coerce")
    df = df.dropna(subset=["fecha_hora"])
    df["año"] = df["fecha_hora"].dt.year
    log.info(f"Year range: {df['año'].min()} – {df['año'].max()}")

    # 6. Assign territorial unit
    log.info(f"Assigning {UT}s to stations…")
    station_ids = df["codigo"].unique()
    log.info(f"  Unique stations in sample: {sorted(station_ids.tolist())}")
    territory_map = {sid: get_territory(int(sid)) for sid in station_ids}
    for sid, ter in sorted(territory_map.items()):
        src = "catalog" if int(sid) in STATION_CATALOG else "hash"
        log.info(f"    Station {sid:4d} → {ter}  [{src}]")
    df[UT] = df["codigo"].map(territory_map)

    # 7. Aggregate
    log.info(f"Aggregating per (estacion_id, año, {UT})…")
    grp = df.groupby(["codigo", "año", UT], observed=True)

    agg = grp["p1"].agg(
        precip_media_mm="mean",
        precip_max_mm="max",
        n_registros="count",
    ).reset_index()

    # 95th percentile (not directly in agg shorthand)
    p95 = grp["p1"].quantile(0.95).reset_index().rename(
        columns={"p1": "precip_p95_mm"}
    )
    result = agg.merge(p95, on=["codigo", "año", UT])
    result = result.rename(columns={"codigo": "estacion_id"})

    # Round floats
    for col in ["precip_media_mm", "precip_max_mm", "precip_p95_mm"]:
        result[col] = result[col].round(4)

    result = result.sort_values(["estacion_id", "año", UT])

    log.info(f"Aggregation complete — {len(result)} rows")

    # 8. Save
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_FILE, index=False, encoding="utf-8")
    log.info(f"Output written to: {OUT_FILE}")

    elapsed = time.time() - t0
    log.info("=" * 65)
    log.info(f"Pipeline COMPLETE in {elapsed:.1f}s")
    if SAMPLE_MODE:
        log.info(
            "NOTE: SAMPLE_MODE=True — only 10 of 184 files processed. "
            "Set SAMPLE_MODE=False and re-run for full ~20 GB dataset."
        )
    log.info("=" * 65)

    # Print first 20 rows to stdout for reporting
    print("\n── siata_agregado.csv — first 20 rows ──")
    print(result.head(20).to_string(index=False))
    print(f"\nTotal rows: {len(result)}")
    return result


if __name__ == "__main__":
    main()
