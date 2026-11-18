# © 2026 Gabriel Jaime Rivera León. Todos los derechos reservados.

"""
Annual rainfall by territory from SIATA minute files.

Full mode (local, about 20 GB of minute files; not run in this repository):

    python -m bigdata.siata_territory_annual --mode full \\
        --rain-dir pluviometrica \\
        --assignment data/external/asignacion_estacion_UT_oficial.json \\
        --catalog data/external/pluvio_siata_catalogo.json \\
        --out data/processed/siata_territorio_anual.csv

The command writes one row per territory-year:

    unidad_territorial, anio, precip_anual_mm, dias_ge_50mm, max_24h_mm,
    max_horario_mm, n_estaciones, metodo_agregacion, estaciones_ids,
    cobertura_horaria_media, idw_power, fuente

Where a station belongs
    1. asignacion_estacion_UT_oficial.json wins when the station is listed.
    2. Otherwise the catalog coordinate is placed with point-in-polygon
       against data/geo/comunas.geojson. This is how station 352 (Santo
       Domingo Savio, catalog comuna "01 Popular") enters Popular: it is
       not in the official file.
    3. Otherwise the catalog comuna text is matched ("01 Popular" → Popular).
    4. Otherwise the station is printed and left out of both averages and IDW.

Coverage
    A station-year with hourly coverage below 80% is excluded from the
    modelling columns. Its gaps are not filled with 0 mm. Years 2012 and
    2026 are flagged anio_parcial and are also excluded: 2012 is the
    network starting, and the 2026 files end in February. Eligible
    station-years are scaled by 1/coverage (missing-at-random), then
    averaged inside the territory or interpolated. The observed sum stays
    in precip_anual_mm for audit.

Aggregation
    Eligible stations inside a territory are averaged (estaciones_interior).
    A territory with none is filled by inverse-distance weighting (power 2)
    from eligible stations that have coordinates. Rainfall is then either
    the climate subindex inside IVC or a lagged predictor, according to
    clima.uso, and never both.

Station metrics, from calidad==1 and p1>=0 minute rows:
    precip_anual_mm   sum of valid minutes in the calendar year. This is an
                      observed sum, not a gap-filled annual total. A station
                      that recorded 30% of the year will show about 30% of
                      the annual rainfall. Medellín's climatology is about
                      1,500–2,500 mm/yr; totals near 300 mm are partial years.
    precip_anual_ajustada_mm
                      sum / coverage for station-years with coverage >= 0.70,
                      then averaged inside the territory or IDW'd. NaN when
                      no station clears that bar. Maxima are not scaled.
    dias_ge_50mm      calendar days whose valid minutes sum to at least 50 mm
    max_horario_mm    maximum clock-hour sum (observed; low coverage misses storms)
    max_24h_mm        maximum sum of 24 consecutive clock hours, with hours
                      that have no valid row treated as 0 mm inside the
                      station-year span (an outage is indistinguishable from
                      a dry hour in these files; see cobertura_horaria)

A station with neither an official territory nor catalog coordinates is
printed and left out of both the interior mean and IDW. Re-run the 20 GB
files after a processor change with the command above; the adjusted columns
are then built at station level (`ajuste=estacion`). A table that was
already aggregated can be scaled with `annotate_observed_table`
(`ajuste=territorio`).

Fixture mode runs the same code on data/fixtures/siata and labels fuente=fixture.
Those rows are a schema check. They are refused as model input.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_loader import get_config

HOURS_IN_NONLEAP_YEAR = 8760.0
# Station-years below this share of the year are kept in the observed sum
# and left out of the coverage-adjusted annual total. A two-month file
# summed as if it were a year is what made 2012 look like ~300 mm.
# Station-years below this share of hours are excluded from the modelling
# series. They are not scaled up and their gaps are not treated as dry hours.
MIN_COVERAGE = 0.80
# A day counts toward "days ≥ 50 mm" only when this many clock hours were
# actually observed. A 24-hour maximum is computed only for windows with at
# least MIN_HOURS_IN_24H observed hours. Missing hours are omitted, not
# inserted as 0 mm.
MIN_HOURS_IN_DAY = 18
MIN_HOURS_IN_24H = 20
# 2012 is the network coming online; 2026 files end in February.
EXCLUDED_MODEL_YEARS = frozenset({2012, 2026})
RAIN_FEATURE_COLS = ("precip_anual_mm", "dias_ge_50mm", "max_24h_mm", "max_horario_mm")


def hours_in_year(year: int) -> int:
    """Hours in a calendar year. 2024 is a leap year (8784); 8760 under-counts it."""
    year = int(year)
    leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    return 8784 if leap else 8760


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(min(1.0, a)))


def load_assignment(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): str(v).strip() for k, v in raw.items()}


def load_catalog(path: Path) -> dict[int, tuple[float, float]]:
    """Coordinates only, for callers that do not need the comuna text."""
    return {
        code: (rec["lat"], rec["lon"])
        for code, rec in load_catalog_records(path).items()
        if rec["lat"] is not None
    }


def load_catalog_records(path: Path) -> dict[int, dict]:
    """Station code → latitude, longitude, and the catalog's comuna text."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw["estaciones"] if isinstance(raw, dict) else raw
    out: dict[int, dict] = {}
    for row in rows:
        try:
            code = int(row["codigo"])
        except (KeyError, TypeError, ValueError):
            continue
        lat = lon = None
        try:
            lat_v = float(row.get("latitud"))
            lon_v = float(row.get("longitud"))
            if np.isfinite(lat_v) and np.isfinite(lon_v):
                lat, lon = lat_v, lon_v
        except (TypeError, ValueError):
            lat = lon = None
        out[code] = {
            "lat": lat,
            "lon": lon,
            "comuna": str(row.get("comuna") or "").strip(),
            "nombre": str(row.get("nombre") or "").strip(),
        }
    return out


def _fold_name(text: str) -> str:
    import unicodedata
    normalized = unicodedata.normalize("NFD", str(text))
    stripped = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    return " ".join(stripped.casefold().split())


def match_comuna_text(text: str, territory_names: list[str]) -> str | None:
    """Map a catalog string such as '01 Popular' onto a config territory name."""
    import re
    raw = str(text or "").strip()
    if not raw:
        return None
    raw = re.sub(r"^\d+\s*[-.]?\s*", "", raw)
    raw = re.sub(r"^comuna\s+\d+\s*", "", raw, flags=re.IGNORECASE)
    folded = _fold_name(raw)
    by_fold = {_fold_name(name): name for name in territory_names}
    return by_fold.get(folded)


def load_territory_polygons(path: Path) -> list[dict]:
    """Polygons from comunas.geojson. Rings are [lon, lat]. No holes in this file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    polygons = []
    for feature in raw.get("features", []):
        props = feature.get("properties") or {}
        name = str(props.get("nombre") or props.get("Comuna") or "").strip()
        geom = feature.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates") or []
        if gtype == "Polygon":
            rings = [coords[0]] if coords else []
        elif gtype == "MultiPolygon":
            rings = [poly[0] for poly in coords if poly]
        else:
            rings = []
        if name and rings:
            polygons.append({"nombre": name, "rings": rings})
    return polygons


def _point_in_ring(lon: float, lat: float, ring: list) -> bool:
    """Even-odd ray cast. `ring` is a sequence of [lon, lat]."""
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = float(ring[i][0]), float(ring[i][1])
        xj, yj = float(ring[j][0]), float(ring[j][1])
        intersects = (yi > lat) != (yj > lat)
        if intersects:
            x_cross = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < x_cross:
                inside = not inside
        j = i
    return inside


def territory_containing_point(lon: float, lat: float, polygons: list[dict]) -> str | None:
    for poly in polygons:
        for ring in poly["rings"]:
            if _point_in_ring(lon, lat, ring):
                return poly["nombre"]
    return None


def resolve_station_territory(
    station_id: int,
    official: dict[int, str],
    catalog: dict[int, dict],
    polygons: list[dict],
    territory_names: list[str],
) -> tuple[str, str]:
    """
    Where a station sits.

    1. The official assignment wins when the station is listed.
    2. Otherwise the catalog coordinate is tested against comunas.geojson
       (point in polygon).
    3. Otherwise the catalog's comuna text is matched to a territory name
       ('01 Popular' → Popular).
    4. Otherwise the station has no territory and is not an interior gauge
       or an IDW donor.
    """
    if station_id in official:
        return official[station_id], "oficial"
    rec = catalog.get(station_id) or {}
    lat, lon = rec.get("lat"), rec.get("lon")
    if lat is not None and lon is not None and polygons:
        hit = territory_containing_point(float(lon), float(lat), polygons)
        if hit:
            return hit, "punto_en_poligono"
    named = match_comuna_text(rec.get("comuna", ""), territory_names)
    if named:
        return named, "comuna_catalogo"
    return "", "sin_ubicacion"


def station_id_from_name(path: Path) -> int:
    # Estacion_pluviometrica_<id>_...
    parts = path.name.split("_")
    for part in parts:
        if part.isdigit():
            return int(part)
    raise ValueError(f"Cannot read a station id from {path.name}")


def metrics_from_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    """hourly columns: anio, hora (Timestamp), p1 (mm)."""
    rows = []
    if hourly.empty:
        return pd.DataFrame(columns=["anio", *RAIN_FEATURE_COLS, "n_horas_validas", "cobertura_horaria"])
    for year, g in hourly.groupby("anio"):
        g = g.sort_values("hora")
        observed_hours = g.groupby("hora")["p1"].sum()
        observed_hours.index = pd.to_datetime(observed_hours.index)
        # Gaps stay missing. They are not written in as 0 mm, which would
        # treat an outage as a dry hour and understate daily and 24-hour maxima.
        start = pd.Timestamp(observed_hours.index.min()).floor("h")
        end = pd.Timestamp(observed_hours.index.max()).floor("h")
        full_index = pd.date_range(start, end, freq="h")
        series = observed_hours.reindex(full_index)
        daily_sum = series.groupby(series.index.floor("D")).sum(min_count=1)
        daily_n = series.groupby(series.index.floor("D")).count()
        days = (daily_sum >= 50.0) & (daily_n >= MIN_HOURS_IN_DAY)
        window = series.rolling(24, min_periods=MIN_HOURS_IN_24H).sum()
        n_hours = int(observed_hours.shape[0])
        denom = float(hours_in_year(int(year)))
        observed = float(observed_hours.sum())
        coverage = float(n_hours / denom)
        max_24 = float(window.max()) if window.notna().any() else float("nan")
        rows.append({
            "anio": int(year),
            "precip_anual_mm": observed,
            "dias_ge_50mm": float(days.sum()),
            "max_24h_mm": max_24,
            "max_horario_mm": float(observed_hours.max()) if len(observed_hours) else float("nan"),
            "n_horas_validas": n_hours,
            "cobertura_horaria": coverage,
        })
    return pd.DataFrame(rows)


def read_station_file(path: Path) -> pd.DataFrame:
    """Stream a minute file down to one row per year. Memory stays on the hourly series."""
    hourly_parts = []
    header = pd.read_csv(path, nrows=0)
    cols = {c.strip(): c for c in header.columns}
    required = ["fecha_hora", "p1"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"{path.name} is missing columns {missing}")
    usecols = [cols[c] for c in ("codigo", "fecha_hora", "p1", "calidad") if c in cols]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=400_000):
        chunk.columns = [c.strip() for c in chunk.columns]
        if "calidad" in chunk.columns:
            chunk = chunk[chunk["calidad"] == 1]
        chunk["p1"] = pd.to_numeric(chunk["p1"], errors="coerce")
        chunk = chunk[chunk["p1"].notna() & (chunk["p1"] >= 0)]
        chunk["fecha_hora"] = pd.to_datetime(chunk["fecha_hora"], errors="coerce")
        chunk = chunk.dropna(subset=["fecha_hora"])
        if chunk.empty:
            continue
        chunk["hora"] = chunk["fecha_hora"].dt.floor("h")
        chunk["anio"] = chunk["hora"].dt.year
        hourly_parts.append(chunk.groupby(["anio", "hora"], as_index=False)["p1"].sum())
    if not hourly_parts:
        return metrics_from_hourly(pd.DataFrame(columns=["anio", "hora", "p1"]))
    hourly = pd.concat(hourly_parts, ignore_index=True)
    hourly = hourly.groupby(["anio", "hora"], as_index=False)["p1"].sum()
    return metrics_from_hourly(hourly)


def idw_value(lat: float, lon: float, donors: list[tuple[float, float, float]], power: float) -> float:
    weights = []
    values = []
    for dlat, dlon, value in donors:
        if not np.isfinite(value):
            continue
        dist = haversine_km(lat, lon, dlat, dlon)
        if dist < 1e-6:
            return float(value)
        weights.append(1.0 / dist ** power)
        values.append(value)
    if not weights:
        return float("nan")
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    return float(np.sum(w * v) / np.sum(w))


def _coverage_divisor(coverage: pd.Series) -> pd.Series:
    """Cap at 1 so a leap-year file scored against 8760 does not shrink the total."""
    return coverage.astype(float).clip(upper=1.0).replace(0, np.nan)


def annotate_observed_table(df: pd.DataFrame, min_coverage: float = MIN_COVERAGE) -> pd.DataFrame:
    """
    Coverage adjustment for a table that is already aggregated.

    `precip_anual_mm` stays the observed sum (what a local full run writes).
    `precip_anual_ajustada_mm` divides that sum by mean coverage when coverage
    is at least `min_coverage`. For a one-station territory this is the
    station-level annualization. For a multi-station mean or an IDW field it
    scales the aggregate by the mean coverage, which matches the mean of the
    station ratios only when those coverages are similar. The column `ajuste`
    is `territorio` here and `estacion` when `aggregate_territories` built the
    adjusted value from eligible stations before averaging.
    """
    out = df.copy()
    coverage = out["cobertura_horaria_media"].astype(float)
    divisor = _coverage_divisor(coverage)
    partial_year = out["anio"].astype(int).isin(EXCLUDED_MODEL_YEARS)
    covered = coverage >= min_coverage
    eligible = covered & ~partial_year
    out["precip_anual_ajustada_mm"] = np.where(
        eligible, out["precip_anual_mm"].astype(float) / divisor, np.nan
    )
    out["dias_ge_50mm_ajustada"] = np.where(
        eligible, out["dias_ge_50mm"].astype(float) / divisor, np.nan
    )
    motivo = np.where(
        partial_year,
        "anio_parcial",
        np.where(covered, "", "cobertura_baja"),
    )
    out["elegible"] = eligible.to_numpy()
    out["motivo_exclusion"] = motivo
    out["ajuste"] = "territorio"
    out["cobertura_minima"] = float(min_coverage)
    return out


def _adjusted_block(frame: pd.DataFrame) -> dict:
    """Mean of station-level annualizations. Empty frame → NaN and not eligible."""
    if frame.empty:
        return {
            "precip_anual_ajustada_mm": float("nan"),
            "dias_ge_50mm_ajustada": float("nan"),
            "elegible": False,
            "n_estaciones_elegibles": 0,
            "metodo_ajustado": "cobertura_insuficiente",
        }
    divisor = _coverage_divisor(frame["cobertura_horaria"])
    return {
        "precip_anual_ajustada_mm": float((frame["precip_anual_mm"] / divisor).mean()),
        "dias_ge_50mm_ajustada": float((frame["dias_ge_50mm"] / divisor).mean()),
        "elegible": True,
        "n_estaciones_elegibles": int(len(frame)),
        "metodo_ajustado": "estaciones_interior",
    }


def _idw_adjusted(donors, lat: float, lon: float, power: float) -> dict:
    eligible = [d for d in donors if float(d[3].cobertura_horaria) >= MIN_COVERAGE]
    if not eligible:
        return {
            "precip_anual_ajustada_mm": float("nan"),
            "dias_ge_50mm_ajustada": float("nan"),
            "elegible": False,
            "n_estaciones_elegibles": 0,
            "metodo_ajustado": "cobertura_insuficiente",
            "ajuste": "estacion",
            "cobertura_minima": float(MIN_COVERAGE),
        }

    def _pack(column: str):
        packed = []
        for _sid, dlat, dlon, rec in eligible:
            divisor = min(1.0, float(rec.cobertura_horaria))
            packed.append((dlat, dlon, float(getattr(rec, column)) / divisor))
        return packed

    return {
        "precip_anual_ajustada_mm": idw_value(lat, lon, _pack("precip_anual_mm"), power),
        "dias_ge_50mm_ajustada": idw_value(lat, lon, _pack("dias_ge_50mm"), power),
        "elegible": True,
        "n_estaciones_elegibles": int(len(eligible)),
        "metodo_ajustado": "idw",
        "ajuste": "estacion",
        "cobertura_minima": float(MIN_COVERAGE),
    }


def _apply_model_year_policy(row: dict, year: int) -> dict:
    """2012 and 2026 stay in the file for audit and are dropped from models."""
    if int(year) in EXCLUDED_MODEL_YEARS:
        row["precip_anual_ajustada_mm"] = float("nan")
        row["dias_ge_50mm_ajustada"] = float("nan")
        row["elegible"] = False
        row["motivo_exclusion"] = "anio_parcial"
    else:
        row["motivo_exclusion"] = "" if row.get("elegible") else "cobertura_baja"
    return row


def aggregate_territories(
    station_year: pd.DataFrame,
    assignment: dict[int, str],
    catalog: dict[int, tuple[float, float]],
    territories: list[dict],
    idw_power: float,
    fuente: str,
) -> pd.DataFrame:
    known = {t["nombre"] for t in territories}
    unknown = sorted({name for name in assignment.values() if name not in known})
    if unknown:
        raise ValueError(f"Assignment names that are not in config.yaml: {unknown}")

    station_year = station_year.copy()
    # unidad_territorial_oficial already applies the official list, then the
    # polygon, then the catalog comuna text. Mapping the official file again
    # would drop gauges such as station 352 in Popular.
    if "unidad_territorial_oficial" in station_year.columns:
        station_year["unidad_oficial"] = station_year["unidad_territorial_oficial"].fillna("").astype(str)
    else:
        station_year["unidad_oficial"] = station_year["estacion_id"].map(assignment)
    years = sorted(int(y) for y in station_year["anio"].unique())
    rows = []
    for year in years:
        year_df = station_year[station_year["anio"] == year]
        donors = []
        for rec in year_df.itertuples(index=False):
            coords = catalog.get(int(rec.estacion_id))
            if coords is None:
                continue
            donors.append((int(rec.estacion_id), coords[0], coords[1], rec))
        for territory in territories:
            name = territory["nombre"]
            inside = year_df[year_df["unidad_oficial"] == name]
            if len(inside):
                ids = sorted(int(v) for v in inside["estacion_id"].tolist())
                eligible = inside[inside["cobertura_horaria"] >= MIN_COVERAGE]
                if len(eligible):
                    adjusted = _adjusted_block(eligible)
                    adjusted["ajuste"] = "estacion"
                    adjusted["cobertura_minima"] = float(MIN_COVERAGE)
                else:
                    # The assigned gauges exist but none cover enough of the year.
                    # The adjusted total then uses eligible neighbours, and the
                    # observed sum below stays the partial interior sum.
                    adjusted = _idw_adjusted(donors, territory["lat"], territory["lon"], idw_power)
                rows.append({
                    "unidad_territorial": name,
                    "anio": year,
                    "precip_anual_mm": float(inside["precip_anual_mm"].mean()),
                    "dias_ge_50mm": float(inside["dias_ge_50mm"].mean()),
                    "max_24h_mm": float(inside["max_24h_mm"].mean()),
                    "max_horario_mm": float(inside["max_horario_mm"].mean()),
                    "n_estaciones": int(len(inside)),
                    "metodo_agregacion": "estaciones_interior",
                    "estaciones_ids": ",".join(str(i) for i in ids),
                    "cobertura_horaria_media": float(inside["cobertura_horaria"].mean()),
                    "idw_power": np.nan,
                    "fuente": fuente,
                    **adjusted,
                })
                rows[-1] = _apply_model_year_policy(rows[-1], year)
                continue
            donor_pack = {
                col: [(lat, lon, float(getattr(rec, col))) for _sid, lat, lon, rec in donors]
                for col in RAIN_FEATURE_COLS
            }
            if not donors:
                metrics = {col: float("nan") for col in RAIN_FEATURE_COLS}
                method = "sin_datos"
                ids = ""
                n_est = 0
                cover = float("nan")
                power_out = np.nan
            else:
                metrics = {
                    col: idw_value(territory["lat"], territory["lon"], donor_pack[col], idw_power)
                    for col in RAIN_FEATURE_COLS
                }
                method = "idw"
                ids = ",".join(str(sid) for sid, *_ in donors)
                n_est = len(donors)
                cover = float(np.mean([rec.cobertura_horaria for *_, rec in donors]))
                power_out = idw_power
            adjusted = _idw_adjusted(donors, territory["lat"], territory["lon"], idw_power)
            rows.append({
                "unidad_territorial": name,
                "anio": year,
                **metrics,
                "n_estaciones": n_est,
                "metodo_agregacion": method,
                "estaciones_ids": ids,
                "cobertura_horaria_media": cover,
                "idw_power": power_out,
                "fuente": fuente,
                **adjusted,
            })
            rows[-1] = _apply_model_year_policy(rows[-1], year)
    return pd.DataFrame(rows)


def assignment_coverage(assignment: dict[int, str], territories: list[dict]) -> dict:
    counts: dict[str, int] = {t["nombre"]: 0 for t in territories}
    for name in assignment.values():
        if name in counts:
            counts[name] += 1
    empty = [name for name, n in counts.items() if n == 0]
    return {
        "n_estaciones_asignadas": len(assignment),
        "territorios_con_estacion": [n for n, c in counts.items() if c > 0],
        "territorios_sin_estacion_idw": empty,
        "conteo": counts,
    }


def process(rain_dir: Path, assignment_path: Path, catalog_path: Path, fuente: str, idw_power: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = get_config()
    territories = [
        {"nombre": t.nombre, "lat": t.lat, "lon": t.lon} for t in cfg.territorios
    ]
    assignment = load_assignment(assignment_path)
    catalog = load_catalog(catalog_path)
    files = sorted(rain_dir.glob("Estacion_pluviometrica_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No Estacion_pluviometrica_*.csv files in {rain_dir}. "
            "Full mode expects the SIATA minute files (about 20 GB) in that directory."
        )
    catalog_records = load_catalog_records(catalog_path)
    geo_path = Path(cfg.rutas_absolutas().geo) / "comunas.geojson"
    polygons = load_territory_polygons(geo_path) if geo_path.exists() else []
    territory_names = [t["nombre"] for t in territories]
    station_ids = {station_id_from_name(path) for path in files}
    station_ids |= set(assignment) | set(catalog_records)
    resolved = {
        sid: resolve_station_territory(
            sid, assignment, catalog_records, polygons, territory_names
        )
        for sid in station_ids
    }
    from collections import Counter
    print("Station placement:", dict(Counter(src for _name, src in resolved.values())))
    station_rows = []
    skipped_unplaced = []
    for path in files:
        sid = station_id_from_name(path)
        territory_name, source = resolved.get(sid, ("", "sin_ubicacion"))
        if source == "sin_ubicacion":
            skipped_unplaced.append(sid)
        metrics = read_station_file(path)
        if metrics.empty:
            continue
        metrics.insert(0, "estacion_id", sid)
        rec = catalog_records.get(sid) or {}
        metrics["lat"] = rec.get("lat") if rec.get("lat") is not None else np.nan
        metrics["lon"] = rec.get("lon") if rec.get("lon") is not None else np.nan
        metrics["unidad_territorial_oficial"] = territory_name
        metrics["fuente_asignacion"] = source
        station_rows.append(metrics)
    if not station_rows:
        raise RuntimeError(f"No valid rainfall rows were read from {rain_dir}")
    if skipped_unplaced:
        print(
            "Stations with neither catalog coordinates nor an official territory "
            f"({len(skipped_unplaced)}): {sorted(skipped_unplaced)}. "
            "They are omitted from interior means and from IDW."
        )
    station_year = pd.concat(station_rows, ignore_index=True)
    territory_year = aggregate_territories(
        station_year, assignment, catalog, territories, idw_power, fuente
    )
    return territory_year, station_year


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description="SIATA annual rainfall by territory")
    parser.add_argument("--mode", choices=["full", "fixture"], default="full")
    parser.add_argument("--rain-dir", type=Path, default=None)
    parser.add_argument("--assignment", type=Path, default=ROOT / "data/external/asignacion_estacion_UT_oficial.json")
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/external/pluvio_siata_catalogo.json")
    parser.add_argument("--out", type=Path, default=ROOT / "data/processed/siata_territorio_anual.csv")
    parser.add_argument("--idw-power", type=float, default=2.0)
    args = parser.parse_args(argv)

    if args.mode == "fixture":
        rain_dir = args.rain_dir or (ROOT / "data/fixtures/siata")
        fuente = "fixture"
    else:
        rain_dir = args.rain_dir or (ROOT / "pluviometrica")
        fuente = "full"
    territory, station = process(rain_dir, args.assignment, args.catalog, fuente, args.idw_power)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    territory.to_csv(args.out, index=False)
    station_out = args.out.with_name(args.out.stem + "_estaciones.csv")
    station.to_csv(station_out, index=False)
    print(f"Wrote {len(territory)} territory-year rows to {args.out}")
    print(f"Wrote {len(station)} station-year rows to {station_out}")
    print("Expected columns:", ", ".join(territory.columns))
    return territory


if __name__ == "__main__":
    main()
