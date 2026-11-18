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

Aggregation
    Stations listed in the official assignment are averaged inside that
    territory (method estaciones_interior). A territory with no assigned
    station is filled by inverse-distance weighting (power 2) from every
    processed station that has catalog coordinates (method idw). Rainfall
    is then either the climate subindex inside IVC or a lagged predictor,
    according to clima.uso, and never both.

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
MIN_COVERAGE = 0.70
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
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw["estaciones"] if isinstance(raw, dict) else raw
    out = {}
    for row in rows:
        try:
            code = int(row["codigo"])
            lat = float(row["latitud"])
            lon = float(row["longitud"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(lat) and np.isfinite(lon):
            out[code] = (lat, lon)
    return out


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
        start = pd.Timestamp(g["hora"].min()).floor("h")
        end = pd.Timestamp(g["hora"].max()).floor("h")
        full_index = pd.date_range(start, end, freq="h")
        series = g.groupby("hora")["p1"].sum().reindex(full_index, fill_value=0.0)
        daily = series.groupby(series.index.floor("D")).sum()
        window = series.rolling(24, min_periods=1).sum()
        # Observed clock hours only. Gap-filled zeros are not counted as measurements.
        # The annual sum is the sum of those hours. It is not gap-filled to a
        # full year: a file that starts in November does not become an annual total.
        n_hours = int(g["hora"].nunique())
        denom = float(hours_in_year(int(year)))
        observed = float(series.sum())
        coverage = float(n_hours / denom)
        rows.append({
            "anio": int(year),
            "precip_anual_mm": observed,
            "dias_ge_50mm": float((daily >= 50.0).sum()),
            "max_24h_mm": float(window.max()) if len(window) else float("nan"),
            "max_horario_mm": float(series.max()) if len(series) else float("nan"),
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
    eligible = coverage >= min_coverage
    out["precip_anual_ajustada_mm"] = np.where(
        eligible, out["precip_anual_mm"].astype(float) / divisor, np.nan
    )
    out["dias_ge_50mm_ajustada"] = np.where(
        eligible, out["dias_ge_50mm"].astype(float) / divisor, np.nan
    )
    out["elegible"] = eligible.to_numpy()
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
    station_rows = []
    skipped_unplaced = []
    for path in files:
        sid = station_id_from_name(path)
        in_catalog = sid in catalog
        in_assignment = sid in assignment
        if not in_catalog and not in_assignment:
            # No coordinates and no official territory: the file cannot be
            # placed inside a territory or used as an IDW donor.
            skipped_unplaced.append(sid)
        metrics = read_station_file(path)
        if metrics.empty:
            continue
        metrics.insert(0, "estacion_id", sid)
        coords = catalog.get(sid)
        metrics["lat"] = coords[0] if coords else np.nan
        metrics["lon"] = coords[1] if coords else np.nan
        metrics["unidad_territorial_oficial"] = assignment.get(sid, "")
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
