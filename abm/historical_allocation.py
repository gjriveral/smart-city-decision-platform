# © 2026 Gabriel Jaime Rivera León. Todos los derechos reservados.

"""
Observed public-investment scenario.

The allocation actually executed by the city is not in this repository.
Until Gabriel supplies it, the scenario is skipped. The CSV schema is:

    unidad_territorial,anio,inversion

    unidad_territorial  Exact `territorios.nombre` from config.yaml.
    anio                Calendar year when
                        simulacion.inversion_historica.tipo_anio is calendario
                        (simulation year 1 maps to anio_inicio).
                        Or the simulation year 1..horizonte when tipo_anio
                        is simulacion.
    inversion           Allocation in the same units as
                        simulacion.presupuesto_anual. Values are used as
                        given; they are not rescaled to the annual budget.

One row per territory-year. A territory missing in a year receives 0.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("unidad_territorial", "anio", "inversion")

SCHEMA = {
    "file": "data/external/inversion_historica_por_territorio.csv",
    "columns": [
        {
            "name": "unidad_territorial",
            "type": "string",
            "description": "Exact territory name from config.yaml territorios.nombre.",
        },
        {
            "name": "anio",
            "type": "integer",
            "description": (
                "Calendar year if simulacion.inversion_historica.tipo_anio is "
                "calendario (simulation year 1 = anio_inicio). Simulation year "
                "1..horizonte if tipo_anio is simulacion."
            ),
        },
        {
            "name": "inversion",
            "type": "number",
            "description": (
                "Allocation in the same units as simulacion.presupuesto_anual. "
                "Not rescaled."
            ),
        },
    ],
    "status": "Data not yet available. The scenario is skipped until this CSV is provided.",
}


def write_schema(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(SCHEMA, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_template(path: str | Path) -> None:
    """Header-only template. Does not invent investment figures."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    pd.DataFrame(columns=list(REQUIRED_COLUMNS)).to_csv(path, index=False)


def load_historical_allocation(path: str | Path) -> tuple[dict[int, dict[str, float]], dict]:
    """
    Read the investment CSV.

    Returns
    -------
    mapping : dict[year, dict[territory, amount]]
    status  : dict with state in {missing, empty, invalid, loaded} and a message.
    """
    path = Path(path)
    if not path.exists():
        return {}, {
            "state": "missing",
            "message": f"No historical investment file at {path}. Scenario skipped.",
            "path": str(path),
            "rows": 0,
        }
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        return {}, {
            "state": "invalid",
            "message": f"Could not read {path}: {exc}",
            "path": str(path),
            "rows": 0,
        }
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return {}, {
            "state": "invalid",
            "message": f"Missing columns {missing}. Required: {list(REQUIRED_COLUMNS)}.",
            "path": str(path),
            "rows": int(len(df)),
        }
    if df.empty:
        return {}, {
            "state": "empty",
            "message": "Historical investment file has a header but no rows. Scenario skipped.",
            "path": str(path),
            "rows": 0,
        }
    out: dict[int, dict[str, float]] = {}
    for _, row in df.iterrows():
        year = int(row["anio"])
        name = str(row["unidad_territorial"]).strip()
        amount = float(row["inversion"])
        out.setdefault(year, {})[name] = amount
    return out, {
        "state": "loaded",
        "message": f"Loaded {len(df)} investment rows from {path}.",
        "path": str(path),
        "rows": int(len(df)),
        "years": sorted(out),
    }
