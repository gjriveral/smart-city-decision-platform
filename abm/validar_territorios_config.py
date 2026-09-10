# © 2026 Gabriel Jaime Rivera León. Todos los derechos reservados.
# El diseño arquitectónico, modelado lógico, simulaciones y estructura de este
# software son obra original del autor. La lógica computacional implementada
# comprende pipelines ETL, índices de vulnerabilidad, ensamble XGBoost con
# validación temporal y simulación basada en agentes (ABM).

"""
validar_territorios_config.py
=============================
Verifica que los nombres de unidad territorial en indices_comunas.csv
coincidan EXACTAMENTE con los declarados en config.yaml (sección
territorios), sin normalización silenciosa.

Uso:
    py abm/validar_territorios_config.py

También se invoca desde abm/04_urban_model.py::main() antes de simular.
Si hay discrepancias, aborta con código de salida 1 y un mensaje claro.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config_loader import get_config


class TerritoryNameMismatchError(ValueError):
    """Nombres de territorio en CSV ≠ nombres en config.yaml."""


def validate_territory_names(
    csv_path: str | Path | None = None,
    raise_on_error: bool = True,
) -> dict:
    """
    Compara sets de nombres (exactos, case-sensitive).

    Returns
    -------
    dict con keys: ok, only_in_config, only_in_csv, n_config, n_csv
    """
    cfg = get_config()
    _abs = cfg.rutas_absolutas()
    ut_col = cfg.unidad_territorial.title()

    if csv_path is None:
        csv_path = Path(_abs.datos_sociales) / "processed" / "indices_comunas.csv"
    else:
        csv_path = Path(csv_path)

    if not csv_path.is_file():
        msg = (
            f"No se encontró indices_comunas.csv en:\n  {csv_path}\n"
            f"Ejecuta primero ml/02_vulnerability_indices.py."
        )
        if raise_on_error:
            raise FileNotFoundError(msg)
        print(f"ERROR: {msg}", file=sys.stderr)
        return {
            "ok": False,
            "only_in_config": set(),
            "only_in_csv": set(),
            "n_config": len(cfg.territorios),
            "n_csv": 0,
            "error": msg,
        }

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    # Aceptar columna canónica o legacy "Comuna"
    col = None
    for candidate in [ut_col, "Comuna", "Localidad", "Corregimiento"]:
        if candidate in df.columns:
            col = candidate
            break
    if col is None:
        msg = (
            f"El CSV {csv_path} no tiene columna de territorio "
            f"(se buscó '{ut_col}' / Comuna / Localidad)."
        )
        if raise_on_error:
            raise KeyError(msg)
        print(f"ERROR: {msg}", file=sys.stderr)
        return {
            "ok": False,
            "only_in_config": set(),
            "only_in_csv": set(),
            "n_config": len(cfg.territorios),
            "n_csv": 0,
            "error": msg,
        }

    names_cfg = {t.nombre for t in cfg.territorios}
    names_csv = {str(x).strip() for x in df[col].dropna().unique()}

    only_cfg = sorted(names_cfg - names_csv)
    only_csv = sorted(names_csv - names_cfg)
    ok = len(only_cfg) == 0 and len(only_csv) == 0

    result = {
        "ok": ok,
        "only_in_config": only_cfg,
        "only_in_csv": only_csv,
        "n_config": len(names_cfg),
        "n_csv": len(names_csv),
        "csv_path": str(csv_path),
        "ut_col": col,
    }

    if ok:
        print(
            f"OK: {len(names_cfg)} territorios coinciden exactamente "
            f"entre config.yaml y {csv_path.name} (columna '{col}')."
        )
        return result

    lines = [
        "DISCREPANCIA DE NOMBRES DE TERRITORIO",
        f"  config.yaml  → {len(names_cfg)} nombres",
        f"  {csv_path.name} → {len(names_csv)} nombres (columna '{col}')",
        "",
        "Los nombres deben coincidir EXACTAMENTE (sin normalización).",
        "Corrige config.yaml o el ETL antes de correr la simulación ABM.",
        "",
    ]
    if only_cfg:
        lines.append(f"Solo en config.yaml ({len(only_cfg)}):")
        for n in only_cfg:
            lines.append(f"  - {n!r}")
        lines.append("")
    if only_csv:
        lines.append(f"Solo en {csv_path.name} ({len(only_csv)}):")
        for n in only_csv:
            lines.append(f"  - {n!r}")
        lines.append("")

    msg = "\n".join(lines)
    if raise_on_error:
        raise TerritoryNameMismatchError(msg)
    print(msg, file=sys.stderr)
    return result


def main() -> int:
    try:
        validate_territory_names(raise_on_error=True)
        return 0
    except (TerritoryNameMismatchError, FileNotFoundError, KeyError) as e:
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
