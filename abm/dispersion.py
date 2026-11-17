# © 2026 Gabriel Jaime Rivera León. Todos los derechos reservados.

"""Dispersion of the composite vulnerability index (IVC) across territories."""

from __future__ import annotations

import numpy as np


def coefficient_of_variation(values, ddof: int = 0) -> float:
    """CV = std / mean. Population std (ddof=0) matches the published cv_t0."""
    v = np.asarray(list(values), dtype=float)
    v = v[np.isfinite(v)]
    mean = float(np.mean(v)) if len(v) else float("nan")
    if not np.isfinite(mean) or mean == 0.0:
        return float("nan")
    return float(np.std(v, ddof=ddof) / mean)


def gini(values) -> float:
    """Gini used by the coefficient sweep (sorted cumulative formula)."""
    v = np.sort(np.asarray(list(values), dtype=float))
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0 or float(v.sum()) == 0.0:
        return 0.0
    cum = np.cumsum(v)
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)
