# © 2026 Gabriel Jaime Rivera León. Todos los derechos reservados.

"""
Rainfall enters the analysis in exactly one place.

    ninguno    Rainfall is not used. The constant climate subindex is not a predictor.
    subindice  Rainfall defines idx_riesgo_clima and therefore IVC. Raw rainfall
               columns are not model predictors.
    predictor  Rainfall columns at year t are predictors of IVC at t+1.
               idx_riesgo_clima is not a predictor (it would double-count climate).

Fixture rainfall (fuente == fixture) is refused as a model input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ALLOWED_USES = ("ninguno", "subindice", "predictor")
SUBINDEX_COLS = (
    "idx_desempleo",
    "idx_habitat",
    "idx_educacion",
    "idx_riesgo_clima",
    "idx_pobreza",
    "idx_tejido_social",
)
CLIMATE_SUBINDEX = "idx_riesgo_clima"
DEFAULT_WEIGHTS = {
    "idx_desempleo": 1.0,
    "idx_habitat": 1.0,
    "idx_educacion": 1.0,
    "idx_riesgo_clima": 0.5,
    "idx_pobreza": 1.0,
    "idx_tejido_social": 1.0,
}


def assert_climate_use(uso: str) -> str:
    uso = str(uso).strip().lower()
    if uso not in ALLOWED_USES:
        raise ValueError(
            f"clima.uso must be one of {ALLOWED_USES} (got {uso!r}). "
            "Rainfall is either inside IVC or a predictor, never both."
        )
    return uso


def predictor_columns(uso: str, rain_columns: list[str] | None = None) -> list[str]:
    """
    Columns allowed as predictors under `uso`.

    Raises if a caller tries to treat raw rainfall as a predictor while the
    climate subindex is also in the feature list (the double-use case).
    """
    uso = assert_climate_use(uso)
    rain_columns = list(rain_columns or [])
    social = [c for c in SUBINDEX_COLS if c != CLIMATE_SUBINDEX]
    if uso == "ninguno":
        if rain_columns:
            raise ValueError(
                "clima.uso is 'ninguno' but rainfall columns were passed as predictors."
            )
        return social
    if uso == "subindice":
        if rain_columns:
            raise ValueError(
                "Rainfall cannot be a model predictor while it is also inside IVC "
                "(clima.uso='subindice'). Pass rain_columns only when uso='predictor'."
            )
        return list(SUBINDEX_COLS)
    if not rain_columns:
        raise ValueError("clima.uso='predictor' requires at least one rainfall column.")
    overlap = [c for c in rain_columns if c in SUBINDEX_COLS]
    if overlap:
        raise ValueError(
            f"Rainfall predictor columns collide with subindices: {overlap}."
        )
    return social + rain_columns


def composite_ivc(frame: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.Series:
    """IVC = weighted mean of the six subindices. Weights need not be uniform."""
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    missing = [c for c in SUBINDEX_COLS if c not in weights]
    if missing:
        raise KeyError(f"IVC weights missing for {missing}")
    cols = list(SUBINDEX_COLS)
    w = np.array([float(weights[c]) for c in cols], dtype=float)
    if np.any(w < 0) or float(w.sum()) == 0.0:
        raise ValueError(f"IVC weights must be non-negative and not all zero (got {weights}).")
    values = frame[cols].to_numpy(dtype=float)
    return pd.Series(values @ w / w.sum(), index=frame.index, name="ivc")


def fit_minmax_train_only(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Column-wise min-max fit on `train` only, then applied to train and test.

    Constant training columns map to 0. Test values outside the training
    range are not clipped, so a test-set extreme stays outside [0, 1].
    """
    train = np.asarray(train, dtype=float)
    test = np.asarray(test, dtype=float)
    col_min = np.nanmin(train, axis=0)
    col_max = np.nanmax(train, axis=0)
    span = col_max - col_min

    def _apply(arr: np.ndarray) -> np.ndarray:
        out = np.empty_like(arr, dtype=float)
        for j in range(arr.shape[1]):
            if not np.isfinite(span[j]) or span[j] == 0.0:
                out[:, j] = 0.0
            else:
                out[:, j] = (arr[:, j] - col_min[j]) / span[j]
        return out

    return _apply(train), _apply(test), {
        "min": col_min.tolist(),
        "max": col_max.tolist(),
    }


def climate_subindex_from_rain(
    rain_train: np.ndarray,
    rain_apply: np.ndarray,
) -> np.ndarray:
    """
    Train-only min-max of one rainfall metric, averaged if several columns.

    This is the subindex path: the result replaces idx_riesgo_clima and is
    not also passed as a raw predictor.
    """
    train = np.atleast_2d(np.asarray(rain_train, dtype=float))
    apply = np.atleast_2d(np.asarray(rain_apply, dtype=float))
    if train.ndim == 1:
        train = train.reshape(-1, 1)
        apply = apply.reshape(-1, 1)
    scaled_train, scaled_apply, _ = fit_minmax_train_only(train, apply)
    return scaled_apply.mean(axis=1)
