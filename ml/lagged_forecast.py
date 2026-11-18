# © 2026 Gabriel Jaime Rivera León. Todos los derechos reservados.

"""
Leak-free lagged forecast of the composite vulnerability index (IVC).

Question answered here
    Do subindices observed in year t predict IVC in year t+1?

Question explicitly not answered here
    Can a model reconstruct IVC_t from the same-year subindices that define it?
    That reconstruction is reported only as a circularity diagnostic.

Protocol
    - Pairs are (features at t, IVC at t+1) within the same territory.
    - Min-max normalization is fit on the training rows of each split only.
    - Outer evaluation is an expanding window by target year.
    - Hyperparameters are chosen on the last training target-year only,
      then the model is refit on the whole training window and scored once
      on the held-out target year.
    - The outer test year is never used to pick hyperparameters.
    - Leave-one-territory-out retunes on the other territories with the same
      temporal inner split. It is a spatial check, not a temporal one.
    - Baselines: persistence (predict IVC_t), ordinary least squares,
      random forest, XGBoost. They share the same rows.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from ml.clima_hook import (
    CLIMATE_SUBINDEX,
    SUBINDEX_COLS,
    assert_climate_use,
    fit_minmax_train_only,
    predictor_columns,
)

RANDOM_STATE = 42
FEATURE_YEAR = "anio"
TARGET_YEAR = "anio_objetivo"
TARGET = "ivc_next"
PERSISTENCE_COL = "ivc_t"

XGB_GRID = [
    {"n_estimators": n, "max_depth": d, "learning_rate": lr, "min_child_weight": w}
    for n in (100, 200)
    for d in (2, 3)
    for lr in (0.05, 0.1)
    for w in (1, 5)
]
RF_GRID = [
    {"n_estimators": 200, "max_depth": d, "min_samples_leaf": leaf}
    for d in (3, 6, None)
    for leaf in (1, 3)
]

# Same grid the legacy contemporaneous script searches. Used only to measure
# version sensitivity of that (leaky, same-year) protocol.
LEGACY_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [2, 3],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.7, 0.8],
    "colsample_bytree": [0.7, 0.8],
    "min_child_weight": [3, 5],
    "reg_alpha": [0.0, 0.1],
    "reg_lambda": [1.0, 2.0],
    "gamma": [0.0, 0.1],
}


@dataclass
class FoldSpec:
    test_year: int
    train_years: list[int]
    params: dict
    n_train: int
    n_test: int


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "n": int(len(y_true)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }


def build_lagged_frame(
    df: pd.DataFrame,
    ut_col: str = "Comuna",
    year_col: str = "Año",
    year_min: int = 2013,
    year_max: int = 2024,
    rain: pd.DataFrame | None = None,
    rain_feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    One row per territory and feature-year t whose t+1 IVC is observed.

    Rainfall, when supplied, is merged on (territory, t), never on t+1.
    """
    d = df.loc[(df[year_col] >= year_min) & (df[year_col] <= year_max)].copy()
    d[ut_col] = d[ut_col].astype(str).str.strip()
    keep = [ut_col, year_col, "ivc", *SUBINDEX_COLS]
    missing = [c for c in keep if c not in d.columns]
    if missing:
        raise KeyError(f"Panel is missing columns {missing}")
    d = d[keep].dropna(subset=["ivc", *SUBINDEX_COLS])

    left = d.rename(columns={year_col: FEATURE_YEAR, "ivc": PERSISTENCE_COL})
    right = d[[ut_col, year_col, "ivc"]].rename(
        columns={year_col: TARGET_YEAR, "ivc": TARGET}
    )
    right[FEATURE_YEAR] = right[TARGET_YEAR] - 1
    out = left.merge(right[[ut_col, FEATURE_YEAR, TARGET_YEAR, TARGET]], on=[ut_col, FEATURE_YEAR], how="inner")
    out = out.sort_values([TARGET_YEAR, ut_col]).reset_index(drop=True)

    if rain is not None:
        rain = rain.copy()
        rain_ut = "unidad_territorial" if "unidad_territorial" in rain.columns else ut_col
        rain_year = "anio" if "anio" in rain.columns else FEATURE_YEAR
        cols = list(rain_feature_cols or [])
        if not cols:
            raise ValueError("rain_feature_cols is required when a rainfall table is passed.")
        if "fuente" in rain.columns and set(rain["fuente"].astype(str)) <= {"fixture"}:
            raise RuntimeError(
                "Fixture rainfall cannot be used as a predictor or as a subindex. "
                "Run bigdata/siata_territory_annual.py --mode full on the SIATA files first."
            )
        slim = rain[[rain_ut, rain_year, *cols]].rename(
            columns={rain_ut: ut_col, rain_year: FEATURE_YEAR}
        )
        before = len(out)
        out = out.merge(slim, on=[ut_col, FEATURE_YEAR], how="inner")
        if len(out) < before:
            # Inner join drops territory-years without rainfall. That is intentional:
            # we do not impute rainfall from other years.
            pass
    return out


def _matrix(frame: pd.DataFrame, columns: list[str], ut_col: str) -> tuple[np.ndarray, np.ndarray]:
    year = frame[FEATURE_YEAR].to_numpy(dtype=float).reshape(-1, 1)
    feats = frame[columns].to_numpy(dtype=float) if columns else np.zeros((len(frame), 0))
    x = np.column_stack([feats, year]) if columns else year
    return x, frame[TARGET].to_numpy(dtype=float)


def _fit_xgb(params: dict):
    import xgboost as xgb

    return xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        verbosity=0,
        n_jobs=1,
        **params,
    )


def _fit_rf(params: dict) -> RandomForestRegressor:
    return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1, **params)


def _candidate_grid(model_name: str) -> list[dict]:
    if model_name == "xgboost":
        return XGB_GRID
    if model_name == "random_forest":
        return RF_GRID
    if model_name == "linear":
        return [{}]
    raise ValueError(model_name)


def _new_model(model_name: str, params: dict):
    if model_name == "xgboost":
        return _fit_xgb(params)
    if model_name == "random_forest":
        return _fit_rf(params)
    if model_name == "linear":
        return LinearRegression()
    raise ValueError(model_name)


def _tune(train: pd.DataFrame, columns: list[str], ut_col: str, model_name: str) -> dict:
    """Pick hyperparameters on the last training target-year. Never sees the test year."""
    years = sorted(int(y) for y in train[TARGET_YEAR].unique())
    grid = _candidate_grid(model_name)
    if model_name == "linear" or len(years) < 2:
        return grid[0]
    val_year = years[-1]
    inner_tr = train[train[TARGET_YEAR] < val_year]
    inner_va = train[train[TARGET_YEAR] == val_year]
    if inner_tr.empty or inner_va.empty:
        return grid[0]
    x_tr_raw, y_tr = _matrix(inner_tr, columns, ut_col)
    x_va_raw, y_va = _matrix(inner_va, columns, ut_col)
    x_tr, x_va, _ = fit_minmax_train_only(x_tr_raw, x_va_raw)
    best_params = grid[0]
    best_rmse = float("inf")
    for params in grid:
        model = _new_model(model_name, params)
        model.fit(x_tr, y_tr)
        pred = model.predict(x_va)
        rmse = float(np.sqrt(mean_squared_error(y_va, pred)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
    return best_params


def _predict_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    ut_col: str,
    model_name: str,
    params: dict,
) -> np.ndarray:
    x_tr_raw, y_tr = _matrix(train, columns, ut_col)
    x_te_raw, _ = _matrix(test, columns, ut_col)
    x_tr, x_te, _ = fit_minmax_train_only(x_tr_raw, x_te_raw)
    model = _new_model(model_name, params)
    model.fit(x_tr, y_tr)
    return np.asarray(model.predict(x_te), dtype=float)


def walk_forward(
    frame: pd.DataFrame,
    columns: list[str],
    ut_col: str,
    model_name: str,
    min_train_years: int = 3,
) -> tuple[pd.DataFrame, list[FoldSpec]]:
    """Expanding-window predictions for one model. Persistence ignores `columns`."""
    years = sorted(int(y) for y in frame[TARGET_YEAR].unique())
    rows: list[pd.DataFrame] = []
    specs: list[FoldSpec] = []
    for test_year in years:
        train_years = [y for y in years if y < test_year]
        if len(train_years) < min_train_years:
            continue
        train = frame[frame[TARGET_YEAR].isin(train_years)]
        test = frame[frame[TARGET_YEAR] == test_year]
        if model_name == "persistence":
            pred = test[PERSISTENCE_COL].to_numpy(dtype=float)
            params: dict = {}
        else:
            params = _tune(train, columns, ut_col, model_name)
            pred = _predict_split(train, test, columns, ut_col, model_name, params)
        piece = test[[ut_col, FEATURE_YEAR, TARGET_YEAR, PERSISTENCE_COL, TARGET]].copy()
        piece["model"] = model_name
        piece["y_pred"] = pred
        piece["params"] = json.dumps(params, default=str)
        rows.append(piece)
        specs.append(FoldSpec(
            test_year=test_year,
            train_years=train_years,
            params=params,
            n_train=int(len(train)),
            n_test=int(len(test)),
        ))
    if not rows:
        return pd.DataFrame(), specs
    return pd.concat(rows, ignore_index=True), specs


def leave_one_territory_out(
    frame: pd.DataFrame,
    columns: list[str],
    ut_col: str,
    model_name: str,
) -> pd.DataFrame:
    """
    Spatial check: train on every other territory, score the held-out one.

    Tuning uses a temporal inner split of the other territories only.
    The held-out territory's rows are never used to choose hyperparameters.
    """
    pieces = []
    territories = sorted(frame[ut_col].unique())
    for held in territories:
        train = frame[frame[ut_col] != held]
        test = frame[frame[ut_col] == held]
        if train.empty or test.empty:
            continue
        if model_name == "persistence":
            pred = test[PERSISTENCE_COL].to_numpy(dtype=float)
            params: dict = {}
        else:
            params = _tune(train, columns, ut_col, model_name)
            pred = _predict_split(train, test, columns, ut_col, model_name, params)
        piece = test[[ut_col, FEATURE_YEAR, TARGET_YEAR, TARGET]].copy()
        piece["model"] = model_name
        piece["y_pred"] = pred
        piece["held_out"] = held
        piece["params"] = json.dumps(params, default=str)
        pieces.append(piece)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def summarize_predictions(pred: pd.DataFrame, ut_col: str, by: str | None = None) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame()
    keys = ["model"] if by is None else ["model", by]
    rows = []
    for key, g in pred.groupby(keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        m = regression_metrics(g[TARGET], g["y_pred"])
        rec = dict(zip(keys, key))
        rec.update(m)
        rows.append(rec)
    return pd.DataFrame(rows)


def contemporaneous_linear_r2(df: pd.DataFrame, year_min: int = 2013, year_max: int = 2024) -> dict:
    """
    Same-year diagnostic. A linear model of the six subindices must recover IVC
    if IVC is their weighted average. This is not a forecast.
    """
    d = df.loc[(df["Año"] >= year_min) & (df["Año"] <= year_max)].dropna(subset=["ivc", *SUBINDEX_COLS])
    x = d[list(SUBINDEX_COLS)].to_numpy(dtype=float)
    y = d["ivc"].to_numpy(dtype=float)
    model = LinearRegression().fit(x, y)
    pred = model.predict(x)
    return {
        "n": int(len(d)),
        "R2_in_sample": float(r2_score(y, pred)),
        "max_abs_residual": float(np.max(np.abs(pred - y))),
        "note": "Same-year reconstruction of IVC from its own subindices. Not a forecast.",
    }


def legacy_contemporaneous_cv(df: pd.DataFrame, year_min: int = 2013, year_max: int = 2024) -> dict:
    """
    Re-run of ml/03_xgboost_model.py's outer protocol on the current install:
    same-year subindices, global year scaling, TimeSeriesSplit by row,
    GridSearchCV with a non-temporal inner K-fold.

    The published mean R² is the mean of these outer-fold scores. Selecting
    the saved model's hyperparameters from the best outer fold is a separate
    step and is reported, not used as the headline score.
    """
    import xgboost as xgb

    d = (
        df.loc[(df["Año"] >= year_min) & (df["Año"] <= year_max)]
        .dropna(subset=["ivc", *SUBINDEX_COLS])
        .sort_values(["Año", "Comuna"])
        .reset_index(drop=True)
    )
    years = d["Año"].to_numpy(dtype=float)
    # This is the leak the new pipeline removes: the denominator uses every year,
    # including years that will sit in a test fold.
    d["anio_normalizado_global"] = (years - years.min()) / max(years.max() - years.min(), 1.0)
    features = [*SUBINDEX_COLS, "anio_normalizado_global"]
    x = d[features].to_numpy(dtype=float)
    y = d["ivc"].to_numpy(dtype=float)
    n_splits = min(5, d["Año"].nunique() - 1)
    splitter = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    best_rmse = float("inf")
    best_params = None
    best_fold = None
    for fold_idx, (tr, te) in enumerate(splitter.split(x), start=1):
        inner_cv = max(2, min(3, len(tr) // 30))
        gs = GridSearchCV(
            xgb.XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                verbosity=0,
                n_jobs=1,
            ),
            LEGACY_PARAM_GRID,
            cv=inner_cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            refit=True,
        )
        gs.fit(x[tr], y[tr])
        pred = gs.predict(x[te])
        m = regression_metrics(y[te], pred)
        test_years = sorted(int(v) for v in d.iloc[te]["Año"].unique())
        train_years = sorted(int(v) for v in d.iloc[tr]["Año"].unique())
        rec = {
            "fold": fold_idx,
            "train_years": train_years,
            "test_years": test_years,
            **m,
            "best_params": gs.best_params_,
        }
        folds.append(rec)
        if m["RMSE"] < best_rmse:
            best_rmse = m["RMSE"]
            best_params = gs.best_params_
            best_fold = fold_idx
    r2s = [f["R2"] for f in folds]
    return {
        "protocol": "legacy_same_year_timeseries_split_nontemporal_inner_grid",
        "xgboost_version": xgb.__version__,
        "folds": folds,
        "mean_R2": float(np.mean(r2s)),
        "mean_RMSE": float(np.mean([f["RMSE"] for f in folds])),
        "hyperparameters_from_best_test_fold": best_params,
        "best_test_fold": best_fold,
        "note": (
            "mean_R2 averages outer test folds. Each fold's hyperparameters come "
            "from GridSearchCV with an integer cv (KFold, not a temporal inner split). "
            "hyperparameters_from_best_test_fold is chosen by comparing those outer "
            "test scores, which is how the saved model was specified."
        ),
    }


def run_lagged_suite(
    df: pd.DataFrame,
    ut_col: str = "Comuna",
    uso: str = "ninguno",
    rain: pd.DataFrame | None = None,
    rain_feature_cols: list[str] | None = None,
    year_min: int = 2013,
    year_max: int = 2024,
    models: tuple[str, ...] = ("persistence", "linear", "random_forest", "xgboost"),
) -> dict:
    uso = assert_climate_use(uso)
    columns = predictor_columns(uso, rain_feature_cols if uso == "predictor" else None)
    if uso == "predictor":
        frame = build_lagged_frame(
            df, ut_col=ut_col, year_min=year_min, year_max=year_max,
            rain=rain, rain_feature_cols=rain_feature_cols,
        )
    else:
        frame = build_lagged_frame(df, ut_col=ut_col, year_min=year_min, year_max=year_max)
    # Drop a climate subindex that is constant on the estimation sample.
    if CLIMATE_SUBINDEX in columns and frame[CLIMATE_SUBINDEX].nunique(dropna=True) <= 1:
        columns = [c for c in columns if c != CLIMATE_SUBINDEX]

    temporal_parts = []
    loto_parts = []
    fold_params = {}
    for name in models:
        pred, specs = walk_forward(frame, columns, ut_col, name)
        temporal_parts.append(pred)
        fold_params[name] = [
            {"test_year": s.test_year, "train_years": s.train_years, "params": s.params}
            for s in specs
        ]
        loto_parts.append(leave_one_territory_out(frame, columns, ut_col, name))

    temporal = pd.concat(temporal_parts, ignore_index=True)
    loto = pd.concat(loto_parts, ignore_index=True)
    pooled = summarize_predictions(temporal, ut_col)
    by_year = summarize_predictions(temporal, ut_col, by=TARGET_YEAR)
    loto_pooled = summarize_predictions(loto, ut_col)
    loto_by_territory = summarize_predictions(loto, ut_col, by=ut_col)
    return {
        "uso": uso,
        "predictor_columns": columns + [FEATURE_YEAR],
        "n_pairs": int(len(frame)),
        "territories": int(frame[ut_col].nunique()),
        "feature_years": [int(y) for y in sorted(frame[FEATURE_YEAR].unique())],
        "target_years": [int(y) for y in sorted(frame[TARGET_YEAR].unique())],
        "temporal_predictions": temporal,
        "temporal_pooled": pooled,
        "temporal_by_year": by_year,
        "loto_predictions": loto,
        "loto_pooled": loto_pooled,
        "loto_by_territory": loto_by_territory,
        "selected_params_by_fold": fold_params,
        "contemporaneous_linear": contemporaneous_linear_r2(df, year_min, year_max),
    }
