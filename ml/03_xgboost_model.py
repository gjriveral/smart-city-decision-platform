# © 2026 Gabriel Jaime Rivera León. Todos los derechos reservados.
# El diseño arquitectónico, modelado lógico, simulaciones y estructura de este
# software son obra original del autor. La lógica computacional implementada
# comprende pipelines ETL, índices de vulnerabilidad, ensamble XGBoost con
# validación temporal y simulación basada en agentes (ABM).

"""
03_xgboost_model.py
Train XGBoost regression model on vulnerability indices.
Temporal cross-validation (TimeSeriesSplit / leave-one-year-out).
Outputs:
  - models/vulnerability_model.pkl
  - models/ivc_predictions_oos.csv   (predicciones OOS por fold → entrada ABM)
  - models/shap_values.csv
  - models/shap_importance_by_fold.csv
  - models/shap_importance_cv_stability.csv   (media ± DE de |SHAP| entre folds)
  - models/shap_importance_cv_stability.png
  - models/feature_importance.png
  - logs/ml_execution.log  (appended)
"""

import os
import sys
import time
import logging
import warnings
import pickle
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ── Config de ciudad ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import get_config

cfg    = get_config()
UT_COL = cfg.unidad_territorial.title()   # "Comuna", "Localidad", etc.
_abs   = cfg.rutas_absolutas()

# ── Paths derivados de config ─────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROC = os.path.join(_abs.datos_sociales, "processed")
MODELS    = _abs.modelos
LOG_DIR   = _abs.logs
os.makedirs(MODELS, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "ml_execution.log"),
                            mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)
log.info("=" * 70)
log.info("03_xgboost_model.py  —  START")
log.info("=" * 70)

# ── Load data ──────────────────────────────────────────────────────────────────
csv_path = os.path.join(DATA_PROC, "indices_comunas.csv")
log.info(f"Loading: {csv_path}")
df = pd.read_csv(csv_path, encoding="utf-8-sig")
log.info(f"Shape: {df.shape}")
log.info(f"Columns: {df.columns.tolist()}")

# Accept configured UT_COL, Unidad_Territorial (demos multi-ciudad) or legacy Comuna
_ut_candidates = [UT_COL, "Unidad_Territorial", "Comuna", "Localidad", "comuna", "localidad"]
_ut_col_found = next((c for c in _ut_candidates if c in df.columns), df.columns[0])
log.info(
    f"{cfg.unidad_territorial.title()}s: {df[_ut_col_found].nunique()} "
    f"| Years: {sorted(df['Año'].unique().tolist())}"
)
# Standardise to configured column name so downstream code is uniform
if _ut_col_found != UT_COL:
    df = df.rename(columns={_ut_col_found: UT_COL})
    log.info(f"  Renamed column '{_ut_col_found}' -> '{UT_COL}'")

if "ivc" not in df.columns and "IVC" in df.columns:
    df = df.rename(columns={"IVC": "ivc"})
    log.info("  Renamed column 'IVC' -> 'ivc'")

# ── Feature engineering ────────────────────────────────────────────────────────
FEATURE_COLS = [
    "idx_desempleo", "idx_habitat", "idx_educacion",
    "idx_riesgo_clima", "idx_pobreza", "idx_tejido_social",
]
TARGET_COL = "ivc"

# Normalised year (0 → 1 range over available years)
years = df["Año"].values.astype(float)
df["año_normalizado"] = (years - years.min()) / max(years.max() - years.min(), 1)
FEATURE_COLS_EXT = FEATURE_COLS + ["año_normalizado"]

# Inventario de NaN tras el anti-fuga de 02_vulnerability_indices.py
_nan_feat = df[FEATURE_COLS_EXT].isna().sum()
_nan_y = int(df[TARGET_COL].isna().sum())
log.info(
    f"NaN en features (post anti-fuga): "
    + ", ".join(f"{c}={int(n)}" for c, n in _nan_feat.items() if n > 0)
    + ("" if _nan_feat.sum() else "ninguno")
)
log.info(f"NaN en target '{TARGET_COL}': {_nan_y}")

X = df[FEATURE_COLS_EXT].values
y = df[TARGET_COL].values

log.info(f"Ciudad             : {cfg.ciudad.nombre}")
log.info(f"Unidad territorial : {cfg.unidad_territorial} ({df[UT_COL].nunique()} zonas)")
log.info(f"Features: {FEATURE_COLS_EXT}")
log.info(
    f"X shape: {X.shape} | y stats (nanmean): "
    f"mean={np.nanmean(y):.4f}, std={np.nanstd(y):.4f}"
)

# ── Import XGBoost ─────────────────────────────────────────────────────────────
try:
    import xgboost as xgb
    log.info(f"XGBoost version: {xgb.__version__}")
except ImportError:
    log.error("XGBoost not installed. Run: pip install xgboost")
    sys.exit(1)

try:
    import shap
    log.info(f"SHAP version: {shap.__version__}")
    SHAP_AVAILABLE = True
except ImportError:
    log.warning("SHAP not installed. SHAP values will be skipped.")
    SHAP_AVAILABLE = False


def _xgb_accepts_nan_in_X(xgb_mod) -> bool:
    """Smoke-test: XGBoost acepta NaN en features (no en etiquetas)."""
    X_probe = np.array([[0.1, np.nan], [0.2, 0.3]], dtype=float)
    y_probe = np.array([0.4, 0.5], dtype=float)
    m = xgb_mod.XGBRegressor(
        objective="reg:squarederror", n_estimators=5, max_depth=1, verbosity=0
    )
    m.fit(X_probe, y_probe)
    _ = m.predict(X_probe)
    return True


# Confirmación explícita del manejo de NaN (punto 1 del requisito)
try:
    _xgb_accepts_nan_in_X(xgb)
    log.info(
        "Confirmado: XGBRegressor acepta NaN en X (enrutamiento nativo por split)."
    )
except Exception as e:
    log.error(f"XGBRegressor rechazó NaN en X: {type(e).__name__}: {e}")
    log.error("Detenido sin imputar. Revise versión de XGBoost / sklearn.")
    sys.exit(1)

# Las etiquetas con NaN SÍ son rechazadas por XGBoost ("Label contains NaN").
# No imputamos. Conservamos copia del corpus con NaN para la comparación
# anti-fuga vs mediana global (run_cv_leaky_comparison).
df_corpus_with_nan = df.copy()
_n_before = len(df)
df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
_n_dropped = _n_before - len(df)
if _n_dropped:
    log.warning(
        f"XGBoost/GridSearchCV rechazan NaN en y. "
        f"Se excluyen {_n_dropped} filas con {TARGET_COL}=NaN "
        f"(sin imputación de features ni del target). "
        f"Filas restantes: {len(df)}. "
        f"Años retenidos: {sorted(df['Año'].unique().tolist())}"
    )
X = df[FEATURE_COLS_EXT].values
y = df[TARGET_COL].values
log.info(
    f"Tras filtro de y válido — X NaN cells={int(np.isnan(X).sum())}, "
    f"y NaN={int(np.isnan(y).sum())}"
)

# ── Temporal Cross-Validation ──────────────────────────────────────────────────
log.info("\n--- Temporal Cross-Validation ---")

# Sort by year for temporal split
df_sorted = df.sort_values(["Año", UT_COL]).reset_index(drop=True)
X_sorted = df_sorted[FEATURE_COLS_EXT].values
y_sorted = df_sorted[TARGET_COL].values

unique_years = sorted(df_sorted["Año"].unique())
log.info(f"Years in data: {unique_years}")

n_splits = min(len(unique_years) - 1, 5)
if n_splits < 2:
    log.warning("Not enough years for proper temporal split. Using 2-fold.")
    n_splits = 2

tscv = TimeSeriesSplit(n_splits=n_splits)

fold_metrics = []
fold_models = []       # estimador ajustado de cada pliegue (para analisis de
                        # estabilidad SHAP entre pliegues — no se conservaba antes)
fold_test_years_list = []
# Importancia SHAP (|SHAP| medio) por feature en el conjunto de VALIDACIÓN
# de cada pliegue — permite cuantificar la estabilidad de la jerarquía.
fold_shap_importances = []   # list[dict feature -> mean_|SHAP|]
fold_shap_rankings = []      # list[list feature ordenada desc]
# Predicciones OOS por pliegue (puente XGBoost → ABM)
oos_prediction_rows = []
best_rmse = np.inf
best_model = None
best_fold_params = {}

# Hyperparameter grid: regularisation terms (reg_alpha, reg_lambda, gamma)
# address the CV R²=0.954 vs full-data R²=0.9985 overfitting gap.
# max_depth capped at 3 and min_child_weight raised to 3–5 for bias–variance
# balance on the small dataset (514 rows, 7 features).
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [2, 3],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.7, 0.8],
    "colsample_bytree": [0.7, 0.8],
    "min_child_weight": [3, 5],
    "reg_alpha": [0.0, 0.1],    # L1 regularisation
    "reg_lambda": [1.0, 2.0],   # L2 regularisation (XGBoost default=1)
    "gamma": [0.0, 0.1],        # min loss reduction to allow a split
}

log.info(f"Hyperparameter grid: {param_grid}")

t0 = time.time()

for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X_sorted)):
    X_tr, X_te = X_sorted[train_idx], X_sorted[test_idx]
    y_tr, y_te = y_sorted[train_idx], y_sorted[test_idx]

    train_years = sorted(df_sorted.iloc[train_idx]["Año"].unique().tolist())
    test_years  = sorted(df_sorted.iloc[test_idx]["Año"].unique().tolist())
    log.info(f"\n  Fold {fold_idx+1}: train years={train_years}, test years={test_years}")
    log.info(f"    Train size={len(X_tr)}, Test size={len(X_te)}")

    # Grid search on training fold
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )

    # Inner CV splits: at least 2, at most 3, bounded by training-fold size.
    inner_cv = max(2, min(3, len(X_tr) // 30))

    gs = GridSearchCV(
        model, param_grid,
        cv=inner_cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    gs.fit(X_tr, y_tr)

    best_params = gs.best_params_
    log.info(f"    Best params: {best_params}")

    y_pred = gs.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae  = mean_absolute_error(y_te, y_pred)
    r2   = r2_score(y_te, y_pred)

    log.info(f"    RMSE={rmse:.5f}, MAE={mae:.5f}, R²={r2:.5f}")
    fold_metrics.append({
        "fold": fold_idx + 1,
        "train_years": str(train_years),
        "test_years": str(test_years),
        "RMSE": rmse, "MAE": mae, "R2": r2,
        "best_params": str(best_params),
    })
    fold_models.append(gs.best_estimator_)
    fold_test_years_list.append(test_years)

    # Acumular predicciones genuinamente fuera de muestra (test del fold)
    meta_te = df_sorted.iloc[test_idx]
    for i, (_, row) in enumerate(meta_te.iterrows()):
        oos_prediction_rows.append({
            UT_COL: row[UT_COL],
            "Año": int(row["Año"]),
            "ivc_observado": float(y_te[i]),
            "ivc_predicho": float(y_pred[i]),
            "fold": fold_idx + 1,
        })

    # ── SHAP sobre el conjunto de validación de ESTE pliegue ───────────────
    # Responde a la crítica del jurado: la atribución no se calcula solo sobre
    # el modelo reentrenado al 100 %, sino en cada fold OOS, para verificar
    # que el ranking de características es estable entre ventanas temporales.
    if SHAP_AVAILABLE:
        try:
            explainer_fold = shap.TreeExplainer(gs.best_estimator_)
            shap_te = explainer_fold.shap_values(X_te)
            mean_abs = np.abs(shap_te).mean(axis=0)
            fold_imp = {
                feat: float(mean_abs[i]) for i, feat in enumerate(FEATURE_COLS_EXT)
            }
            ranking = [FEATURE_COLS_EXT[i] for i in np.argsort(-mean_abs)]
            fold_shap_importances.append(fold_imp)
            fold_shap_rankings.append(ranking)
            top3 = ", ".join(f"{f}={fold_imp[f]:.4f}" for f in ranking[:3])
            log.info(f"    SHAP (val) top-3: {top3}")
        except Exception as e:
            log.warning(f"    SHAP en fold {fold_idx + 1} falló: {e}")
            fold_shap_importances.append({f: np.nan for f in FEATURE_COLS_EXT})
            fold_shap_rankings.append(FEATURE_COLS_EXT[:])

    if rmse < best_rmse:
        best_rmse = rmse
        best_model = gs.best_estimator_
        best_fold_params = gs.best_params_.copy()
        log.info(f"    → New best model (RMSE={best_rmse:.5f}) from fold {fold_idx + 1}")

training_time = time.time() - t0
log.info(f"\n  Training time: {training_time:.1f}s")

# Summarise metrics
metrics_df = pd.DataFrame(fold_metrics)
log.info("\n--- Cross-Validation Summary ---")
log.info(f"\n{metrics_df[['fold','RMSE','MAE','R2']].to_string(index=False)}")
log.info(f"\nMean RMSE : {metrics_df['RMSE'].mean():.5f} ± {metrics_df['RMSE'].std():.5f}")
log.info(f"Mean MAE  : {metrics_df['MAE'].mean():.5f} ± {metrics_df['MAE'].std():.5f}")
log.info(f"Mean R²   : {metrics_df['R2'].mean():.5f} ± {metrics_df['R2'].std():.5f}")

# ── Predicciones OOS concatenadas (puente hacia abm/04_urban_model.py) ────────
log.info("\n--- IVC predicciones fuera de muestra (OOS) ---")
# Años que solo aparecen en entrenamiento del primer pliegue no reciben
# predicción OOS (TimeSeriesSplit no los pone nunca en test).
_all_years_sorted = sorted(df_sorted["Año"].unique().tolist())
_oos_years = sorted({r["Año"] for r in oos_prediction_rows})
_train_only_years = [y for y in _all_years_sorted if y not in _oos_years]
if fold_metrics:
    _first_train = fold_metrics[0].get("train_years", "")
    log.info(
        f"Primer pliegue — años solo de entrenamiento (sin predicción OOS): "
        f"{_first_train}. "
        f"Años sin ninguna predicción OOS en el corpus filtrado: {_train_only_years}."
    )
log.info(
    f"Años con predicción OOS (unión de sets de validación): {_oos_years}"
)
oos_df = pd.DataFrame(oos_prediction_rows)
# Columnas pedidas + fold para trazabilidad; orden estable
oos_df = oos_df[[UT_COL, "Año", "ivc_observado", "ivc_predicho", "fold"]].sort_values(
    ["Año", UT_COL]
).reset_index(drop=True)
oos_path = os.path.join(MODELS, "ivc_predictions_oos.csv")
oos_df.to_csv(oos_path, index=False, encoding="utf-8-sig")
log.info(
    f"OOS predictions saved: {oos_path} "
    f"({len(oos_df)} filas, {oos_df[UT_COL].nunique()} {cfg.unidad_territorial}s)"
)
log.info(
    f"  RMSE OOS global = "
    f"{np.sqrt(mean_squared_error(oos_df['ivc_observado'], oos_df['ivc_predicho'])):.5f}"
)


def run_cv_leaky_comparison(
    df_full: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    param_grid: dict,
    n_splits: int,
    ut_col: str,
) -> None:
    """
    Comparación temporal (solo log) entre:
      A) anti-fuga: datos con NaN; filas sin target se excluyen (sin imputar)
      B) leaky: mediana global sobre TODO el corpus ANTES del split
         (comportamiento previo a 02_vulnerability_indices anti-fuga)

    No altera el pipeline principal ni la selección del mejor pliegue.
    """
    log.info("\n" + "=" * 70)
    log.info("run_cv_leaky_comparison() — impacto de la fuga temporal (log only)")
    log.info("=" * 70)

    def _cv_mean_rmse(X_arr, y_arr, df_meta, label: str):
        tscv_local = TimeSeriesSplit(n_splits=n_splits)
        rmses = []
        log.info(f"\n  [{label}] n={len(y_arr)}, "
                 f"años={sorted(df_meta['Año'].unique().tolist())}, "
                 f"X_NaN={int(np.isnan(X_arr).sum())}, y_NaN={int(np.isnan(y_arr).sum())}")
        for fold_idx, (tr, te) in enumerate(tscv_local.split(X_arr)):
            X_tr, X_te = X_arr[tr], X_arr[te]
            y_tr, y_te = y_arr[tr], y_arr[te]
            # Si el fold de test aún tiene NaN en y (no debería tras filtro), fallar claro
            if np.isnan(y_tr).any() or np.isnan(y_te).any():
                raise ValueError(
                    f"[{label}] fold {fold_idx+1}: NaN en y — "
                    "GridSearchCV/XGBoost lo rechazan; no se imputa."
                )
            model = xgb.XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                verbosity=0,
                n_jobs=-1,
            )
            inner_cv = max(2, min(3, len(X_tr) // 30))
            gs = GridSearchCV(
                model, param_grid,
                cv=inner_cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
                refit=True,
                verbose=0,
            )
            gs.fit(X_tr, y_tr)
            y_pred = gs.predict(X_te)
            rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
            rmses.append(rmse)
            ty = sorted(df_meta.iloc[te]["Año"].unique().tolist())
            log.info(f"    Fold {fold_idx+1} test_years={ty}  RMSE={rmse:.5f}")
        mean_rmse = float(np.mean(rmses))
        std_rmse = float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0
        log.info(f"  [{label}] Mean RMSE = {mean_rmse:.5f} ± {std_rmse:.5f}")
        return mean_rmse, std_rmse, rmses

    # ── A) Anti-fuga: sin mediana global; solo filas con y finito ───────────
    df_clean = (
        df_full.dropna(subset=[target_col])
        .sort_values(["Año", ut_col])
        .reset_index(drop=True)
    )
    X_clean = df_clean[feature_cols].to_numpy(dtype=float)
    y_clean = df_clean[target_col].to_numpy(dtype=float)
    mean_clean, std_clean, rmses_clean = _cv_mean_rmse(
        X_clean, y_clean, df_clean, "ANTI-FUGA (NaN explícitos; sin mediana global)"
    )

    # ── B) Leaky: mediana global del corpus completo ANTES del split ───────
    df_leaky = df_full.sort_values(["Año", ut_col]).reset_index(drop=True).copy()
    cols_impute = feature_cols + [target_col]
    global_medians = df_leaky[cols_impute].median(numeric_only=True)
    log.info(
        "  [LEAKY] Medianas globales (corpus completo, pre-split): "
        + ", ".join(f"{c}={global_medians[c]:.4f}" for c in cols_impute)
    )
    df_leaky[cols_impute] = df_leaky[cols_impute].fillna(global_medians)
    X_leaky = df_leaky[feature_cols].to_numpy(dtype=float)
    y_leaky = df_leaky[target_col].to_numpy(dtype=float)
    mean_leaky, std_leaky, rmses_leaky = _cv_mean_rmse(
        X_leaky, y_leaky, df_leaky,
        "LEAKY (mediana global pre-split, como antes del fix)"
    )

    delta = mean_clean - mean_leaky
    rel_pct = 100.0 * delta / mean_leaky if mean_leaky != 0 else float("nan")
    log.info("\n--- Resumen impacto de la fuga temporal ---")
    log.info(f"  RMSE medio ANTI-FUGA : {mean_clean:.5f} ± {std_clean:.5f}")
    log.info(f"  RMSE medio LEAKY     : {mean_leaky:.5f} ± {std_leaky:.5f}")
    log.info(
        f"  Diferencia (anti-fuga − leaky) = {delta:+.5f} "
        f"({rel_pct:+.2f}% relativo al leaky)"
    )
    if abs(delta) < 0.001:
        log.info(
            "  Interpretación: impacto MATERIAL BAJO (|ΔRMSE| < 0.001); "
            "la fuga no alteraba de forma sustancial el RMSE medio de CV."
        )
    else:
        log.info(
            "  Interpretación: impacto MATERIAL DETECTABLE (|ΔRMSE| ≥ 0.001); "
            "documentar en tesis la magnitud y dirección del sesgo."
        )
    log.info(
        "  Nota: el brazo anti-fuga excluye años sin IVC; el brazo leaky "
        "incluye todo el corpus tras rellenar con mediana global — la "
        "comparación mide el efecto conjunto (cobertura + fuga de escala)."
    )
    log.info("=" * 70)


# Comparación explícita anti-fuga vs leaky (no sustituye el pipeline principal)
run_cv_leaky_comparison(
    df_full=df_corpus_with_nan,
    feature_cols=FEATURE_COLS_EXT,
    target_col=TARGET_COL,
    param_grid=param_grid,
    n_splits=n_splits,
    ut_col=UT_COL,
)

# ── Retrain on full data ───────────────────────────────────────────────────────
log.info("\n--- Retraining on full data ---")
# Use the hyperparameters from the fold that achieved the lowest RMSE,
# rather than the last fold — avoids implicit dependence on fold ordering.
final_params = best_fold_params
log.info(f"Final hyperparameters (from best CV fold): {final_params}")
final_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    verbosity=0,
    n_jobs=-1,
    **final_params,
)
final_model.fit(X_sorted, y_sorted)
log.info("Final model trained.")

# Full-data in-sample metrics (for reference)
y_full_pred = final_model.predict(X_sorted)
full_rmse = np.sqrt(mean_squared_error(y_sorted, y_full_pred))
full_r2   = r2_score(y_sorted, y_full_pred)
log.info(f"Full-data RMSE={full_rmse:.5f}, R²={full_r2:.5f}")

# ── Save model ─────────────────────────────────────────────────────────────────
model_path = os.path.join(MODELS, "vulnerability_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(final_model, f)
log.info(f"Model saved: {model_path}")

# Save metrics JSON
metrics_path = os.path.join(MODELS, "cv_metrics.json")
summary = {
    "cv_folds": fold_metrics,
    "mean_RMSE": float(metrics_df["RMSE"].mean()),
    "mean_MAE":  float(metrics_df["MAE"].mean()),
    "mean_R2":   float(metrics_df["R2"].mean()),
    "std_RMSE":  float(metrics_df["RMSE"].std()),
    "full_data_RMSE": float(full_rmse),
    "full_data_R2":   float(full_r2),
    "training_time_s": float(training_time),
    "best_params": final_params,
    "features": FEATURE_COLS_EXT,
}
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
log.info(f"Metrics saved: {metrics_path}")

# ── Feature importance plot ────────────────────────────────────────────────────
log.info("\n--- Feature Importance ---")
fi = final_model.feature_importances_
fi_df = pd.DataFrame({
    "feature": FEATURE_COLS_EXT,
    "importance": fi,
}).sort_values("importance", ascending=False)
log.info(f"\n{fi_df.to_string(index=False)}")

fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#d62728" if i == 0 else "#1f77b4" for i in range(len(fi_df))]
ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color=colors[::-1])
ax.set_xlabel("XGBoost Gain Importance")
ax.set_title("Feature Importance — Vulnerability Model\n(higher = more influential on IVC)")
ax.axvline(x=fi_df["importance"].mean(), color="gray", linestyle="--", label="Mean")
ax.legend()
plt.tight_layout()
fig_path = os.path.join(MODELS, "feature_importance.png")
plt.savefig(fig_path, dpi=150)
plt.close()
log.info(f"Feature importance plot saved: {fig_path}")

# ── SHAP values ────────────────────────────────────────────────────────────────
if SHAP_AVAILABLE:
    log.info("\n--- SHAP Values ---")
    try:
        # ── Estabilidad SHAP entre pliegues de CV (crítica del jurado) ─────
        if fold_shap_importances:
            log.info("\n--- SHAP Stability across CV folds (validation sets) ---")
            # Matriz folds × features
            fold_imp_df = pd.DataFrame(fold_shap_importances)
            fold_imp_df.insert(0, "fold", np.arange(1, len(fold_imp_df) + 1))
            # Ranking por fold
            rank_rows = []
            for i, ranking in enumerate(fold_shap_rankings):
                for rank, feat in enumerate(ranking, start=1):
                    rank_rows.append({"fold": i + 1, "feature": feat, "rank": rank})
            fold_rank_df = pd.DataFrame(rank_rows)

            # Media y desviación estándar de |SHAP| medio a través de folds
            stability = pd.DataFrame({
                "feature": FEATURE_COLS_EXT,
                "shap_abs_mean_across_folds": [
                    float(fold_imp_df[f].mean()) for f in FEATURE_COLS_EXT
                ],
                "shap_abs_std_across_folds": [
                    float(fold_imp_df[f].std(ddof=1)) if len(fold_imp_df) > 1
                    else 0.0
                    for f in FEATURE_COLS_EXT
                ],
            })
            # Coeficiente de variación (robustez relativa)
            stability["cv_pct"] = np.where(
                stability["shap_abs_mean_across_folds"] > 0,
                100.0 * stability["shap_abs_std_across_folds"]
                / stability["shap_abs_mean_across_folds"],
                np.nan,
            )
            # Ranking medio (1 = más importante)
            mean_rank = (
                fold_rank_df.groupby("feature")["rank"].mean()
                .reindex(FEATURE_COLS_EXT)
            )
            stability["mean_rank_across_folds"] = mean_rank.values
            stability = stability.sort_values(
                "shap_abs_mean_across_folds", ascending=False
            ).reset_index(drop=True)
            stability["ranking_by_mean"] = np.arange(1, len(stability) + 1)

            log.info(
                "\nSHAP |importancia| media ± DE entre pliegues "
                "(conjunto de validación de cada fold):\n"
                + stability.to_string(index=False)
            )

            # Exportar: detalle por fold + resumen de estabilidad
            fold_shap_path = os.path.join(MODELS, "shap_importance_by_fold.csv")
            fold_imp_df.to_csv(fold_shap_path, index=False, encoding="utf-8-sig")
            log.info(f"SHAP por pliegue guardado: {fold_shap_path}")

            stability_path = os.path.join(MODELS, "shap_importance_cv_stability.csv")
            stability.to_csv(stability_path, index=False, encoding="utf-8-sig")
            log.info(f"Estabilidad SHAP (media/DE) guardada: {stability_path}")

            # Figura: barras con error = DE entre folds
            fig_stab, ax_stab = plt.subplots(figsize=(9, 5))
            stab_plot = stability.sort_values(
                "shap_abs_mean_across_folds", ascending=True
            )
            ax_stab.barh(
                stab_plot["feature"],
                stab_plot["shap_abs_mean_across_folds"],
                xerr=stab_plot["shap_abs_std_across_folds"],
                color="#2ca02c",
                ecolor="#333333",
                capsize=3,
                alpha=0.9,
            )
            ax_stab.set_xlabel("Mean |SHAP| across CV folds (± 1 SD)")
            ax_stab.set_title(
                "SHAP Importance Stability — Temporal CV\n"
                "(computed on each fold's validation set)"
            )
            plt.tight_layout()
            stab_fig_path = os.path.join(MODELS, "shap_importance_cv_stability.png")
            plt.savefig(stab_fig_path, dpi=150)
            plt.close()
            log.info(f"SHAP stability plot saved: {stab_fig_path}")

            # Spearman media entre rankings de pliegues (resumen de robustez)
            if len(fold_shap_rankings) >= 2:
                from scipy import stats as _stats
                order0 = fold_shap_rankings[0]
                spearman_pairs = []
                for a in range(len(fold_shap_rankings)):
                    for b in range(a + 1, len(fold_shap_rankings)):
                        ra = [fold_shap_rankings[a].index(f) for f in order0]
                        rb = [fold_shap_rankings[b].index(f) for f in order0]
                        rho, _ = _stats.spearmanr(ra, rb)
                        spearman_pairs.append(rho)
                mean_spearman = float(np.mean(spearman_pairs))
                # Umbral de clasificación "estable" / "inestable": 0.70.
                # Origen: ARBITRARIO (no proviene de un test de significancia ni
                # de una convención bibliográfica fijada en el protocolo).
                # Razonamiento adoptado para el informe: con 7 características,
                # se considera estable una concordancia de rango equivalente a
                # acuerdo en al menos ~5 de 7 pares ordenados, lo que corresponde
                # aproximadamente a un coeficiente de Spearman ρ ≥ 0.70. El valor
                # no se altera aquí; solo se documenta para trazabilidad doctoral.
                log.info(
                    f"Spearman medio entre rankings SHAP de pliegues: "
                    f"{mean_spearman:.4f} "
                    f"({'estable' if mean_spearman >= 0.7 else 'inestable'})"
                )
                # Adjuntar al JSON de métricas si ya se escribió — se reescribe abajo
                summary["shap_rank_spearman_mean"] = mean_spearman
                summary["shap_stability"] = {
                    row["feature"]: {
                        "mean": round(float(row["shap_abs_mean_across_folds"]), 6),
                        "std": round(float(row["shap_abs_std_across_folds"]), 6),
                        "mean_rank": round(float(row["mean_rank_across_folds"]), 3),
                    }
                    for _, row in stability.iterrows()
                }
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)

        # ── SHAP global sobre el modelo final (exploratorio / territorio-año) ─
        # shap_values.csv NO es la fuente del ranking headline de determinantes.
        # Ese rol corresponde a shap_importance_cv_stability.csv (ranking_by_mean),
        # validado OOS por CV. Este bloque in-sample se conserva para detalle
        # espacial (dashboard expander, pesos ABM por comuna) y gráficas
        # exploratorias — nunca para afirmar "el determinante dominante".
        explainer = shap.TreeExplainer(final_model)
        shap_vals = explainer.shap_values(X_sorted)
        shap_df = pd.DataFrame(shap_vals, columns=FEATURE_COLS_EXT)
        shap_df["Comuna"] = df_sorted[UT_COL].values
        shap_df["Año"]    = df_sorted["Año"].values

        shap_path = os.path.join(MODELS, "shap_values.csv")
        shap_df.to_csv(shap_path, index=False, encoding="utf-8-sig")
        log.info(f"SHAP values (full-data model, exploratorio) saved: {shap_path}")

        # Mean |SHAP| per feature (referencia in-sample; ranking oficial = CV stability)
        mean_shap = pd.DataFrame({
            "feature": FEATURE_COLS_EXT,
            "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        log.info(
            f"\nTop 5 SHAP features (full model, IN-SAMPLE — no oficial):\n"
            f"{mean_shap.head(5).to_string(index=False)}"
        )

        # SHAP summary bar plot
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        ax2.barh(
            mean_shap["feature"][::-1],
            mean_shap["mean_abs_shap"][::-1],
            color="#2ca02c",
        )
        ax2.set_xlabel("Mean |SHAP value|")
        ax2.set_title(
            "SHAP Feature Importance — full data (exploratory; "
            "official ranking = CV stability CSV)"
        )
        plt.tight_layout()
        shap_fig_path = os.path.join(MODELS, "shap_importance.png")
        plt.savefig(shap_fig_path, dpi=150)
        plt.close()
        log.info(f"SHAP plot saved: {shap_fig_path}")

    except Exception as e:
        log.error(f"SHAP computation failed: {e}")
else:
    log.warning("SHAP not available — skipping SHAP analysis.")

# ── Most Vulnerable Communes ───────────────────────────────────────────────────
log.info(f"\n--- Most Vulnerable {cfg.unidad_territorial.title()}s (mean IVC) ---")
vuln_rank = df.groupby(UT_COL)["ivc"].mean().sort_values(ascending=False)
log.info(f"\n{vuln_rank.head(10).to_string()}")

# Save vulnerability ranking
rank_path = os.path.join(MODELS, "vulnerability_ranking.csv")
vuln_rank.reset_index().rename(columns={UT_COL: UT_COL, "ivc": "mean_ivc"}).to_csv(
    rank_path, index=False, encoding="utf-8-sig"
)
log.info(f"Ranking saved: {rank_path}")

log.info("\n" + "=" * 70)
log.info("03_xgboost_model.py  —  DONE")
log.info("=" * 70)
