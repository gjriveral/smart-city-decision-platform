"""
03_xgboost_model.py
Train XGBoost regression model on vulnerability indices.
Temporal cross-validation (TimeSeriesSplit / leave-one-year-out).
Outputs:
  - models/vulnerability_model.pkl
  - models/shap_values.csv
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

# Accept both the configured UT_COL and the legacy "Comuna" header
_ut_col_found = UT_COL if UT_COL in df.columns else (
    "Comuna" if "Comuna" in df.columns else df.columns[0]
)
log.info(
    f"{cfg.unidad_territorial.title()}s: {df[_ut_col_found].nunique()} "
    f"| Years: {sorted(df['Año'].unique().tolist())}"
)
# Standardise to configured column name so downstream code is uniform
if _ut_col_found != UT_COL:
    df = df.rename(columns={_ut_col_found: UT_COL})
    log.info(f"  Renamed column '{_ut_col_found}' -> '{UT_COL}'")

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

X = df[FEATURE_COLS_EXT].values
y = df[TARGET_COL].values

log.info(f"Ciudad             : {cfg.ciudad.nombre}")
log.info(f"Unidad territorial : {cfg.unidad_territorial} ({df[UT_COL].nunique()} zonas)")
log.info(f"Features: {FEATURE_COLS_EXT}")
log.info(f"X shape: {X.shape} | y stats: mean={y.mean():.4f}, std={y.std():.4f}")

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
        explainer = shap.TreeExplainer(final_model)
        shap_vals = explainer.shap_values(X_sorted)
        shap_df = pd.DataFrame(shap_vals, columns=FEATURE_COLS_EXT)
        shap_df["Comuna"] = df_sorted["Comuna"].values
        shap_df["Año"]    = df_sorted["Año"].values

        shap_path = os.path.join(MODELS, "shap_values.csv")
        shap_df.to_csv(shap_path, index=False, encoding="utf-8-sig")
        log.info(f"SHAP values saved: {shap_path}")

        # Mean |SHAP| per feature
        mean_shap = pd.DataFrame({
            "feature": FEATURE_COLS_EXT,
            "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        log.info(f"\nTop 5 SHAP features:\n{mean_shap.head(5).to_string(index=False)}")

        # SHAP summary bar plot
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        ax2.barh(
            mean_shap["feature"][::-1],
            mean_shap["mean_abs_shap"][::-1],
            color="#2ca02c",
        )
        ax2.set_xlabel("Mean |SHAP value|")
        ax2.set_title("SHAP Feature Importance — Vulnerability Model")
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
