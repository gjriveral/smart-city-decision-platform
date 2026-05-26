"""
run_ml_pipeline.py
Master runner — install dependencies, run vulnerability indices + XGBoost.
Execute from project root:
  python run_ml_pipeline.py
"""
import subprocess
import sys
import os
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

log_path = os.path.join(LOG_DIR, "ml_execution.log")

PACKAGES = ["xgboost", "shap", "scikit-learn", "openpyxl", "matplotlib", "pandas", "numpy"]

def run(cmd, label):
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=False, text=True
    )
    elapsed = time.time() - t0
    print(f"  [{label}] Exit code: {result.returncode} | Time: {elapsed:.1f}s")
    return result.returncode

# 1. Install packages
print("\n>>> Installing / verifying packages ...")
for pkg in PACKAGES:
    run([sys.executable, "-m", "pip", "install", pkg, "--quiet"], f"pip install {pkg}")

# 2. Run vulnerability indices
print("\n>>> Running 02_vulnerability_indices.py ...")
rc1 = run([sys.executable, os.path.join(ROOT, "ml", "02_vulnerability_indices.py")],
          "02_vulnerability_indices")

# 3. Run XGBoost model
if rc1 == 0:
    print("\n>>> Running 03_xgboost_model.py ...")
    rc2 = run([sys.executable, os.path.join(ROOT, "ml", "03_xgboost_model.py")],
              "03_xgboost_model")
else:
    print("\n!!! Skipping model training — indices script failed.")

print(f"\n{'='*60}")
print(f"  Pipeline complete. Log: {log_path}")
print(f"{'='*60}")
