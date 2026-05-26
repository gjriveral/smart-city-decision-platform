"""
run_abm_pipeline.py
===================
Master runner: installs deps, runs ABM model + scenario analysis.
Execute: python run_abm_pipeline.py
"""

import subprocess
import sys
import os
import logging

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "abm_execution.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

def run(cmd, desc):
    logger.info(f">> {desc}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        for line in result.stdout.strip().splitlines():
            logger.info(f"  {line}")
    if result.stderr:
        for line in result.stderr.strip().splitlines():
            logger.warning(f"  {line}")
    if result.returncode != 0:
        logger.error(f"  FAILED (code {result.returncode})")
    else:
        logger.info(f"  OK")
    return result.returncode

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("ABM PIPELINE RUNNER")
    logger.info("=" * 60)

    # 1. Install dependencies
    run([sys.executable, "-m", "pip", "install", "mesa", "matplotlib", "seaborn", "--quiet"],
        "Installing dependencies")

    # 2. Run ABM model
    model_path = os.path.join(ROOT, "abm", "04_urban_model.py")
    rc = run([sys.executable, model_path], "Running ABM model (3 scenarios × 10 years)")
    if rc != 0:
        logger.error("Model run failed. Aborting.")
        sys.exit(1)

    # 3. Run scenario analysis
    analysis_path = os.path.join(ROOT, "abm", "05_scenario_analysis.py")
    rc = run([sys.executable, analysis_path], "Running scenario analysis + HTML report")
    if rc != 0:
        logger.error("Analysis run failed.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Results: {os.path.join(ROOT, 'data', 'processed', 'simulation_results.csv')}")
    logger.info(f"  Report:  {os.path.join(ROOT, 'reports', 'scenario_report.html')}")
    logger.info("=" * 60)
