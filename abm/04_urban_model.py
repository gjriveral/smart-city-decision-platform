"""
04_urban_model.py
=================
Mesa Agent-Based Model for Urban Strategic Investment.
- GobiernoAgent      : allocation strategies (focalizado, distribuido, adaptativo)
- PoblacionAgent     : wellbeing dynamics per territorial unit
- InfrastructuraAgent: infrastructure capacity dynamics per territorial unit

Three scenarios x N_YEARS simulation years → simulation_results.csv

Configuración de ciudad:
  La lista de unidades territoriales (antes COMUNAS fija con 16 entradas) y
  sus IVC iniciales se cargan DINÁMICAMENTE desde el CSV generado por el
  pipeline ML o, como fallback, desde los territorios definidos en config.yaml.
  El nombre de la columna de unidad territorial se lee de config.yaml via
  config_loader, por lo que el modelo funciona con cualquier ciudad colombiana.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import logging
from pathlib import Path

# ─── Config de ciudad ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config_loader import get_config

cfg    = get_config()
UT     = cfg.unidad_territorial        # "comuna", "localidad", etc.
UT_COL = UT.title()                    # "Comuna", "Localidad", etc.
_abs   = cfg.rutas_absolutas()

# ─── Logging ────────────────────────────────────────────────────────────────
LOG_DIR = _abs.logs
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

# ─── Try to import Mesa ───────────────────────────────────────────────────────
try:
    import mesa
    MESA_VERSION = tuple(int(x) for x in mesa.__version__.split(".")[:2])
    logger.info(f"Mesa version detected: {mesa.__version__}")
except ImportError:
    logger.error("Mesa not installed. Run: py -m pip install mesa")
    sys.exit(1)

# ─── Simulation parameters ───────────────────────────────────────────────────
ANNUAL_BUDGET = 1_000_000   # million COP total per year
N_YEARS       = 10
SCENARIOS     = ["focalizado", "distribuido", "adaptativo"]


# ─── Dynamic IVC loader ───────────────────────────────────────────────────────
def load_ivc_from_csv(csv_path: str) -> dict[str, float]:
    """
    Loads IVC initial values from indices_comunas.csv (latest year per territory).
    Returns a dict {territory_name: ivc_value} for ALL territories found in the CSV.
    Falls back to config.yaml territorios with IVC=0.35 if CSV is unavailable.
    """
    fallback = {t.nombre: 0.35 for t in cfg.territorios}

    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]

        # Accept both the configured UT_COL and the generic "Comuna" header
        ut_col_found = None
        for candidate in [UT_COL, "Comuna", "Localidad", "Corregimiento"]:
            if candidate in df.columns:
                ut_col_found = candidate
                break

        if ut_col_found is None or "ivc" not in df.columns:
            logger.warning(
                f"CSV missing expected columns ('{UT_COL}' / 'ivc'). "
                f"Using config.yaml fallback ({len(fallback)} {UT}s)."
            )
            return fallback

        # Take the last available year for each territory
        latest = df.sort_values("Año").groupby(ut_col_found).last().reset_index()
        ivc_map = {
            str(row[ut_col_found]).strip(): float(row["ivc"])
            for _, row in latest.iterrows()
        }
        logger.info(
            f"IVC loaded from CSV: {len(ivc_map)} {UT}s "
            f"(years {df['Año'].min()}–{df['Año'].max()})"
        )
        return ivc_map

    except Exception as e:
        logger.warning(f"Could not load CSV ({e}). Using config.yaml fallback.")
        return fallback


# ─── Agent Definitions ────────────────────────────────────────────────────────

class PoblacionAgent:
    """Represents the population of one territorial unit."""

    def __init__(self, unique_id, model, territory_id: str, ivc_initial: float):
        self.unique_id   = unique_id
        self.model       = model
        self.territory_id = territory_id
        self.ivc         = ivc_initial
        self.wellbeing   = max(0.0, min(1.0, 1.0 - ivc_initial))
        self.investment_received   = 0.0
        self._pending_investment   = 0.0

    def receive_investment(self, amount: float):
        self._pending_investment = amount

    def step(self):
        self.investment_received = self._pending_investment
        # avg_inv derived from model's actual territory count — no hardcoded 16
        avg_inv   = ANNUAL_BUDGET / self.model.n_territorios
        inv_ratio = self.investment_received / avg_inv if avg_inv > 0 else 0
        noise     = max(0.85, min(1.15, 1.0 + random.gauss(0, 0.05)))
        delta     = 0.15 * inv_ratio * (1.0 - self.wellbeing) * noise
        self.wellbeing = max(0.0, min(1.0, self.wellbeing + delta))
        self.ivc       = max(0.0, self.ivc * (1.0 - 0.05 * inv_ratio))
        self._pending_investment = 0.0


class InfrastructuraAgent:
    """Represents infrastructure capacity of one territorial unit."""

    DECAY_RATE = 0.03

    def __init__(self, unique_id, model, territory_id: str, ivc_initial: float):
        self.unique_id    = unique_id
        self.model        = model
        self.territory_id = territory_id
        self.capacity     = max(0.1, min(0.9, 1.0 - ivc_initial))
        self.maintenance_cost      = ivc_initial * 50
        self.investment_received   = 0.0
        self._pending_investment   = 0.0

    def receive_investment(self, amount: float):
        self._pending_investment = amount

    def step(self):
        self.investment_received = self._pending_investment
        avg_inv      = ANNUAL_BUDGET / self.model.n_territorios
        inv_normalized = self.investment_received / avg_inv if avg_inv > 0 else 0
        self.capacity  = max(0.0, self.capacity - self.DECAY_RATE)
        self.capacity  = min(1.0, self.capacity + 0.10 * inv_normalized)
        self._pending_investment = 0.0


class GobiernoAgent:
    """Single government agent that allocates budget across all territorial units."""

    def __init__(self, unique_id, model, strategy: str):
        self.unique_id    = unique_id
        self.model        = model
        self.strategy     = strategy
        self.budget_total = ANNUAL_BUDGET
        self._year        = 0
        # History indexed by territory — derived from model's dynamic territory list
        self._improvement_history = {t: [] for t in self.model.territorios}

    def _get_ivc_scores(self) -> dict[str, float]:
        return {t: self.model.poblacion_agents[t].ivc for t in self.model.territorios}

    def allocate_focalizado(self) -> dict[str, float]:
        """80% to top-5 most vulnerable, 20% equally distributed."""
        scores = self._get_ivc_scores()
        sorted_t = sorted(scores, key=scores.get, reverse=True)
        n_top    = min(5, len(sorted_t))
        top_n    = sorted_t[:n_top]
        rest     = sorted_t[n_top:]
        pool_top  = 0.80 * self.budget_total
        pool_rest = 0.20 * self.budget_total
        allocation: dict[str, float] = {}
        for t in self.model.territorios:
            if t in top_n:
                allocation[t] = pool_top / len(top_n)
            else:
                allocation[t] = pool_rest / max(len(rest), 1)
        return allocation

    def allocate_distribuido(self) -> dict[str, float]:
        """Equal allocation to all territorial units."""
        equal = self.budget_total / self.model.n_territorios
        return {t: equal for t in self.model.territorios}

    def allocate_adaptativo(self) -> dict[str, float]:
        """Reweight every 2 years based on IVC x (1 / improvement_rate)."""
        scores  = self._get_ivc_scores()
        weights = {}
        for t in self.model.territorios:
            ivc_w = scores[t]
            hist  = self._improvement_history[t]
            if self._year >= 2 and len(hist) >= 2:
                improvement_rate  = max(0.01, hist[-1] - hist[-2])
                efficiency_penalty = 1.0 / (1.0 + improvement_rate * 5)
            else:
                efficiency_penalty = 1.0
            weights[t] = ivc_w * efficiency_penalty

        total_w    = sum(weights.values()) or 1.0
        allocation = {t: (weights[t] / total_w) * self.budget_total
                      for t in self.model.territorios}
        return allocation

    def step(self) -> dict[str, float]:
        self._year += 1
        for t in self.model.territorios:
            wb = self.model.poblacion_agents[t].wellbeing
            self._improvement_history[t].append(wb)

        if self.strategy == "focalizado":
            allocation = self.allocate_focalizado()
        elif self.strategy == "distribuido":
            allocation = self.allocate_distribuido()
        elif self.strategy == "adaptativo":
            allocation = self.allocate_adaptativo()
        else:
            allocation = self.allocate_distribuido()

        for t in self.model.territorios:
            self.model.poblacion_agents[t].receive_investment(allocation[t])
            self.model.infraestructura_agents[t].receive_investment(allocation[t])

        return allocation


# ─── Model ────────────────────────────────────────────────────────────────────

class UrbanModel:
    """
    Mesa-compatible Urban Investment Model.
    Works with Mesa 2.x and 3.x; does not rely on scheduler or datacollector.

    Territory list and agent count are derived at runtime from ivc_data,
    so the model runs correctly regardless of whether the city has 16 or 20
    territorial units.
    """

    def __init__(
        self,
        strategy: str,
        ivc_data: dict[str, float],
        n_years: int = N_YEARS,
        annual_budget: float = ANNUAL_BUDGET,
        seed: int = 42,
    ):
        random.seed(seed)
        np.random.seed(seed)
        self.strategy      = strategy
        self.n_years       = n_years
        self.annual_budget = annual_budget
        self.current_step  = 0

        # Territory list and count — 100% data-driven, never hardcoded
        self.territorios   = sorted(ivc_data.keys())
        self.n_territorios = len(self.territorios)

        logger.info(
            f"  UrbanModel({strategy}): {self.n_territorios} {UT}s loaded from data"
        )

        # Instantiate agents
        uid = 0
        self.gobierno = GobiernoAgent(uid, self, strategy)
        uid += 1

        self.poblacion_agents:      dict[str, PoblacionAgent]      = {}
        self.infraestructura_agents: dict[str, InfrastructuraAgent] = {}

        for t in self.territorios:
            ivc_init = ivc_data.get(t, 0.35)
            self.poblacion_agents[t]      = PoblacionAgent(uid, self, t, ivc_init)
            uid += 1
            self.infraestructura_agents[t] = InfrastructuraAgent(uid, self, t, ivc_init)
            uid += 1

        self._records    = []
        self._ivc_initial = dict(ivc_data)

    def step(self):
        """Execute one simulation year."""
        self.current_step += 1
        allocation = self.gobierno.step()

        for t in self.territorios:
            self.poblacion_agents[t].step()
            self.infraestructura_agents[t].step()

        for t in self.territorios:
            pa = self.poblacion_agents[t]
            ia = self.infraestructura_agents[t]
            # Column name for territory uses UT (dynamic) so downstream scripts
            # can join on the same key without hardcoding "comuna"
            self._records.append({
                "scenario":              self.strategy,
                "year":                  self.current_step,
                UT:                      t,
                "ivc_initial":           self._ivc_initial.get(t, 0.35),
                "investment_allocated":  allocation[t],
                "wellbeing":             pa.wellbeing,
                "ivc_current":           pa.ivc,
                "infrastructure_capacity": ia.capacity,
                "cumulative_investment": sum(
                    r["investment_allocated"]
                    for r in self._records
                    if r[UT] == t
                ) + allocation[t],
            })

    def run_simulation(self) -> pd.DataFrame:
        """Run all n_years steps and return results DataFrame."""
        logger.info(
            f"Running scenario: '{self.strategy}' for {self.n_years} years "
            f"({self.n_territorios} {UT}s) ..."
        )
        for _ in range(self.n_years):
            self.step()
        logger.info(
            f"Scenario '{self.strategy}' complete. Records: {len(self._records)}"
        )
        return pd.DataFrame(self._records)


# ─── Main execution ───────────────────────────────────────────────────────────

def main():
    CSV_PATH = os.path.join(_abs.datos_sociales, "processed", "indices_comunas.csv")
    OUT_DIR  = _abs.procesados
    os.makedirs(OUT_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"ABM Urban Investment Model — {cfg.ciudad.nombre}")
    logger.info(f"  Unidad territorial : {UT}  ({UT_COL} column)")
    logger.info("=" * 60)

    ivc_data = load_ivc_from_csv(CSV_PATH)
    n_t = len(ivc_data)
    logger.info(f"IVC data loaded for {n_t} {UT}s.")
    for t, v in sorted(ivc_data.items(), key=lambda x: -x[1]):
        logger.info(f"  {t:<28}: IVC={v:.4f}")

    all_results = []
    for scenario in SCENARIOS:
        model = UrbanModel(strategy=scenario, ivc_data=ivc_data, seed=42)
        df    = model.run_simulation()
        all_results.append(df)

    results   = pd.concat(all_results, ignore_index=True)
    out_path  = os.path.join(OUT_DIR, "simulation_results.csv")
    results.to_csv(out_path, index=False)
    logger.info(f"Results saved: {out_path}")
    logger.info(f"Total rows: {len(results)}")

    logger.info(f"\n-- Year-{N_YEARS} Mean Wellbeing by Scenario --")
    year_n = results[results["year"] == N_YEARS]
    for scenario in SCENARIOS:
        mw = year_n[year_n["scenario"] == scenario]["wellbeing"].mean()
        logger.info(f"  {scenario:<15}: {mw:.4f}")

    return results


if __name__ == "__main__":
    main()
