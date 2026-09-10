# © 2026 Gabriel Jaime Rivera León. Todos los derechos reservados.
# El diseño arquitectónico, modelado lógico, simulaciones y estructura de este
# software son obra original del autor. La lógica computacional implementada
# comprende pipelines ETL, índices de vulnerabilidad, ensamble XGBoost con
# validación temporal y simulación basada en agentes (ABM).

"""
04_urban_model.py
=================
Urban Agent-Based Model for Strategic Investment (Python puro; sin paquete mesa).
- GobiernoAgent      : allocation strategies (focalizado, distribuido, adaptativo)
- PoblacionAgent     : wellbeing dynamics per territorial unit
- InfrastructuraAgent: infrastructure capacity dynamics per territorial unit

Three scenarios x N_YEARS simulation years → simulation_results.csv

Integración XGBoost → ABM (acoplamiento predictivo–simulación):
  El escenario «focalizado» ya NO prioriza solo el IVC histórico. Carga
  `models/shap_values.csv` (salida de `ml/03_xgboost_model.py`) y asigna
  presupuesto según el peso de atribución SHAP por unidad territorial:
  la interpretabilidad del motor predictivo dirige la heurística de inversión.

Configuración de ciudad:
  La lista de unidades territoriales y sus IVC iniciales se cargan
  dinámicamente desde el CSV del pipeline ML o, como fallback, desde
  config.yaml. El nombre de la columna de unidad territorial se lee de
  config.yaml vía config_loader.
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

# ABM secuencial en Python puro (sin dependencia del paquete mesa).

# ─── Simulation parameters (desde config.yaml → simulacion) ───────────────────
ANNUAL_BUDGET = float(cfg.simulacion.presupuesto_anual)
N_YEARS       = int(cfg.simulacion.horizonte_anios)
SCENARIOS     = list(cfg.simulacion.escenario_ids)
N_REPLICAS_MC = int(cfg.simulacion.n_replicas_montecarlo)
SCENARIO_META = {e.id: e for e in cfg.simulacion.escenarios}
# ρ por defecto del focalizado = peso declarado en config
RHO_FOCALIZADO_DEFAULT = float(
    cfg.simulacion.peso_escenario("focalizado", default=0.80)
)
# Fuente de IVC inicial para el ABM:
#   "predicho"  → ivc_predicho OOS del XGBoost (models/ivc_predictions_oos.csv)
#   "historico" → ivc observado de indices_comunas.csv
# Override por CLI: py abm/04_urban_model.py --fuente predicho|historico
FUENTE_IVC = "predicho"

_UT_HEADER_CANDIDATES = [
    UT_COL, "Comuna", "Localidad", "Corregimiento",
    "Unidad_Territorial", "unidad_territorial",
]


def _find_ut_header(columns) -> str | None:
    for candidate in _UT_HEADER_CANDIDATES:
        if candidate in columns:
            return candidate
    return None


# ─── Dynamic IVC loader ───────────────────────────────────────────────────────
def load_ivc_from_csv(csv_path: str) -> dict[str, float]:
    """
    Loads IVC initial values from indices_comunas.csv (latest year per territory).
    Returns a dict {territory_name: ivc_value} for ALL territories found in the CSV.
    Falls back to config.yaml territorios with IVC=0.35 if CSV is unavailable.
    """
    fallback = {t.nombre: 0.35 for t in cfg.territorios}

    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        df.columns = [c.strip() for c in df.columns]

        ut_col_found = _find_ut_header(df.columns)

        if ut_col_found is None or "ivc" not in df.columns or "Año" not in df.columns:
            logger.warning(
                f"CSV missing expected columns ('{UT_COL}' / 'ivc' / 'Año'). "
                f"Using config.yaml fallback ({len(fallback)} {UT}s)."
            )
            return fallback

        # Take the last available year for each territory
        latest = df.sort_values("Año").groupby(ut_col_found).last().reset_index()
        ivc_map = dict(
            zip(
                latest[ut_col_found].astype(str).str.strip(),
                latest["ivc"].astype(float),
            )
        )
        logger.info(
            f"IVC loaded from CSV: {len(ivc_map)} {UT}s "
            f"(years {df['Año'].min()}–{df['Año'].max()})"
        )
        return ivc_map

    except Exception as e:
        logger.warning(f"Could not load CSV ({e}). Using config.yaml fallback.")
        return fallback


def load_ivc_from_predictions(
    csv_path: str,
    historico_fallback: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Carga IVC inicial desde models/ivc_predictions_oos.csv (predicciones OOS
    del XGBoost). Por territorio toma ivc_predicho del año más reciente.

    Misma firma de retorno que load_ivc_from_csv: dict {territorio: ivc}.

    Si un territorio no aparece en el archivo OOS (p. ej. solo estuvo en el
    bloque de entrenamiento del primer pliegue), se usa el valor histórico
    de historico_fallback / load_ivc_from_csv y se registra un warning.
    """
    if historico_fallback is None:
        hist_path = os.path.join(
            _abs.datos_sociales, "processed", "indices_comunas.csv"
        )
        historico_fallback = load_ivc_from_csv(hist_path)

    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        df.columns = [c.strip() for c in df.columns]
    except Exception as e:
        logger.warning(
            f"No se pudo leer ivc_predictions_oos.csv ({e}). "
            f"Fallback total a IVC histórico ({len(historico_fallback)} {UT}s)."
        )
        return dict(historico_fallback)

    ut_col_found = _find_ut_header(df.columns)
    if ut_col_found is None or "ivc_predicho" not in df.columns:
        logger.warning(
            f"ivc_predictions_oos.csv sin columnas esperadas "
            f"('{UT_COL}' / 'ivc_predicho'). Fallback a histórico."
        )
        return dict(historico_fallback)
    if "Año" not in df.columns:
        logger.warning(
            "ivc_predictions_oos.csv sin columna 'Año'. Fallback a histórico."
        )
        return dict(historico_fallback)

    latest = (
        df.sort_values("Año")
        .groupby(ut_col_found, as_index=False)
        .last()
    )
    ivc_map: dict[str, float] = dict(
        zip(
            latest[ut_col_found].astype(str).str.strip(),
            latest["ivc_predicho"].astype(float),
        )
    )
    year_min, year_max = int(df["Año"].min()), int(df["Año"].max())
    logger.info(
        f"IVC predicho (OOS) cargado: {len(ivc_map)} {UT}s "
        f"(años OOS {year_min}–{year_max}; valor = ivc_predicho del año más reciente)"
    )

    # Territorios del histórico sin predicción OOS → fallback puntual
    for t, v_hist in historico_fallback.items():
        if t not in ivc_map:
            ivc_map[t] = float(v_hist)
            logger.warning(
                f"  Sin predicción OOS para '{t}' "
                f"(solo en entrenamiento del primer pliegue o ausente). "
                f"Fallback histórico IVC={v_hist:.4f}"
            )

    return ivc_map


# Columnas de contribución SHAP producidas por ml/03_xgboost_model.py
# (mismas features del XGBRegressor; no incluyen Comuna/Año).
_SHAP_FEATURE_COLS = [
    "idx_desempleo",
    "idx_habitat",
    "idx_educacion",
    "idx_riesgo_clima",
    "idx_pobreza",
    "idx_tejido_social",
    "año_normalizado",
]


def load_shap_priority(shap_path: str) -> dict[str, float]:
    """
    Carga shap_values.csv (artefacto del XGBoost) y construye un peso de
    priorización por unidad territorial (detalle territorio-año).

    Importante — separación de roles de los artefactos SHAP:
      • shap_importance_cv_stability.csv + ranking_by_mean = fuente OFICIAL
        del ranking de *determinantes* (variables) validado OOS por CV.
        NO se usa aquí: este loader no elige "el determinante dominante".
      • shap_values.csv = atribución in-sample por territorio-año del modelo
        al 100 %. Se mantiene SOLO como señal de detalle espacial para
        allocate_focalizado (suma firmada de SHAP por comuna), no como
        ranking headline de features.

    Integración XGBoost→ABM (focalizado):
      Para cada unidad se toma el último año disponible y se suma el valor
      CON SIGNO de las contribuciones SHAP de las variables predictoras.
    """
    try:
        df = pd.read_csv(shap_path, encoding="utf-8-sig")
        df.columns = [c.strip() for c in df.columns]

        ut_col_found = _find_ut_header(df.columns)
        if ut_col_found is None or "Año" not in df.columns:
            logger.warning(
                f"shap_values.csv sin columna de territorio/'Año'. "
                f"Integración SHAP→focalizado desactivada."
            )
            return {}

        feat_cols = [c for c in _SHAP_FEATURE_COLS if c in df.columns]
        if not feat_cols:
            logger.warning(
                "shap_values.csv no contiene columnas de features SHAP "
                "reconocidas. Integración SHAP→focalizado desactivada."
            )
            return {}

        # Último año por territorio = señal de interpretabilidad vigente
        latest = df.sort_values("Año").groupby(ut_col_found).last().reset_index()
        # Peso = suma CON SIGNO de SHAP_j (misma señal que el bucle original).
        # SHAP > 0 empuja el IVC predicho hacia arriba (más vulnerable);
        # SHAP < 0 lo empuja hacia abajo. Usar |SHAP| priorizaría por
        # intensidad de atribución en cualquier dirección (p. ej. El
        # Poblado tendría alto peso por contribuciones negativas fuertes),
        # lo cual contradice el objetivo de focalizar inversión en
        # vulnerabilidad. La suma firmada es la señal correcta para que
        # la interpretabilidad del XGBoost dirija el presupuesto.
        names = latest[ut_col_found].astype(str).str.strip()
        weights = latest[feat_cols].astype(float).sum(axis=1)
        priority = dict(zip(names, weights.astype(float)))

        logger.info(
            f"SHAP priority loaded from {shap_path}: {len(priority)} {UT}s "
            f"(features usadas: {feat_cols}; score = suma firmada de SHAP)"
        )
        top5 = sorted(priority.items(), key=lambda kv: -kv[1])[:5]
        for t, w in top5:
            logger.info(f"  SHAP-priority top: {t:<28}: SHAP_sum={w:.4f}")
        return priority

    except Exception as e:
        logger.warning(
            f"No se pudo cargar shap_values.csv ({e}). "
            f"El escenario focalizado usará IVC como fallback."
        )
        return {}


# ─── Agent Definitions ────────────────────────────────────────────────────────

class PoblacionAgent:
    """Represents the population of one territorial unit."""

    def __init__(
        self,
        unique_id,
        model,
        territory_id: str,
        ivc_initial: float,
        beta: float = 0.15,
        gamma: float = 0.05,
    ):
        self.unique_id   = unique_id
        self.model       = model
        self.territory_id = territory_id
        self.ivc         = ivc_initial
        self.wellbeing   = max(0.0, min(1.0, 1.0 - ivc_initial))
        self.investment_received   = 0.0
        self._pending_investment   = 0.0
        # Coeficientes estructurales (defaults = literales históricos del POM)
        self.beta  = beta   # β_u: tasa mejora wellbeing
        self.gamma = gamma  # γ_u: tasa reducción IVC

    def receive_investment(self, amount: float):
        self._pending_investment = amount

    def step(self):
        self.investment_received = self._pending_investment
        # avg_inv derived from model's actual territory count — no hardcoded 16
        avg_inv   = self.model.annual_budget / max(self.model.n_territorios, 1)
        inv_ratio = self.investment_received / avg_inv if avg_inv > 0 else 0
        # Factor de eficiencia decreciente η(C) según capacidad de infraestructura
        # ya instalada en el mismo territorio (estado al inicio del periodo):
        #   η(C) = min(1.0, 1.5 - C) ,  C = capacity ∈ [0, 1]
        # Justificación: rendimientos decrecientes — cuando C es baja (≤ 0.5),
        # η = 1 (inversión a plena eficacia); cuando C > 0.5, cada peso
        # adicional aporta menos a bienestar/IVC porque la planta instalada
        # ya absorbe buena parte de las necesidades básicas. La forma lineal
        # acotada es cóncava en el tramo operativo (C↑ ⇒ η↓) y evita
        # sobre-corregir territorios con déficit de infraestructura.
        cap = self.model.infraestructura_agents[self.territory_id].capacity
        eff = min(1.0, 1.5 - cap)
        inv_ratio = inv_ratio * eff
        # Perturbación estocástica (ε_t ~ N(1, 0.05²), truncada a [0.85, 1.15]).
        # ANTES: noise solo multiplicaba el delta de wellbeing; la actualización
        # del IVC era determinista en inv_ratio. Como focalizado/distribuido no
        # leen wellbeing, sus réplicas Monte Carlo tenían IC95 de ancho cero.
        # AHORA: el mismo término ε_t escala la tasa de reducción del IVC, de
        # modo que la vulnerabilidad hereda varianza en CADA escenario.
        noise     = max(0.85, min(1.15, 1.0 + random.gauss(0, 0.05)))
        d_wb      = self.beta * inv_ratio * (1.0 - self.wellbeing) * noise
        self.wellbeing = max(0.0, min(1.0, self.wellbeing + d_wb))
        # Transición estocástica del IVC:
        #   IVC_{t+1} = IVC_t * (1 - γ_u * inv_ratio * ε_t)
        #   (inv_ratio ya incluye η(C))
        self.ivc = max(
            0.0,
            min(1.0, self.ivc * (1.0 - self.gamma * inv_ratio * noise)),
        )
        self._pending_investment = 0.0


class InfrastructuraAgent:
    """Represents infrastructure capacity of one territorial unit."""

    DECAY_RATE = 0.03  # valor base δ_u (default de delta)

    def __init__(
        self,
        unique_id,
        model,
        territory_id: str,
        ivc_initial: float,
        delta: float = 0.03,
        theta: float = 0.10,
    ):
        self.unique_id    = unique_id
        self.model        = model
        self.territory_id = territory_id
        self.capacity     = max(0.1, min(0.9, 1.0 - ivc_initial))
        self.maintenance_cost      = ivc_initial * 50
        self.investment_received   = 0.0
        self._pending_investment   = 0.0
        # Coeficientes estructurales (defaults = DECAY_RATE / literal 0.10)
        self.delta = delta  # δ_u: depreciación de capacidad
        self.theta = theta  # θ_u: recuperación por inversión

    def receive_investment(self, amount: float):
        self._pending_investment = amount

    def step(self):
        self.investment_received = self._pending_investment
        avg_inv      = self.model.annual_budget / max(self.model.n_territorios, 1)
        inv_normalized = self.investment_received / avg_inv if avg_inv > 0 else 0
        self.capacity  = max(0.0, self.capacity - self.delta)
        self.capacity  = min(1.0, self.capacity + self.theta * inv_normalized)
        self._pending_investment = 0.0


class GobiernoAgent:
    """Single government agent that allocates budget across all territorial units."""

    def __init__(self, unique_id, model, strategy: str, rho: float = RHO_FOCALIZADO_DEFAULT):
        self.unique_id    = unique_id
        self.model        = model
        self.strategy     = strategy
        self.budget_total = model.annual_budget
        self._year        = 0
        # rho: fraccion del presupuesto concentrada en las top-5 unidades mas
        # vulnerables bajo el escenario "focalizado" (antes constante 0.80,
        # ahora parametrizada para permitir el barrido de sensibilidad de rho
        # exigido en el analisis doctoral; el valor por defecto preserva el
        # comportamiento original del modelo).
        self.rho = rho
        # History indexed by territory — derived from model's dynamic territory list
        self._improvement_history = {t: [] for t in self.model.territorios}

    def _get_ivc_scores(self) -> dict[str, float]:
        return {t: self.model.poblacion_agents[t].ivc for t in self.model.territorios}

    def _get_shap_scores(self) -> dict[str, float]:
        """
        Pesos de priorización provenientes del XGBoost (SHAP).
        Si el modelo no cargó shap_values.csv, retorna {} y el llamador
        debe caer al ranking por IVC (fallback explícito).
        """
        shap_map = getattr(self.model, "shap_priority", None) or {}
        return {t: float(shap_map.get(t, 0.0)) for t in self.model.territorios}

    def allocate_focalizado(self) -> dict[str, float]:
        """
        Escenario Focalizado — Optimización Algorítmica (integrado con XGBoost).

        Antes: ordenaba por IVC histórico corriente y repartía rho en partes
        iguales entre las top-5 unidades más vulnerables.

        Ahora (acoplamiento XGBoost→ABM):
          1. Obtiene el peso SHAP por unidad (suma de |contribuciones SHAP|
             del modelo predictivo sobre el IVC predicho).
          2. Ordena las unidades por ese peso (la interpretabilidad del
             XGBoost define quién es prioritario).
          3. Destina `rho` del presupuesto a las top-5, repartido de forma
             PROPORCIONAL al peso SHAP (no en partes iguales): mayor
             atribución SHAP ⇒ mayor inversión.
          4. El remanente (1-rho) se reparte por igual entre el resto.

        Si shap_values.csv no está disponible, cae al ranking por IVC
        (comportamiento legacy) y lo registra en el log una sola vez.
        """
        shap_scores = self._get_shap_scores()
        # Hay señal SHAP usable si al menos una unidad tiene score distinto
        # de cero (la suma firmada puede ser negativa en unidades de baja
        # vulnerabilidad; eso es informativo, no un fallo de carga).
        using_shap = bool(getattr(self.model, "shap_priority", None)) and any(
            t in (self.model.shap_priority or {}) for t in self.model.territorios
        )

        if using_shap:
            scores = shap_scores
        else:
            if not getattr(self, "_warned_no_shap", False):
                logger.warning(
                    "Focalizado: sin pesos SHAP válidos — fallback a ranking "
                    "por IVC histórico (integración XGBoost desactivada)."
                )
                self._warned_no_shap = True
            scores = self._get_ivc_scores()

        sorted_t = sorted(scores, key=scores.get, reverse=True)
        n_top = min(5, len(sorted_t))
        top_n = sorted_t[:n_top]
        rest = sorted_t[n_top:]
        pool_top = self.rho * self.budget_total
        pool_rest = (1.0 - self.rho) * self.budget_total

        # Dentro del pool focalizado: reparto PROPORCIONAL al peso SHAP.
        # Solo se usan scores positivos (contribución neta al alza del IVC);
        # si por redondeo algún top quedara en 0, se asigna epsilon para no
        # excluirlo del pool concentrado.
        top_weights = {t: max(float(scores[t]), 0.0) + 1e-9 for t in top_n}
        weight_sum = sum(top_weights.values())

        allocation: dict[str, float] = {}
        for t in self.model.territorios:
            if t in top_n:
                allocation[t] = pool_top * (top_weights[t] / weight_sum)
            else:
                allocation[t] = pool_rest / max(len(rest), 1)
        return allocation

    def allocate_distribuido(self) -> dict[str, float]:
        """Equal allocation to all territorial units."""
        equal = self.budget_total / max(self.model.n_territorios, 1)
        return {t: equal for t in self.model.territorios}

    def allocate_adaptativo(self) -> dict[str, float]:
        """
        Heurística original del autor (no es mezcla de Focalizado y Distribuido).
        Asigna el presupuesto en proporción al IVC corriente penalizado por la
        mejora reciente del bienestar: φ = 1/(1+5g), con g acotado inferiormente
        en 0,01 y activación desde el tercer año. Los literales 5, 0,01 y el
        umbral de 2 años son parámetros de diseño del autor, no valores de
        una librería ABM.
        """
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
    Urban Investment Model (simulación secuencial en Python puro).
    No usa scheduler ni datacollector de frameworks ABM externos.

    Territory list and agent count are derived at runtime from ivc_data.
    Opcionalmente recibe `shap_priority` (pesos de interpretabilidad del
    XGBoost) para que el escenario focalizado asigne inversión según SHAP
    y no solo según el IVC histórico — este es el acoplamiento XGBoost→ABM.
    """

    def __init__(
        self,
        strategy: str,
        ivc_data: dict[str, float],
        n_years: int = N_YEARS,
        annual_budget: float = ANNUAL_BUDGET,
        seed: int = 42,
        rho: float = RHO_FOCALIZADO_DEFAULT,
        shap_priority: dict[str, float] | None = None,
        beta: float = 0.15,
        gamma: float = 0.05,
        delta: float = 0.03,
        theta: float = 0.10,
    ):
        random.seed(seed)
        np.random.seed(seed)
        self.strategy      = strategy
        self.n_years       = n_years
        self.annual_budget = annual_budget
        self.current_step  = 0
        # Coeficientes estructurales (valores base POM; sobrescribibles
        # para análisis de sensibilidad en abm/sensibilidad_coeficientes.py)
        self.beta  = beta    # β_u — mejora de wellbeing
        self.gamma = gamma   # γ_u — reducción de IVC
        self.delta = delta   # δ_u — depreciación de infraestructura
        self.theta = theta   # θ_u — recuperación de infraestructura

        # Territory list and count — 100% data-driven, never hardcoded
        self.territorios   = sorted(ivc_data.keys())
        self.n_territorios = len(self.territorios)
        self._cum_inv      = {t: 0.0 for t in self.territorios}

        # Pesos SHAP del XGBoost (integración predictivo → simulación).
        # Se almacenan en el modelo para que GobiernoAgent.allocate_focalizado
        # los lea en cada paso sin volver a tocar el CSV.
        self.shap_priority = dict(shap_priority or {})
        n_shap = sum(1 for t in self.territorios if t in self.shap_priority)
        logger.info(
            f"  UrbanModel({strategy}): {self.n_territorios} {UT}s loaded from data"
            f" | SHAP priorities matched: {n_shap}/{self.n_territorios}"
        )

        # Instantiate agents
        uid = 0
        self.gobierno = GobiernoAgent(uid, self, strategy, rho=rho)
        uid += 1

        self.poblacion_agents:      dict[str, PoblacionAgent]      = {}
        self.infraestructura_agents: dict[str, InfrastructuraAgent] = {}

        for t in self.territorios:
            ivc_init = ivc_data.get(t, 0.35)
            self.poblacion_agents[t] = PoblacionAgent(
                uid, self, t, ivc_init, beta=self.beta, gamma=self.gamma
            )
            uid += 1
            self.infraestructura_agents[t] = InfrastructuraAgent(
                uid, self, t, ivc_init, delta=self.delta, theta=self.theta
            )
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
            self._cum_inv[t] = self._cum_inv.get(t, 0.0) + allocation[t]
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
                "cumulative_investment": self._cum_inv[t],
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

def _resolve_fuente(cli_fuente: str | None) -> str:
    fuente = (cli_fuente or FUENTE_IVC or "predicho").strip().lower()
    if fuente not in ("predicho", "historico"):
        logger.warning(
            f"FUENTE_IVC='{fuente}' no válida; se usa 'predicho'."
        )
        return "predicho"
    return fuente


def _run_all_scenarios(
    ivc_data: dict[str, float],
    shap_priority: dict[str, float],
    fuente_label: str,
) -> pd.DataFrame:
    all_results = []
    for scenario in SCENARIOS:
        rho_sc = cfg.simulacion.peso_escenario(scenario, default=RHO_FOCALIZADO_DEFAULT)
        model = UrbanModel(
            strategy=scenario,
            ivc_data=ivc_data,
            seed=42,
            rho=rho_sc if scenario == "focalizado" else RHO_FOCALIZADO_DEFAULT,
            shap_priority=shap_priority,
        )
        df = model.run_simulation()
        df = df.copy()
        df["fuente_ivc"] = fuente_label
        all_results.append(df)
    return pd.concat(all_results, ignore_index=True)


def main(fuente: str | None = None):
    # Validación estricta de nombres territorio (config ↔ indices_comunas.csv)
    _abm_dir = str(Path(__file__).resolve().parent)
    if _abm_dir not in sys.path:
        sys.path.insert(0, _abm_dir)
    from validar_territorios_config import validate_territory_names
    validate_territory_names(raise_on_error=True)

    CSV_PATH = os.path.join(_abs.datos_sociales, "processed", "indices_comunas.csv")
    PRED_PATH = os.path.join(_abs.modelos, "ivc_predictions_oos.csv")
    # Artefacto de interpretabilidad del XGBoost: acopla el motor predictivo
    # con la heurística de asignación del escenario focalizado.
    SHAP_PATH = os.path.join(_abs.modelos, "shap_values.csv")
    OUT_DIR  = _abs.procesados
    os.makedirs(OUT_DIR, exist_ok=True)

    fuente_default = _resolve_fuente(fuente)

    logger.info("=" * 60)
    logger.info(f"ABM Urban Investment Model — {cfg.ciudad.nombre}")
    logger.info(f"  Unidad territorial : {UT}  ({UT_COL} column)")
    logger.info(f"  Horizonte (años)   : {N_YEARS}  [config.simulacion]")
    logger.info(f"  Presupuesto anual  : {ANNUAL_BUDGET:,.0f}  [config.simulacion]")
    logger.info(f"  Escenarios         : {SCENARIOS}")
    logger.info(f"  Réplicas MC (cfg)  : {N_REPLICAS_MC}")
    logger.info(f"  SHAP (XGBoost→ABM) : {SHAP_PATH}")
    logger.info(f"  FUENTE_IVC default : {fuente_default}")
    logger.info(f"  Predicciones OOS   : {PRED_PATH}")
    logger.info("=" * 60)

    # Histórico siempre disponible (fallback + corrida comparativa)
    ivc_historico = load_ivc_from_csv(CSV_PATH)
    ivc_predicho = load_ivc_from_predictions(
        PRED_PATH, historico_fallback=ivc_historico
    )

    fuentes = {
        "historico": ivc_historico,
        "predicho": ivc_predicho,
    }
    # Fuente primaria según constante / CLI
    ivc_data = fuentes[fuente_default]
    shap_priority = load_shap_priority(SHAP_PATH)

    n_t = len(ivc_data)
    logger.info(f"IVC data (fuente={fuente_default}) loaded for {n_t} {UT}s.")
    for t, v in sorted(ivc_data.items(), key=lambda x: -x[1]):
        shap_w = shap_priority.get(t, float("nan"))
        v_alt = fuentes["historico" if fuente_default == "predicho" else "predicho"].get(
            t, float("nan")
        )
        logger.info(
            f"  {t:<28}: IVC_{fuente_default}={v:.4f}  "
            f"IVC_alt={v_alt:.4f}  SHAP_priority={shap_w:.4f}"
        )

    # Simulación completa con AMBAS fuentes (comparación tesis)
    results_by_fuente: dict[str, pd.DataFrame] = {}
    for fuente_name, ivc_map in fuentes.items():
        logger.info(f"\n--- Simulación con fuente_ivc='{fuente_name}' ---")
        results_by_fuente[fuente_name] = _run_all_scenarios(
            ivc_map, shap_priority, fuente_name
        )
        out_named = os.path.join(
            OUT_DIR, f"simulation_results_{fuente_name}.csv"
        )
        results_by_fuente[fuente_name].to_csv(out_named, index=False)
        logger.info(
            f"Results saved: {out_named} "
            f"({len(results_by_fuente[fuente_name])} rows)"
        )

    # Archivo canónico = fuente por defecto (compatibilidad con scripts previos)
    results = results_by_fuente[fuente_default]
    out_path = os.path.join(OUT_DIR, "simulation_results.csv")
    results.to_csv(out_path, index=False)
    logger.info(f"Canonical results (FUENTE_IVC={fuente_default}): {out_path}")

    # Resumen comparativo wellbeing t=N y media IVC t=N
    logger.info(f"\n-- Comparación fuentes (year={N_YEARS}) --")
    for fuente_name, res in results_by_fuente.items():
        year_n = res[res["year"] == N_YEARS]
        for scenario in SCENARIOS:
            sub = year_n[year_n["scenario"] == scenario]
            mw = sub["wellbeing"].mean()
            mi = sub["ivc_current"].mean()
            logger.info(
                f"  [{fuente_name:<10}] {scenario:<15}: "
                f"wellbeing={mw:.4f}  ivc_medio={mi:.4f}"
            )

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ABM Urban Investment Model")
    parser.add_argument(
        "--fuente",
        choices=["predicho", "historico"],
        default=None,
        help="Fuente de IVC inicial (default: constante FUENTE_IVC)",
    )
    args = parser.parse_args()
    main(fuente=args.fuente)
