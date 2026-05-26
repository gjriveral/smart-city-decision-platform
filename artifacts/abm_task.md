# ABM Task Checklist — Simulación Urban Investment Mesa
*Generado: 2026-05-25 18:31 UTC · Medellín, Colombia*

## Estado: ✅ COMPLETADO

---

## Dependencias

- [x] `mesa==3.5.1` — instalado (Python 3.14.x)
- [x] `matplotlib==3.10.9` — disponible
- [x] `seaborn==0.13.2` — disponible
- [x] `networkx` — instalado (dependencia de mesa)
- [x] `numpy==2.4.4` — disponible
- [x] `pandas==3.0.2` — disponible

---

## Agentes Implementados

- [x] **GobiernoAgent** — estrategias: focalizado, distribuido, adaptativo
  - [x] `allocate_focalizado()` — 80% top-5 IVC, 20% distribuido igual
  - [x] `allocate_distribuido()` — presupuesto equitativo 16 comunas
  - [x] `allocate_adaptativo()` — reponderación cada 2 años por IVC × efficiency_penalty
- [x] **PoblacionAgent** × 16 comunas — bienestar con retornos decrecientes
- [x] **InfrastructuraAgent** × 16 comunas — degradación 3%/año + mejora con inversión
- [x] **UrbanModel** — compatible Mesa 2.x y 3.x, sin dependencia del scheduler

---

## Condiciones Iniciales (desde indices_comunas.csv real)

| Commune | IVC Inicial |
|---------|------------|
| Santa Cruz | 0.5005 |
| Manrique | 0.4832 |
| Villa Hermosa | 0.4751 |
| Popular | 0.4732 |
| San Javier | 0.3990 |
| Aranjuez | 0.3936 |
| Robledo | 0.3809 |
| Buenos Aires | 0.3767 |
| Castilla | 0.3330 |
| La Candelaria | 0.3285 |
| Doce de Octubre | 0.3259 |
| Guayabal | 0.2940 |
| Laureles | 0.2897 |
| El Poblado | 0.2866 |
| Belén | 0.2778 |
| La América | 0.2454 |

---

## Ejecución de Escenarios

- [x] Escenario A: Focalizado — 10 años, 160 registros
- [x] Escenario B: Distribuido — 10 años, 160 registros
- [x] Escenario C: Adaptativo — 10 años, 160 registros
- [x] `simulation_results.csv` generado — 480 filas, 9 columnas, 59 KB

---

## Análisis y Reporte

- [x] Figura 1: Evolución bienestar promedio (línea temporal)
- [x] Figura 2: Heatmap bienestar comunas × escenarios (año 10)
- [x] Figura 3: Perfil asignación presupuestal (stacked bars)
- [x] Figura 4: Capacidad infraestructura en el tiempo (área)
- [x] `reports/scenario_report.html` — reporte dark-themed con plots base64

---

## Resultados — Año 10

| Escenario | Bienestar Medio | Infraestructura Media |
|-----------|----------------|----------------------|
| **A — Focalizado** ⭐ | **0.9321** | 0.9709 |
| C — Adaptativo | 0.9302 | 1.0000 |
| B — Distribuido | 0.9283 | 1.0000 |

**Escenario óptimo: FOCALIZADO** (mayor bienestar población, +0.0038 vs distribuido)

### Notas Técnicas
- Bienestar infraestructura es max (1.0) en escenarios B y C por asignación uniforme que cubre decay=3%/año
- Escenario A sacrifica ligeramente capacidad física en comunas no prioritarias a favor de mayor bienestar social
- Parámetro `noise = Gauss(0, 0.05)` introduce variabilidad realista; fijar seed=42 para reproducibilidad

---

## Archivos Generados

| Archivo | Ruta | Tamaño |
|---------|------|--------|
| `04_urban_model.py` | `abm/` | 13.9 KB |
| `05_scenario_analysis.py` | `abm/` | 22.5 KB |
| `run_abm_pipeline.py` | raíz | — |
| `simulation_results.csv` | `data/processed/` | 59 KB |
| `scenario_report.html` | `reports/` | ~5 MB (plots embebidos) |
| `abm_execution.log` | `logs/` | — |

---

## Criterios de Éxito ✅

- [x] Simulación corre los 3 escenarios sin errores fatales
- [x] `simulation_results.csv` con métricas completas (9 cols, 480 filas)
- [x] Reporte HTML generado con 4 gráficos embebidos
- [x] Escenario óptimo identificado: **FOCALIZADO**
