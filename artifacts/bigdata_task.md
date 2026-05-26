# 📊 Big Data SIATA Pipeline — Task Checklist
**Project:** Inversión Estratégica en Ciudades · UdeA TG 2026  
**Pipeline:** Red Pluviométrica SIATA → Medellín Comunas  
**Executed:** 2026-05-25 18:04 COT  
**Mode:** SAMPLE_MODE = True (10 / 184 files)

---

## Phase 1 — Data Inspection [DONE]

- [x] Inspect CSV structure (Estacion_pluviometrica_10_2014-07-23_2026-02-28.csv)
- [x] Confirm column names: codigo, fecha_hora, p1, p2, calidad
- [x] Confirm separator: comma
- [x] Confirm encoding: UTF-8 / ASCII
- [x] Identify quality flags: calidad=1 (valid), 2, 1510, 1511, 1512 (invalid)
- [x] Identify precipitation unit: p1 = mm per 1-min interval (0.254 mm = 1 bucket tip)
- [x] Confirm station ID format: codigo column = integer
- [x] Count total files: 184 CSV files totalling ~33.6 GB

## Phase 2 — Directory Structure [DONE]

- [x] bigdata/ created
- [x] data/processed/ created
- [x] artifacts/ created
- [x] logs/ created

## Phase 3 — Script bigdata/01_siata_processor.py [DONE]

- [x] Embedded commune centroids (16 comunas WGS84)
- [x] Embedded station catalog (~80 stations)
- [x] Haversine distance for nearest-commune assignment
- [x] Deterministic hash fallback for unknown stations
- [x] Dask lazy read with dd.read_csv() + pandas fallback
- [x] Quality filter: calidad == 1 only
- [x] Year extraction from fecha_hora
- [x] GroupBy aggregation (estacion_id, año, comuna)
- [x] Metrics: precip_media_mm, precip_max_mm, precip_p95_mm, n_registros
- [x] Output: data/processed/siata_agregado.csv
- [x] Logging to logs/bigdata_execution.log + stdout
- [x] SAMPLE_MODE = True (10 files)

## Phase 4 — Execution SAMPLE_MODE=True [DONE]

Files processed: 10 / 184
Valid rows (calidad==1): ~54.3 million
Date range: 2013-11-15 to 2026-02-28
Unique stations: 10
Output rows: 100
Estimated run time (sample): ~195 seconds

Station-Commune Mapping:
  Station 10  -> El Poblado     [catalog]
  Station 11  -> Popular        [catalog]
  Station 12  -> Aranjuez       [catalog]
  Station 14  -> La Candelaria  [catalog]
  Station 15  -> La America     [catalog]
  Station 121 -> Aranjuez       [catalog]
  Station 127 -> Buenos Aires   [hash fallback]
  Station 129 -> Villa Hermosa  [hash fallback]
  Station 146 -> Doce de Octubre [hash fallback]
  Station 154 -> San Javier     [hash fallback]

## Phase 5 — Output Files [DONE]

- [x] data/processed/siata_agregado.csv (100 rows, 7 columns)
- [x] logs/bigdata_execution.log
- [x] artifacts/bigdata_task.md (this file)

NOTE: precip_p95_mm = 0.0 is expected at 1-minute resolution.
>95% of 1-minute records are zero-precipitation in any climate.
For meaningful P95, aggregate to daily totals first.

## Phase 6 — PENDING (Full Run)

- [ ] Set SAMPLE_MODE = False in bigdata/01_siata_processor.py
- [ ] Run full pipeline on all 184 files (~33.6 GB, est. 60-90 min)
- [ ] Load official SIATA coordinates for hash-assigned stations
- [ ] Consider daily resampling before P95 calculation
- [ ] Validate commune assignments vs. official SIATA metadata

## Quality Flag Warning

calidad values observed: 1, 2, 1510, 1511, 1512
Current filter (calidad == 1) may exclude valid data.
Consider calidad < 10 or consult SIATA codebook.

## Files Created

model_strategic_investment_cities/
  bigdata/01_siata_processor.py
  bigdata/run_sample.py
  data/processed/siata_agregado.csv
  logs/bigdata_execution.log
  artifacts/bigdata_task.md
