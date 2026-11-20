# Prompt 2 — Tablas y figuras (Urban Science)

## Qué se cambió

Las tablas y figuras del artículo quedaron numeradas para Urban Science (MDPI). Las notas ya no citan archivos, rutas ni una especificación anterior del pronóstico: cada nota dice "Source: authors' elaboration based on ...". La fuente es Arial; los R², errores, CV, Gini e índices de Sobol van a 3 decimales; los cambios porcentuales a 1 decimal; la lluvia al milímetro; la cobertura horaria a 1 decimal, en porcentaje.

La Tabla 7 y la Figura 6 usan el Sobol nuevo (N = 1024, intervalos bootstrap al 95 %). No se reestimó el pronóstico ni se volvió a correr el muestreo.

El detalle de cada tabla y figura está en `figures_tables_changes.md`.

## Archivos de esta carpeta

Tablas (DOCX editable y CSV), sufijo `_v2`:

- `table_01_forecast_metrics_v2` — pronóstico walk-forward agrupado.
- `table_02_forecast_by_year_v2` — el mismo pronóstico por año objetivo.
- `table_03_loto_metrics_v2` — leave-one-territory-out agrupado.
- `table_04_abm_dispersion_v2` — escenarios de asignación al año 10.
- `table_05_abm_monte_carlo_v2` — 30 réplicas con ruido.
- `table_06_rain_forecast_v2` — lluvia como predictor o dentro del índice.
- `table_07_sobol_indices_v2` — S1 y ST del CV, N = 1024.
- `table_08_rain_qc_by_year_v2` — cobertura y totales de lluvia por año.
- `table_S1_loto_by_territory_v2` — leave-one-territory-out por territorio.
- `table_S2_sobol_mean_ivc_gini_v2` — Sobol del índice medio y del Gini.
- `table_S3_abm_cv_path_v2` — sendas de la Figura 4.

Figuras (PDF vectorial y TIFF a 600 dpi), sufijo `_v2`:

- `fig_01_territories_v2` — mapa de las 21 unidades.
- `fig_02_forecast_r2_v2` — R² temporal y espacial.
- `fig_03_forecast_by_year_v2` — R² por año objetivo.
- `fig_04_abm_cv_paths_v2` — sendas del CV.
- `fig_05_rain_forecast_r2_v2` — lluvia y pronóstico.
- `fig_06_sobol_cv_v2` — índices de Sobol del CV.

También está `figures_tables_changes.md`.

## Pull request

Pendiente: el enlace del pull request se agrega al abrir el PR.

Esta carpeta es `Entregables_UrbanScience/Prompt2_Tablas_Figuras`, dentro del directorio de trabajo. Ese directorio se sincroniza solo con https://drive.google.com/drive/u/0/folders/1t3a20nixkH2IsaaEHzBVO_eiPbqpH6Hp. No se usó el conector de Drive. Los nombres llevan `_v2`, así que no reemplazan archivos que ya estuvieran en Drive.

## Problemas pendientes

El intervalo del ST del tamaño del conjunto prioritario, en la regla focalizada, todavía llega por encima de 1. La estimación puntual queda dentro de [0, 1]. La nota de la Tabla 7 reporta el intervalo del percentil, no un recorte a la unidad.
