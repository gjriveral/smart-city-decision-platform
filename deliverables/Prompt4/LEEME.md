# Prompt 4 — Calibración del ABM

## Qué se hizo

Se calibró la respuesta del índice de vulnerabilidad a la inversión pública relativa. La inversión ejecutada no estaba en el repositorio: se agregó la serie oficial de la Alcaldía de Medellín con corte al 31 de diciembre, de 2013 a 2020, a las 21 unidades territoriales. El índice observado cubre 2013–2024; 2021–2024 no entran en la pérdida porque no hay un archivo anual cerrado y legible por máquina para esos años.

De las ocho transiciones posibles, las seis primeras (inversión 2013–2018) calibran y las dos últimas (2020 y 2021 del índice) validan. El único parámetro identificado es γ, la respuesta del índice. El mínimo del RMSE de la trayectoria da γ = 0,0144 (intervalo bootstrap al 95 %: 0,0083–0,0218), frente al valor de diseño 0,05. Semilla 42 en el bootstrap y en el muestreo ABC.

En calibración el RMSE es 0,049, mejor que la persistencia (0,055). En validación el RMSE es 0,049 y la persistencia gana (0,042). El orden de los territorios en los años retenidos coincide con la persistencia. β, δ, θ, la regla de reparto y la lluvia no entran en esa transición con la saturación apagada: no se les asignó un valor de la literatura como si estuviera estimado. La elasticidad del capital público de Bom y Ligthart (2014, doi:10.1111/joes.12037) no es γ y no se usó.

Los escenarios y el Monte Carlo se volvieron a correr con γ = 0,0144. La baja del CV a diez años en la regla focalizada pasó de 68,3 % a 43,0 %. Con el ranking fijo, el CV subía 126,9 % con γ = 0,05 y baja 14,0 % con el valor calibrado.

## Archivos de esta carpeta

- `table_calibration_parameters_v1.csv` y `.docx` — parámetros, con intervalo solo para γ.
- `table_calibration_metrics_v1.csv` y `.docx` — RMSE, MAE y R² en calibración y validación, contra persistencia.
- `table_calibration_patterns_v1.csv` y `.docx` — dispersión y correlación de Spearman en 2020 y 2021.
- `table_05_abm_dispersion_v2.csv` y `.docx` — dispersión a diez años con γ calibrado.
- `table_06_abm_monte_carlo_v2.csv` y `.docx` — 30 réplicas con ruido.
- `fig_04_abm_cv_v2.pdf` y `.tif` — sendas del CV, PDF vectorial y TIFF a 600 dpi.
- `fig_07_calibration_validation_v1.pdf` y `.tif` — trayectorias simuladas y observadas en validación.
- `calibration_update.md` — datos, método, métricas y los párrafos en inglés para Methods y Results.
- `ODD_protocol.md` — descripción del modelo en la estructura ODD+D.

## Pull request

https://github.com/gjriveral/smart-city-decision-platform/pull/7

Esta carpeta es `Entregables_UrbanScience/Prompt4_Calibracion`, dentro del directorio de trabajo. Ese directorio se sincroniza solo con https://drive.google.com/drive/u/0/folders/1t3a20nixkH2IsaaEHzBVO_eiPbqpH6Hp. No se usó el conector de Drive. Los nombres llevan sufijo de versión, así que no reemplazan archivos que ya estuvieran en Drive.

## Problemas pendientes

La persistencia pronostica mejor los dos años retenidos que la trayectoria calibrada. β, δ, θ y una sensibilidad a la lluvia siguen sin identificarse. La inversión de 2021–2024 no está en un archivo anual cerrado y legible por máquina en la misma publicación. El campo de corte del archivo de 2019 dice 31/12/20, aunque la vigencia y la cobertura declarada son el año 2019.
