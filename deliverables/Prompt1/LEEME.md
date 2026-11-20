# Prompt 1 — Sobol

## Qué se cambió

El análisis de Sobol del ABM usaba el diseño de Saltelli con N = 128 (1.152 evaluaciones, términos de segundo orden apagados). El índice total del tamaño del conjunto prioritario, en el escenario focalizado y para el CV del IVC, salía en 1,49. Ese estimador no está acotado a [0, 1] y con esa muestra el intervalo era demasiado ancho.

Ahora N = 1024 (9.216 evaluaciones). `calc_second_order` es False en el muestreo y en el análisis, porque el artículo reporta S1 y ST y no S2. La semilla 42 fija el modelo y los 1.000 remuestreos bootstrap. El intervalo al 95 % es el percentil 2,5–97,5 de esos remuestreos. El cómputo tardó 49,2 segundos. El ST de ese parámetro pasó a 0,95 (intervalo 0,80–1,11). Los 63 índices tienen un intervalo que intersecta 0 ≤ S1 ≤ ST ≤ 1, así que no se subió N por encima de 1024. Siete estimaciones puntuales quedan apenas fuera de ese rango (S1 negativo del orden de 0,004 o menos, o S1 mayor que ST por menos de 0,00002 en efectos casi nulos); el intervalo sí cubre un valor válido.

## Archivos de esta carpeta

- `table_07_sobol_indices_v2.csv` — tabla de índices, CSV, sin convertir a hoja de cálculo.
- `table_07_sobol_indices_v2.docx` — la misma tabla en Word editable, no en Google Docs.
- `fig_06_sobol_cv_v2.pdf` — figura vectorial (S1 y ST del CV, con barras del intervalo).
- `fig_06_sobol_cv_v2.tif` — la misma figura en TIFF a 600 dpi, compresión LZW.
- `sobol_update.md` — valores nuevos contra N = 128, tiempo de cómputo y el párrafo en inglés para el artículo.

En el repositorio los mismos productos están en `article_outputs/` (`table_sobol_indices.csv`, `table_sobol_indices.docx`, `fig_sobol_cv.pdf`, `fig_sobol_cv.tiff`) y en `docs/sobol_update.md`.

## Pull request

https://github.com/gjriveral/smart-city-decision-platform/pull/5

Esta carpeta es `Entregables_UrbanScience/Prompt1_Sobol`, dentro del directorio de trabajo. Ese directorio se sincroniza solo con Google Drive; no hace falta subir nada por el conector. Los nombres son `_v2`, así que no reemplazan archivos que ya estuvieran en Drive.

## Problemas pendientes

El intervalo del ST del tamaño del conjunto prioritario todavía llega a 1,11: el estimador de Jansen no está acotado y la estimación puntual (0,95) ya está dentro de [0, 1].
