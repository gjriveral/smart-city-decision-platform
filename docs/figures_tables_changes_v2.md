# Nota de signo y numeración (v2)

Fecha: 20 de noviembre de 2026. No se reestimaron escenarios, Monte Carlo, Sobol ni la calibración.

## El error de signo

En las tablas sueltas del Prompt 2 y en las calibradas del Prompt 4, la columna se llamaba “Change in CV (%)” y la nota decía que un valor positivo significaba que la dispersión había aumentado. Esa frase es falsa. El valor guardado es una reducción: (CV en el año 0 − CV en el año 10) / CV en el año 0, en porcentaje.

Ejemplo del diseño con γ = 0,05: el CV pasa de 0,188 a 0,059. Eso es una reducción de +68,3 %, no un aumento. El ranking fijo de ese mismo diseño sube el CV hasta 0,425; la reducción firmada es −126,9. Un valor positivo es una caída del CV. Un valor negativo es un aumento de la dispersión.

Los archivos `_v2` de `deliverables/Prompt2/` y `deliverables/Prompt4/` no se tocaron. Las cadenas de las notas en `scripts/build_publication_assets.py` (`NOTE_04`, `NOTE_05`) y en `scripts/calibrate_abm.py` se corrigieron para que una ejecución futura no vuelva a imprimir la frase falsa. Esos dos scripts no se ejecutaron en este paso, porque volverían a escribir los entregables `_v2`.

## Archivos nuevos, con el encabezado “Reduction in CV (%)”

| Archivo | Qué es | Origen, sin reestimar |
| --- | --- | --- |
| `table_04_abm_dispersion_v3.csv` / `.docx` | Tabla 4 del artículo. Diez años con γ = 0,0144 | Prompt 4, `table_05_abm_dispersion_v2` |
| `table_05_abm_monte_carlo_v3.csv` / `.docx` | Tabla 5. Monte Carlo calibrado | Prompt 4, `table_06_abm_monte_carlo_v2` |
| `table_09_calibration_metrics_v2.csv` / `.docx` | Tabla 9. RMSE, MAE, R² frente a persistencia, y Spearman 2020 y 2021 | Prompt 4, métricas de calibración y validación |
| `table_S4_design_dispersion_v3.csv` / `.docx` | Tabla suplementaria. Diseño γ = 0,05. No es el resultado principal | Prompt 2, `table_04` |
| `table_S5_design_monte_carlo_v3.csv` / `.docx` | Tabla suplementaria. Monte Carlo del diseño | Prompt 2, `table_05` |

En el Monte Carlo calibrado de partes iguales, el archivo de origen guardaba −0,0. Se reporta 0,0. El rango sigue siendo −0,6 a 0,5.

La nota de la Tabla 4 incluye la frase: “calibrated index response; official executed investment 2013–2020 and observed index; not a synthetic series”.

## Numeración del artículo

- Tabla 4 y Figura 4: escenarios a 10 años con γ calibrado (0,0144).
- Tabla 5: Monte Carlo calibrado.
- Tabla 6: lluvia (sigue siendo la Tabla 6 del Prompt 2; no se renombró como Tabla 6 del Prompt 4).
- Tabla 7: Sobol, N = 1024. Los intervalos que superan 1 se conservan. El estimador de Jansen no está acotado. No se reestimó.
- Tabla 8: control de calidad de la lluvia.
- Tabla 9: métricas de calibración y validación.
- Figura 7: validación. `fig_07_calibration_validation_v2.pdf` y `.tif` son copia de los archivos v1 del Prompt 4. No se rasterizó el PDF.
- `fig_04_abm_cv_v3.pdf` y `.tif` son copia de `fig_04_abm_cv_v2` (γ calibrado). No se regeneró la figura.

Comprobación con Pillow y con el contenido del PDF, hecha al copiar:

- `fig_04_abm_cv_v3.tif`: dpi (600, 600).
- `fig_07_calibration_validation_v2.tif`: dpi (600, 600).
- Ambos PDF no contienen un XObject de imagen y sí contienen fuentes: son vectoriales.
