# LEEME — Prompt 5, correcciones Urban Science

## 1. Inventario que faltaba (Prompt 0)

Se escribió `docs/data_inventory.md` y se copió aquí como `data_inventory.md`. El Prompt 0 no se había entregado: no había carpeta `Entregables_UrbanScience/Prompt0_…`.

El inventario cubre, con ruta local, años, grano territorial y si el archivo es real u oficial: el índice (IVC, panel de la Encuesta de Calidad de Vida); la inversión ejecutada (`data/external/inversion_ejecutada_oficial_territorio_anio.csv`, más `_inv_raw` y `_inv_probe`); el POAI territorializado; el presupuesto participativo; Medellín Cómo Vamos; la población DANE; y la lluvia SIATA. Dice con claridad lo que no está.

## 2. Signo del CV

Se corrigió el signo en las notas y en los encabezados de las tablas de dispersión y de Monte Carlo. Un valor positivo es una reducción del CV. El paso de 0,188 a 0,059 es +68,3 de reducción, no un aumento. Un aumento de dispersión va con signo negativo (el ranking fijo del diseño es −126,9). El encabezado usado es “Reduction in CV (%)”.

No se reestimaron los escenarios. Los archivos del Prompt 2 y del Prompt 4 se dejaron como estaban. Los corregidos, con numeración del artículo, son:

- `table_04_abm_dispersion_v3.csv` y `.docx` (γ calibrado)
- `table_05_abm_monte_carlo_v3.csv` y `.docx`
- `table_S4_design_dispersion_v3.csv` y `.docx` y `table_S5_design_monte_carlo_v3.csv` y `.docx` (sensibilidad con γ = 0,05)

La nota breve está en `figures_tables_changes_v2.md`.

## 3. Numeración y calibración dentro de los resultados

`manuscript_UrbanScience_v2.docx` y `manuscript_UrbanScience_v2.pdf` (este último exportado desde Word a partir del DOCX; no se convirtió a Google Docs).

- Tabla 4 y Figura 4: diez años con γ = 0,0144. La nota incluye: “calibrated index response; official executed investment 2013–2020 and observed index; not a synthetic series”.
- Tabla 5: Monte Carlo calibrado.
- Tabla 6: lluvia. Tabla 7: Sobol. Tabla 8: control de lluvia. Tabla 9: métricas de calibración y validación (`table_09_calibration_metrics_v2.csv` y `.docx`).
- Figura 7: `fig_07_calibration_validation_v2.pdf` y `.tif`. Son copia de los archivos v1 del Prompt 4. No se regeneraron y no se rasterizaron.
- `fig_04_abm_cv_v3.pdf` y `.tif` son copia de la figura calibrada v2 del Prompt 4.
- En Resultados el escenario principal es el calibrado (43,0 %; ranking fijo 14,0 %). El diseño con γ = 0,05 (68,3 %; ranking fijo −126,9 %) queda como sensibilidad, en el texto y en las tablas suplementarias.
- El resumen tiene 179 palabras, en un párrafo y sin citas. El detalle frente a v1 está en `changes_UrbanScience_v2.md`.
- Se conservó la subsección 5.3, el cierre MDPI y la declaración de IA. El Data Availability Statement sigue diciendo que Zenodo 10.5281/zenodo.21766975 es v2.0.0 y es anterior al Sobol N = 1024 y a las tablas de lluvia y de calibración.

## La calibración usa datos reales

La calibración no usa una serie sintética. Usa la inversión ejecutada oficial de la Alcaldía de Medellín, 2013–2020 (21 territorios, 168 filas, sin imputar; 2019 = MEData 1-002-11-000278, con la salvedad de que el campo de corte dice 31/12/20 y la vigencia declarada es 2019) y el índice observado. γ = 0,0144 (intervalo bootstrap 0,0083–0,0218). La partición es 6 de 8 transiciones (75/25). En validación 2020–2021 la persistencia gana (RMSE 0,042 frente a 0,049). No se rellenó 2021–2024.

## Datos que aún faltan, por prioridad

No se descargó nada y no se pidió nada a nombre del autor.

1. (A) Inversión ejecutada georreferenciada, de año cerrado y legible por máquina, 2021–2024, en MEData o en Planeación.
2. (B) POAI territorializado.
3. (C) Presupuesto participativo por territorio y año.
4. (D) Medellín Cómo Vamos.
5. (E) Población DANE por comuna y corregimiento.

## Comprobación de dpi

Pillow leyó los TIFF sueltos antes de esta entrega:

- `fig_04_abm_cv_v3.tif`: 3767 × 2836 px, dpi (600, 600), RGB.
- `fig_07_calibration_validation_v2.tif`: 4317 × 3042 px, dpi (600, 600), RGB.

Los PDF sueltos de esas dos figuras no contienen un objeto de imagen y sí contienen fuentes: siguen siendo vectoriales. Las figuras dentro del DOCX van incrustadas a tamaño legible.

## Referencias

No se inventaron referencias. Siguen siendo 57, con los DOI ya verificados. No se reutilizaron 10.7591/9780801460982, 10.1177/0020852308095311 ni 10.3828/idpr.2015.14.

El resumen tiene 179 palabras.

Esta carpeta es Entregables_UrbanScience/Prompt5_Correcciones, dentro del directorio de trabajo que sincroniza con https://drive.google.com/drive/u/0/folders/1t3a20nixkH2IsaaEHzBVO_eiPbqpH6Hp. Los nombres van versionados para no reemplazar entregas previas.
