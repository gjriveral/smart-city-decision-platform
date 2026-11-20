# Cambios del manuscrito v2 respecto de v1

Manuscrito anterior: `manuscript_UrbanScience_v1.docx` (Prompt 3).
Manuscrito nuevo: `manuscript_UrbanScience_v2.docx`. No se añadió ninguna referencia y no se alteró ningún DOI.

## Conteo

- Palabras del resumen de v2 (un solo párrafo, sin citas): **179** (límite 200). El resumen de v1 tenía 184.
- Referencias numeradas, en orden de aparición: **57**, las mismas de v1.
- Palabras de la lista de referencias: **1458**.
- Artículos de Urban Science (2023–2026): **8**.

## Qué cambió en el cuerpo

- El resumen ya no destaca “68 to 69 percent”. La cifra principal es la del escenario con γ calibrado: la lista reescrita cada año reduce el CV 43,0 % y la lista fija lo reduce 14,0 %. El párrafo dice que, en los años retenidos, la persistencia es más precisa que la trayectoria calibrada.
- La Tabla 4 y la Figura 4 dejan de ser la simulación con γ = 0,05. Son los diez años con γ = 0,0144, tomados de la calibración del Prompt 4, con la nota de signo corregida. La nota dice que la serie es inversión ejecutada oficial 2013–2020 y el índice observado, no una serie sintética.
- La Tabla 5 es el Monte Carlo calibrado. La media de partes iguales se reporta como 0,0 (el archivo de origen guardaba −0,0).
- La Tabla 6 sigue siendo la comparación de lluvia. La Tabla 7 es Sobol (N = 1024, sin reestimar; los intervalos que superan 1 se conservan porque el estimador de Jansen no está acotado). La Tabla 8 es el control de calidad de la lluvia. No se renombró la lluvia como Tabla 6 del Prompt 4.
- Se añadió la Figura 7 (copia de `fig_07_calibration_validation_v1`; el PDF sigue siendo vectorial y el TIFF sigue a 600 dpi). No se rasterizó.
- Se añadió la Tabla 9, con RMSE, MAE y R² frente a la persistencia, y Spearman 0,979 (2020) y 0,897 (2021).
- El diseño con γ = 0,05 (68,3 % y ranking fijo −126,9 %) pasó a las Tablas suplementarias S4 y S5. En el texto es comparación de sensibilidad, no el resultado principal.
- Se reescribieron 4.3, 5.2, 5.4 y las Conclusiones para que los números coincidan con esas tablas. La partición se declara como 6 de 8 transiciones (75/25).
- La subsección 5.3 se conservó. Se corrigió la frase que decía que la lista congelada eleva el CV: con γ calibrado esa lista lo reduce 14,0 %. La inversión de signo queda solo en el diseño (Tabla S4).
- Se conservaron el cierre MDPI y la declaración de uso de IA (Grok y los agentes de Cursor). El Data Availability Statement sigue diciendo que el depósito de Zenodo 10.5281/zenodo.21766975 es la versión v2.0.0 y es anterior a la muestra Sobol N = 1024, a las tablas de lluvia y a las tablas de calibración.
- Los títulos del cuerpo usan los estilos Heading 1 y Heading 2 de Word. El número de sección va en el texto del título; el estilo no añade un segundo número.
- Autores sin cambio: Gabriel Jaime Rivera León (ORCID 0000-0001-5317-9313) y Yony Fernando Ceballos, Universidad de Antioquia (ORCID 0000-0001-5787-8832).

## Lo que no se hizo

- No se reestimó Sobol, ni los escenarios, ni la calibración.
- No se inventaron datos ni referencias. No se reutilizaron 10.7591/9780801460982, 10.1177/0020852308095311 ni 10.3828/idpr.2015.14.
- No se rellenó la inversión de 2021–2024.
- No se envió el artículo.
