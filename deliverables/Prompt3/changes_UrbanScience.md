# Cambios para Urban Science

Manuscrito de origen: `manuscript_SEPS_full.docx` (envío a Socio-Economic Planning Sciences, noviembre de 2026).
Manuscrito nuevo: `manuscript_UrbanScience_v1.docx`, en inglés, para un número regular de Urban Science (MDPI).

## Conteo

- Palabras del resumen (un solo párrafo): **184** (límite 200).
- Referencias numeradas, en orden de aparición: **57**.
- Palabras de la lista de referencias: **1458**.
- Artículos de Urban Science (2023–2026) citados: **8**.

## Por sección

- Título. El título pasa de “territorial equity” y “explainable prioritization” a un problema de justicia espacial. El modelo de asignación es el aporte; XGBoost y SHAP quedan como capa de diagnóstico. Tres títulos alternativos están en `LEEME.md`.
- Resumen. Un solo párrafo, sin citas, con contexto, métodos, resultados y conclusión. Se corrigen las cifras del pronóstico al panel agrupado (persistencia 0.821, bosque aleatorio 0.704, XGBoost 0.667, regresión lineal 0.488; n = 168). Se retira la frase que atribuía a la lluvia un R² de 0.732 para XGBoost: la tabla actualizada no reproduce ese aumento.
- Introducción. Reescrita como problema de gobernanza urbana y justicia espacial. El modelo basado en agentes es la contribución central. La capa de aprendizaje automático explica el índice y aporta una lista fija de contraste. Se eliminan las preguntas (i)–(iii) y las contribuciones “First / Second / Third / Fourth”.
- Marco teórico (sección 2, nueva). Justicia espacial (Soja, Marcuse, Fainstein, Campbell), presupuesto participativo (Cabannes, Sintomer, Goldfrank, y el procedimiento de Medellín) y urbanismo social (Brand y Dávila, Maclean, Anguelovski y coautores, Sotomayor, Franz). La subsección 2.3 conserva el puente hacia el diagnóstico, el protocolo ODD y el análisis de Sobol.
- Métodos. Se elimina “legacy model”. El modelo del mismo año se describe como reconstrucción del índice (media ponderada exacta; residuo máximo 1.1 × 10⁻¹⁶; R² lineal = 1). El Sobol pasa de N = 128 (1,152 evaluaciones, ST = 1.49 en la muestra pequeña) a N = 1024 (9,216 evaluaciones, intervalos percentiles de 1,000 remuestreos, semilla 42). Se añade una cota calibrada de γ (0.014; intervalo 0.008–0.022) para que las tablas de diseño no se lean como pronóstico. Las ecuaciones (1)–(8) se conservan.
- Resultados. Numeración de las tablas del Prompt 2: Tabla 1 pronóstico agrupado; Tabla 2 por año; Tabla 3 leave-one-territory-out; Tabla 4 dispersión; Tabla 5 Monte Carlo; Tabla 6 lluvia; Tabla 7 Sobol del coeficiente de variación; Tabla 8 control de calidad de la lluvia. Figuras 1 a 6 insertadas. La columna de cambio del coeficiente de variación se rotula como reducción: un valor positivo significa que el coeficiente bajó. La nota del Prompt 2 que decía lo contrario no se copió, porque contradice los valores (0.188 → 0.059 con entrada +68.3). La comparación de lluvia usa 126 territorio-años, no 168, y la persistencia sigue por encima de los modelos ajustados. No hay rutas de archivos en las notas.
- Discusión. La derrota de XGBoost frente a la persistencia se presenta como hallazgo útil: el insumo operativo es el índice observado más reciente. Nueva subsección “Implications for urban governance”, en prosa, ligada al presupuesto participativo y a la justicia espacial. Las limitaciones dejan de ser una lista (i)–(viii). Se retira la afirmación de que no había inversión por territorio. La reducción del 68 por ciento queda explícita como simulación con γ = 0.05; con γ calibrado la reducción del escenario focalizado anual es 43.0 por ciento, y en 2020–2021 la persistencia gana fuera de muestra (RMSE 0.042 frente a 0.049).
- Cierre MDPI. Author Contributions (CRediT), Funding, Institutional Review Board Statement (Not applicable), Informed Consent Statement (Not applicable), Data Availability Statement (GitHub y Zenodo), Acknowledgments con el uso de Grok y de los agentes de Cursor, y Conflicts of Interest.
- Referencias. De 33 a 57, numeradas [n] por orden de aparición. Cada DOI citado se resolvió en doi.org antes de incluirlo. Los identificadores que resolvieron a otro trabajo no se citaron (véase el LEEME).
- Voz. Se eliminaron “rather than”, las enumeraciones “First/Second” y las listas (i)–(viii) del cuerpo del artículo.
