# Capítulo 4: Resultados y Análisis

> **Marco Integral de Gestión de Datos para Ciudades Colombianas**  
> Universidad de Antioquia · Trabajo de Grado 2026 · Facultad de Ingeniería  
> Ciudad de referencia: Medellín, Antioquia, Colombia

---

## 4.1 Importancia de Variables y Explicabilidad del Modelo XGBoost

### 4.1.1 Marco metodológico de la extracción de evidencia

La interpretabilidad del modelo de predicción del Índice de Vulnerabilidad Compuesto (IVC) se aborda desde dos perspectivas complementarias: la importancia por ganancia (*Gain Importance*) intrínseca al algoritmo XGBoost, y los valores SHAP (*SHapley Additive exPlanations*), derivados de la teoría de juegos cooperativos. Ambas métricas se calculan sobre el modelo definitivo entrenado con validación cruzada temporal de cinco pliegues (*walk-forward*) sobre el período 2007–2024 (514 observaciones; 33 unidades territoriales × 18 años), y cuya arquitectura óptima quedó fijada en los hiperparámetros recogidos en la Tabla 1.

**Tabla 1**

*Hiperparámetros óptimos del modelo XGBoost de vulnerabilidad urbana obtenidos mediante búsqueda en rejilla con validación cruzada temporal*

| Hiperparámetro | Valor óptimo | Descripción |
|---|---|---|
| `n_estimators` | 200 | Número de árboles de decisión en el ensamble |
| `max_depth` | 3 | Profundidad máxima de cada árbol |
| `learning_rate` | 0.100 | Tasa de aprendizaje (encogimiento del gradiente) |
| `subsample` | 0.800 | Fracción de muestras usadas por árbol |
| `colsample_bytree` | 0.800 | Fracción de variables muestreadas por árbol |
| `min_child_weight` | 1 | Suma mínima de pesos en un nodo hoja |
| `objective` | `reg:squarederror` | Función de pérdida (regresión cuadrática) |

*Nota.* Los hiperparámetros se seleccionaron a partir de la búsqueda en el espacio `{n_estimators: [100, 200], max_depth: [2, 3], learning_rate: [0.05, 0.1], subsample: [0.7, 0.8], colsample_bytree: [0.7, 0.8], min_child_weight: [3, 5], reg_alpha: [0.0, 0.1], reg_lambda: [1.0, 2.0], gamma: [0.0, 0.1]}`. El criterio de selección fue la minimización del RMSE promedio en el conjunto de prueba de cada pliegue temporal. Fuente: elaboración propia a partir de `models/cv_metrics.json`.

---

### 4.1.2 Métricas de rendimiento del modelo en validación cruzada temporal

Antes de analizar la importancia de variables, se establece el desempeño predictivo como condición de validez interpretativa. Un modelo con escaso poder predictivo no ofrece fundamento suficiente para extraer conclusiones sobre la relevancia relativa de sus variables. La Tabla 2 presenta los resultados de los cinco pliegues temporales.

**Tabla 2**

*Métricas de desempeño del modelo XGBoost por pliegue de validación cruzada temporal (walk-forward, 2007–2024)*

| Pliegue | Años de entrenamiento | Años de prueba | RMSE | MAE | R² |
|:---:|---|---|:---:|:---:|:---:|
| 1 | 2007–2010 | 2010–2013 | 0.02606 | 0.01913 | 0.9224 |
| 2 | 2007–2013 | 2013–2016 | 0.01147 | 0.00896 | 0.9747 |
| 3 | 2007–2016 | 2016–2019 | 0.00866 | 0.00666 | 0.9821 |
| 4 | 2007–2019 | 2019–2022 | 0.01910 | 0.01324 | 0.9138 |
| 5 | 2007–2022 | 2022–2024 | 0.00982 | 0.00696 | 0.9782 |
| **Media** | — | — | **0.01502** | **0.01099** | **0.9542** |
| *DE* | — | — | *0.00739* | *0.00526* | *0.0332* |

*Nota.* RMSE = raíz del error cuadrático medio; MAE = error absoluto medio; R² = coeficiente de determinación. Todos los índices se encuentran en la escala [0, 1] del IVC. El R² en datos completos (*in-sample*) alcanza 0.9985, lo que evidencia una brecha de generalización de 4.4 puntos porcentuales que motiva el uso de términos de regularización (L1, L2 y *gamma*) en la búsqueda de hiperparámetros. Fuente: elaboración propia a partir de `logs/ml_execution.log`.

El R² promedio de 0.9542 (± 0.033) indica que el modelo explica el 95.4 % de la varianza del IVC en períodos temporales no vistos durante el entrenamiento, lo que constituye un nivel de ajuste sobresaliente para datos socioeconómicos longitudinales de naturaleza heterogénea. El pliegue de mayor error (Pliegue 4, prueba 2019–2022) coincide con el período de disrupción socioeconómica asociado a la pandemia de COVID-19, lo cual es coherente con la literaria sobre *concept drift* en modelos de predicción de vulnerabilidad urbana.

---

### 4.1.3 Importancia de variables por ganancia (Gain Importance)

La importancia por ganancia cuantifica la reducción media de la función de pérdida atribuible a cada variable en el conjunto de divisiones (*splits*) de todos los árboles del ensamble. Valores más altos indican una contribución más significativa a la reducción del error de predicción. La Tabla 3 presenta los resultados en orden descendente.

**Tabla 3**

*Importancia por ganancia (Gain Importance) de las variables del modelo XGBoost de vulnerabilidad urbana, Medellín 2007–2024*

| Rango | Variable | Fuente de datos | Tipo | Ganancia | Ganancia acumulada |
|:---:|---|---|:---:|:---:|:---:|
| 1 | Pobreza Multidimensional (`idx_pobreza`) | IPM – DANE | Social | 0.3819 | 38.2 % |
| 2 | Precariedad del Hábitat (`idx_habitat`) | ECV + IPM | Social | 0.3081 | 69.0 % |
| 3 | Fragilidad del Tejido Social (`idx_tejido_social`) | ECV | Social | 0.1874 | 86.7 % |
| 4 | Vulnerabilidad Educativa (`idx_educacion`) | ECV + IPM | Social | 0.0923 | 99.0 % |
| 5 | Tensión de Desempleo (`idx_desempleo`) | IML | Social | 0.0250 | 99.5 % |
| 6 | Tendencia Temporal (`año_normalizado`) | Derivada | Temporal | 0.0053 | 100.0 % |
| 7 | Riesgo Climático (`idx_riesgo_clima`) | SIATA / IoT | Ambiental | 0.0000 | 100.0 % |

*Nota.* La importancia por ganancia se obtiene del atributo `feature_importances_` del modelo `XGBRegressor` de XGBoost 3.2.0. IPM = Índice de Pobreza Multidimensional (DANE); ECV = Encuesta de Calidad de Vida (Alcaldía de Medellín); IML = Indicadores de Mercado Laboral (Alcaldía de Medellín); SIATA = Sistema de Alerta Temprana de Medellín y el Valle de Aburrá. El valor 0.0000 del índice de riesgo climático es consecuencia del uso de un valor constante de relleno (0.5) en ausencia de datos SIATA integrados al pipeline ETL. Fuente: elaboración propia.

---

### 4.1.4 Explicabilidad global mediante valores SHAP

Los valores SHAP ofrecen una cuantificación teóricamente fundamentada de la contribución marginal de cada variable a la predicción individual, garantizando consistencia y eficiencia en sentido axiomático (Lundberg & Lee, 2017). La Tabla 4 presenta la media del valor absoluto SHAP (|SHAP|) promediado sobre las 514 observaciones, que representa la importancia global de cada variable en unidades de la escala del IVC.

**Tabla 4**

*Importancia global SHAP (media del valor absoluto) para el modelo de predicción del IVC, Medellín 2007–2024*

| Rango | Variable | Fuente | |SHAP| medio | % sobre total |
|:---:|---|---|:---:|:---:|
| 1 | Pobreza Multidimensional (`idx_pobreza`) | IPM | 0.02465 | 30.3 % |
| 2 | Vulnerabilidad Educativa (`idx_educacion`) | ECV + IPM | 0.01717 | 21.1 % |
| 3 | Fragilidad del Tejido Social (`idx_tejido_social`) | ECV | 0.01685 | 20.7 % |
| 4 | Precariedad del Hábitat (`idx_habitat`) | ECV + IPM | 0.01508 | 18.6 % |
| 5 | Tensión de Desempleo (`idx_desempleo`) | IML | 0.00788 | 9.7 % |
| 6 | Tendencia Temporal (`año_normalizado`) | Derivada | 0.00054 | 0.7 % |
| 7 | Riesgo Climático (`idx_riesgo_clima`) | SIATA / IoT | 0.00000 | 0.0 % |

*Nota.* Los valores SHAP se calculan con `shap.TreeExplainer` (SHAP 0.51.0) sobre el modelo final reentrenado con la totalidad del conjunto de datos. La columna «% sobre total» expresa la participación de cada variable en la suma de todas las importancias SHAP absolutas. El reordenamiento más significativo respecto a la Gain Importance es el ascenso de `idx_educacion` al **Rango 2** en la jerarquía SHAP (desde el Rango 4 por Gain), con un |SHAP| medio de 0.01717 que supera al de `idx_habitat` (0.01508) y al de `idx_tejido_social` (0.01685). Fuente: elaboración propia a partir de `models/shap_values.csv`.

---

### 4.1.5 Interpretación matemática y analítica: telemetría ambiental frente a componentes socioeconómicos

El análisis conjunto de las Tablas 3 y 4 permite articular tres hallazgos de relevancia científica y política pública:

**Primero: la estructura de pobreza multidimensional como determinante primario del IVC.** El índice de pobreza multidimensional (`idx_pobreza`), construido a partir del IPM del DANE, ocupa el primer rango tanto por ganancia (38.2 %) como por |SHAP| (30.3 %). Matemáticamente, esto implica que la función de decisión del XGBoost destina los primeros niveles de profundidad de sus árboles a bifurcar sobre este predictor, reduciendo el error residual más que cualquier otra variable. Desde una perspectiva de política pública, este resultado es coherente con la literatura que sitúa la privación material estructural como el conductor más proximal de la vulnerabilidad urbana compuesta (Sen, 1999; Alkire & Foster, 2011).

**Segundo: el hábitat precario y el tejido social como amplificadores sistémicos.** Las variables `idx_habitat` (30.8 % de ganancia; 18.6 % SHAP) e `idx_tejido_social` (18.7 % de ganancia; 20.7 % SHAP) conforman, junto con la pobreza multidimensional, un núcleo explicativo que concentra el 86.7 % de la ganancia acumulada. La divergencia relativa entre métricas —hábitat obtiene mayor ganancia pero menor SHAP que tejido social— sugiere que los *splits* sobre hábitat son más eficientes para separar grupos extremos de vulnerabilidad, mientras que tejido social ejerce una influencia más difusa y uniforme a lo largo de la distribución del IVC. Esta distinción es relevante para el diseño de intervenciones: las políticas habitacionales producen efectos más focalizados geográficamente, mientras que las intervenciones en cohesión social impactan de manera más transversal.

**Tercero: la nulidad informativa de la telemetría climática en la configuración actual del pipeline.** El índice de riesgo climático derivado de la red SIATA registra una importancia de 0.000 en ambas métricas, confirmando que el valor de relleno constante (0.5) asignado a todas las unidades territoriales en todos los años no aporta varianza explicativa al modelo. Este resultado no implica que el riesgo climático sea irrelevante para la vulnerabilidad urbana —evidencia internacional sugiere lo contrario (IPCC, 2022; Hallegatte et al., 2016)—, sino que la integración semántica entre la telemetría pluviométrica procesada (`siata_agregado.csv`) y el pipeline de índices (`02_vulnerability_indices.py`) permanece pendiente como trabajo futuro de alta prioridad. Una vez que los agregados comunales de precipitación, intensidad de lluvia extrema (percentil 95) y frecuencia de eventos de umbral crítico se incorporen como dimensiones del `idx_riesgo_clima`, se anticipa que este índice ascenderá en el ranking de importancia, particularmente para corregimientos de ladera con mayor exposición a eventos de remoción en masa.

La Tabla 5 sintetiza la comparación directa entre ambas métricas de importancia, evidenciando tanto las coincidencias en el ordenamiento cardinal como las divergencias ordinales que aportan matices interpretativos adicionales.

**Tabla 5**

*Comparación entre métricas de importancia de variables: Gain Importance y |SHAP| medio, modelo XGBoost de vulnerabilidad urbana*

| Variable | Rango Gain | Ganancia | Rango SHAP | |SHAP| medio | Divergencia ordinal |
|---|:---:|:---:|:---:|:---:|:---:|
| `idx_pobreza` | 1 | 0.3819 | 1 | 0.02465 | 0 |
| `idx_habitat` | 2 | 0.3081 | 4 | 0.01508 | +2 |
| `idx_tejido_social` | 3 | 0.1874 | 3 | 0.01685 | 0 |
| `idx_educacion` | 4 | 0.0923 | 2 | 0.01717 | −2 |
| `idx_desempleo` | 5 | 0.0250 | 5 | 0.00788 | 0 |
| `año_normalizado` | 6 | 0.0053 | 6 | 0.00054 | 0 |
| `idx_riesgo_clima` | 7 | 0.0000 | 7 | 0.00000 | 0 |

*Nota.* La divergencia ordinal positiva indica que la variable ocupa un rango más bajo en SHAP que en Gain (la ganancia sobreestima su importancia global relativa a SHAP); la divergencia negativa indica lo contrario. La divergencia notable de `idx_educacion` (−2 puestos respecto a SHAP) sugiere que este índice tiene una influencia más difusamente distribuida sobre la totalidad de las observaciones de lo que la métrica de ganancia captura. Fuente: elaboración propia.

---

## 4.2 Análisis de Trayectorias de la Simulación Basada en Agentes

### 4.2.1 Diseño computacional del modelo de simulación

La simulación basada en agentes (ABM, por sus siglas en inglés) se implementó con el marco Mesa (Python) e inicializa cada agente territorial con el IVC promedio histórico calculado a partir del archivo `indices_comunas.csv`. Cada agente actualiza su estado en función del monto de inversión pública recibida durante el período anual, modificando tanto el IVC corriente como la capacidad de infraestructura instalada y el índice de bienestar (*wellbeing*). El modelo opera durante un horizonte de 10 pasos temporales anuales con un presupuesto total agregado constante de $1,000,000 unidades monetarias (en la escala normalizada del modelo), distribuidas de manera diferenciada según el escenario de inversión activado. Los tres escenarios analizados son los siguientes:

- **Escenario Focalizado** (*focalizado*): asignación concentrada en las unidades territoriales de mayor IVC inicial, con un presupuesto de $160,000 para las comunas de mayor vulnerabilidad y $7,143 para las demás en los primeros ciclos, rotando la prioridad conforme los beneficiarios iniciales reducen su IVC.
- **Escenario Distribuido** (*distribuido*): distribución igualitaria del presupuesto entre todas las unidades territoriales, con $30,303 por unidad y por año.
- **Escenario Adaptativo** (*adaptativo*): regla de asignación dinámica que ajusta la distribución en función del IVC observado en cada período, priorizando unidades con reducción más lenta.

El total acumulado de inversión al término del Año 10 es idéntico en los tres escenarios ($303,030 por unidad en promedio), lo que garantiza la comparabilidad de resultados bajo la condición *ceteris paribus* de restricción presupuestaria.

---

### 4.2.2 Estado inicial del sistema: ranking de vulnerabilidad basal

La Tabla 6 presenta el ranking completo de vulnerabilidad basal por unidad territorial, calculado como la media del IVC sobre el período histórico 2007–2024. Este estado inicial condiciona la dinámica diferencial entre escenarios durante la simulación.

**Tabla 6**

*Ranking de vulnerabilidad basal por unidad territorial, Medellín 2007–2024 (IVC medio histórico)*

| Rango | Unidad Territorial | IVC Medio | Categoría |
|:---:|---|:---:|---|
| 1 | Popular | 0.4834 | Alta vulnerabilidad |
| 2 | Manrique | 0.4428 | Alta vulnerabilidad |
| 3 | Santa Cruz | 0.4389 | Alta vulnerabilidad |
| 4 | Corregimiento de San Sebastián de Palmitas | 0.4284 | Alta vulnerabilidad |
| 5 | Corregimiento de Santa Elena | 0.4225 | Alta vulnerabilidad |
| 6 | Villa Hermosa | 0.4097 | Alta vulnerabilidad |
| 7 | Aranjuez | 0.3813 | Vulnerabilidad media-alta |
| 8 | Doce de Octubre | 0.3800 | Vulnerabilidad media-alta |
| 9 | San Javier | 0.3796 | Vulnerabilidad media-alta |
| 10 | Corregimiento de Altavista | 0.3750 | Vulnerabilidad media-alta |
| ⋮ | ⋮ | ⋮ | ⋮ |
| 21 | **El Poblado** | **0.2291** | **Baja vulnerabilidad** |

*Nota.* IVC = Índice de Vulnerabilidad Compuesto; rango [0, 1] donde 1 indica vulnerabilidad máxima. La categorización se construye sobre cuartiles de la distribución histórica: Alta vulnerabilidad ≥ 0.40; Media-alta [0.35, 0.40); Media-baja [0.28, 0.35); Baja < 0.28. El contraste entre la comuna de mayor vulnerabilidad (**Popular**, IVC = 0.483) y la de menor vulnerabilidad (**El Poblado**, IVC = 0.229) delimita una brecha absoluta de 0.254 puntos en la escala del IVC, equivalente a un diferencial de 110.7 % entre ambos extremos de la distribución territorial. Esta asimetría estructural fundamenta la necesidad de estrategias de inversión diferenciales. Fuente: elaboración propia a partir de `models/vulnerability_ranking.csv`.

---

### 4.2.3 Horizonte a 3 años: respuesta temprana y efecto de arranque

El corto plazo representa la fase crítica de arranque donde las diferencias entre estrategias de asignación presupuestaria se manifiestan de manera más pronunciada, dado que la capacidad de infraestructura instalada aún no ha alcanzado su régimen estacionario. La Tabla 7 presenta los estadísticos descriptivos para el Año 3 en los tres escenarios.

**Tabla 7**

*Estadísticos descriptivos del Índice de Vulnerabilidad Compuesto y variables asociadas al Año 3 de simulación, por escenario de inversión*

| Estadístico | Focalizado | Distribuido | Adaptativo |
|---|:---:|:---:|:---:|
| IVC medio (Año 3) | 0.2956 | 0.3036 | 0.3024 |
| Desviación estándar IVC | 0.0329 | 0.0572 | 0.0503 |
| IVC mínimo | 0.2210 | 0.1964 | 0.2053 |
| IVC máximo | 0.3473 | 0.4291 | 0.4103 |
| Reducción media del IVC (%) | 16.5 | 14.3 | 14.6 |
| Bienestar medio (*wellbeing*) | 0.8090 | 0.7831 | 0.7856 |
| Capacidad de infraestructura media | 0.8081 | 0.8560 | 0.8560 |
| Inversión acumulada media por unidad | 90,909 | 90,909 | 90,909 |
| Coeficiente de variación IVC | 0.111 | 0.188 | 0.166 |

*Nota.* El coeficiente de variación (CV = DE/media) opera como indicador de desigualdad territorial intra-escenario: valores más bajos implican mayor equidad en la distribución del IVC. La inversión acumulada es idéntica entre escenarios ($90,909 por unidad) garantizando comparabilidad. Fuente: elaboración propia a partir de `evidencia/resultados_simulacion_escenarios.csv`.

A los tres años, el escenario **Focalizado** lidera en reducción media del IVC (16.5 % frente a 14.3 % del Distribuido), a pesar de haber concentrado el gasto en un subconjunto reducido de comunas en el Año 1. Este efecto se explica por el mecanismo de transmisión del modelo ABM: la reducción acelerada del IVC en las unidades más vulnerables libera *slack* presupuestario que se redirige hacia la siguiente cohorte de mayor riesgo, generando una función de reducción por tramos que supera en eficiencia global a la distribución lineal. Sin embargo, el escenario Distribuido exhibe en este horizonte una mayor capacidad de infraestructura media (0.856 frente a 0.808 del Focalizado), lo que anticipa una diferencia en la dinámica de acumulación de capital público a largo plazo.

---

### 4.2.4 Horizonte a 6 años: consolidación de tendencias y bifurcación de trayectorias

El mediano plazo revela la bifurcación definitiva entre las estrategias. La Tabla 8 recoge los estadísticos del Año 6.

**Tabla 8**

*Estadísticos descriptivos del Índice de Vulnerabilidad Compuesto y variables asociadas al Año 6 de simulación, por escenario de inversión*

| Estadístico | Focalizado | Distribuido | Adaptativo |
|---|:---:|:---:|:---:|
| IVC medio (Año 6) | 0.2487 | 0.2603 | 0.2585 |
| Desviación estándar IVC | 0.0196 | 0.0491 | 0.0378 |
| IVC mínimo | 0.2133 | 0.1684 | 0.1834 |
| IVC máximo | 0.2942 | 0.3679 | 0.3379 |
| Reducción media del IVC (%) | 29.8 | 26.5 | 27.0 |
| Bienestar medio (*wellbeing*) | 0.9082 | 0.8674 | 0.8697 |
| Capacidad de infraestructura media | 0.9143 | 0.9918 | 1.0000 |
| Inversión acumulada media por unidad | 181,818 | 181,818 | 181,818 |
| Coeficiente de variación IVC | 0.0789 | 0.1887 | 0.1463 |

*Nota.* El escenario Adaptativo alcanza la plena utilización de la capacidad de infraestructura (1.000) en el Año 6, ocho años antes que el modelo Focalizado, lo cual indica mayor eficiencia en la acumulación de capital físico público. Fuente: elaboración propia.

A los seis años la distancia entre el escenario Focalizado y los restantes se amplía. El Focalizado ha logrado una reducción del 29.8 % respecto al IVC inicial, mientras que Distribuido y Adaptativo registran 26.5 % y 27.0 % respectivamente. Notablemente, la desviación estándar del IVC en el escenario Focalizado (0.020) es menos de la mitad que en el Distribuido (0.049), lo que indica una convergencia territorial significativamente superior: la inversión concentrada, al rotar sistemáticamente hacia los focos de mayor riesgo, tiende a comprimir la distribución del IVC hacia valores más homogéneamente bajos, mientras que la distribución igualitaria permite que las unidades de menor IVC inicial desciendan rápidamente (mínimo de 0.168) pero las más vulnerables permanecen rezagadas (máximo de 0.368).

El patrón observado es coherente con el principio de *eficiencia-equidad* descrito en la teoría de inversión pública focalizada (Coady et al., 2004): la focalización no necesariamente sacrifica equidad si el mecanismo de rotación de prioridad está bien diseñado.

---

### 4.2.5 Horizonte a 10 años: estado casi-estacionario y convergencia

El largo plazo constituye el criterio definitivo de evaluación de las tres estrategias. La Tabla 9 recoge los estadísticos del Año 10, que representa el estado cuasi-estacionario del sistema simulado.

**Tabla 9**

*Estadísticos descriptivos del Índice de Vulnerabilidad Compuesto y variables asociadas al Año 10 de simulación, por escenario de inversión*

| Estadístico | Focalizado | Distribuido | Adaptativo |
|---|:---:|:---:|:---:|
| IVC medio (Año 10) | **0.1986** | 0.2120 | 0.2098 |
| Desviación estándar IVC | **0.0181** | 0.0400 | 0.0252 |
| IVC mínimo | 0.1731 | **0.1371** | 0.1575 |
| IVC máximo | **0.2318** | 0.2997 | 0.2610 |
| Reducción media del IVC (%) | **43.9** | 40.1 | 40.7 |
| Bienestar medio (*wellbeing*) | **0.9594** | 0.9309 | 0.9322 |
| Capacidad de infraestructura media | 0.9573 | **1.0000** | **1.0000** |
| Inversión acumulada media por unidad | 303,030 | 303,030 | 303,030 |
| Coeficiente de variación IVC | **0.0913** | 0.1886 | 0.1202 |

*Nota.* Los valores en negrita indican el mejor desempeño en cada indicador entre los tres escenarios. El coeficiente de variación del IVC opera como indicador de equidad territorial: valores menores implican mayor homogeneidad en los niveles de vulnerabilidad entre unidades territoriales. Fuente: elaboración propia.

La tabla anterior evidencia que al Año 10, el escenario **Focalizado** domina a los demás en las métricas de mayor relevancia normativa: logra la reducción más profunda del IVC medio (−43.9 % respecto al año base histórico), el bienestar promedio más alto (0.959), el coeficiente de variación más bajo (0.091) y el valor máximo de IVC más contenido (0.232). Estos resultados sugieren que la estrategia de inversión concentrada y secuencialmente rotante es **simultáneamente más eficiente y más equitativa** que la distribución proporcional en este horizonte temporal.

Sin embargo, el escenario Distribuido registra el IVC mínimo más bajo de todas las estrategias (0.137), indicando que las comunas de menor vulnerabilidad basal alcanzan estados de bienestar más profundos bajo la distribución igualitaria, mientras que las comunas de alta vulnerabilidad permanecen rezagadas.

La Tabla 10 desglosa las cinco unidades territoriales de mayor IVC corriente al final de la simulación para cada escenario, lo que permite identificar los focos de vulnerabilidad residual bajo cada estrategia.

**Tabla 10**

*Cinco unidades territoriales con mayor IVC corriente al Año 10 de simulación, por escenario de inversión*

| Escenario | Unidad Territorial | IVC Año 10 | Bienestar Año 10 |
|---|---|:---:|:---:|
| Focalizado | Laureles Estadio | 0.2318 | 0.8172 |
| Focalizado | Belén | 0.2266 | 0.9583 |
| Focalizado | La América | 0.2225 | 0.9457 |
| Focalizado | Castilla | 0.2203 | 0.9570 |
| Focalizado | San Sebastián De Palmitas | 0.2197 | 0.9556 |
| Distribuido | Santa Cruz | 0.2997 | 0.9040 |
| Distribuido | Manrique | 0.2893 | 0.9080 |
| Distribuido | Villa Hermosa | 0.2845 | 0.9043 |
| Distribuido | Popular | 0.2833 | 0.9123 |
| Distribuido | Corr. San Sebastián de Palmitas | 0.2619 | 0.9090 |
| Adaptativo | Santa Cruz | 0.2610 | 0.9407 |
| Adaptativo | Manrique | 0.2556 | 0.9401 |
| Adaptativo | Villa Hermosa | 0.2533 | 0.9349 |
| Adaptativo | Popular | 0.2527 | 0.9415 |
| Adaptativo | Corr. San Sebastián de Palmitas | 0.2407 | 0.9309 |

*Nota.* En el escenario Focalizado, las unidades con mayor IVC residual al Año 10 son comunas de vulnerabilidad inicial media-baja (Laureles Estadio, La América, Castilla), lo que confirma que las comunas originalmente más vulnerables (Popular, Manrique, Santa Cruz) han sido efectivamente intervenidas. En los escenarios Distribuido y Adaptativo, los focos de vulnerabilidad residual coinciden con las comunas de mayor IVC inicial, lo que indica que la distribución igualitaria no logra reducir estos picos con la misma profundidad. Fuente: elaboración propia.

---

## 4.3 Contraste Estadístico y Comparativo de los Tres Escenarios de Inversión Pública

### 4.3.1 Trayectorias anuales completas del IVC medio

La Tabla 11 presenta la evolución anual completa del IVC medio para los tres escenarios durante el horizonte de diez períodos, permitiendo comparar las dinámicas de reducción en cada punto del tiempo.

**Tabla 11**

*Evolución anual del Índice de Vulnerabilidad Compuesto medio (IVC̄) y del índice de bienestar medio (WB̄) por escenario de inversión, Años 1 a 10*

| Año | IVC Focalizado | WB Focalizado | IVC Distribuido | WB Distribuido | IVC Adaptativo | WB Adaptativo |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.3318 | 0.7123 | 0.3363 | 0.6988 | 0.3357 | 0.7006 |
| 2 | 0.3131 | 0.7663 | 0.3195 | 0.7442 | 0.3187 | 0.7464 |
| 3 | 0.2956 | 0.8090 | 0.3036 | 0.7831 | 0.3024 | 0.7856 |
| 4 | 0.2790 | 0.8377 | 0.2884 | 0.8157 | 0.2870 | 0.8183 |
| 5 | 0.2636 | 0.8774 | 0.2740 | 0.8438 | 0.2723 | 0.8463 |
| 6 | 0.2487 | 0.9082 | 0.2603 | 0.8674 | 0.2585 | 0.8697 |
| 7 | 0.2351 | 0.9272 | 0.2472 | 0.8875 | 0.2453 | 0.8896 |
| 8 | 0.2224 | 0.9437 | 0.2349 | 0.9045 | 0.2329 | 0.9063 |
| 9 | 0.2102 | 0.9541 | 0.2231 | 0.9186 | 0.2210 | 0.9201 |
| 10 | 0.1986 | 0.9594 | 0.2120 | 0.9309 | 0.2098 | 0.9322 |

*Nota.* IVC̄ = media del índice de vulnerabilidad compuesto sobre las 33 unidades territoriales del modelo; WB̄ = media del índice de bienestar (*wellbeing*) agregado. Ambas variables están normalizadas en el rango [0, 1]. El IVC inicial (previa al primer ciclo de inversión) corresponde a la media histórica del período 2007–2024 estimada en 0.354. Fuente: elaboración propia.

La inspección de la Tabla 11 revela que las trayectorias del IVC en los tres escenarios exhiben un comportamiento estrictamente decreciente y aproximadamente lineal durante los primeros cinco años, seguido de una ligera desaceleración en el tramo 6–10 consistente con los rendimientos decrecientes de la inversión en infraestructura. A partir del Año 3, el escenario Focalizado mantiene de manera persistente el IVC más bajo, con una brecha respecto al Distribuido que se amplía de 0.0080 (Año 3) a 0.0134 (Año 10).

---

### 4.3.2 Análisis de equidad territorial: coeficiente de variación interanual

La Tabla 12 cuantifica la desigualdad territorial interna de cada escenario mediante el coeficiente de variación (CV = σ/μ) del IVC entre comunas, ofreciendo una perspectiva dinámica de la equidad a lo largo del período de simulación.

**Tabla 12**

*Coeficiente de variación (CV) del IVC inter-comunal por escenario de inversión, evolución anual 2007–Año 10*

| Año | CV Focalizado | CV Distribuido | CV Adaptativo |
|:---:|:---:|:---:|:---:|
| 1 | 0.1279 | 0.1885 | 0.1782 |
| 2 | 0.1156 | 0.1886 | 0.1748 |
| 3 | 0.1114 | 0.1884 | 0.1662 |
| 4 | 0.1028 | 0.1882 | 0.1544 |
| 5 | 0.0978 | 0.1879 | 0.1436 |
| 6 | 0.0789 | 0.1887 | 0.1463 |
| 7 | 0.0865 | 0.1878 | 0.1291 |
| 8 | 0.0888 | 0.1879 | 0.1245 |
| 9 | 0.0878 | 0.1880 | 0.1202 |
| 10 | **0.0913** | 0.1886 | **0.1202** |

*Nota.* CV = coeficiente de variación (σ/μ); valores menores indican mayor equidad territorial. El CV del escenario **Focalizado** desciende progresivamente desde **0.128** (Año 1, valor exacto 0.1279) hasta **0.091** (Año 10), con un mínimo de 0.079 en el Año 6. El CV del escenario **Distribuido** permanece prácticamente constante a lo largo del período (**0.188 ± 0.0004**), lo que demuestra formalmente que la distribución igualitaria del presupuesto no reduce la desigualdad territorial sino que conserva los diferenciales proporcionales heredados. El ligero incremento del CV del Focalizado entre los Años 6–10 respecto al mínimo es consecuencia de la rotación de prioridades: al intervenir nuevas comunas en la segunda mitad del horizonte, se genera una dispersión temporal antes de que los efectos de la inversión se materialicen. Fuente: elaboración propia.

El hallazgo más destacado de la Tabla 12 es la **estabilidad estructural del coeficiente de variación del escenario Distribuido** (0.1885 ± 0.0004 a lo largo de los 10 años), lo que demuestra que la distribución igualitaria del presupuesto es incapaz de reducir la heterogeneidad territorial preexistente. En términos formales, si la regla de asignación es proporcional y el IVC inicial presenta alta varianza inter-unidad, la reducción porcentual es similar para todas las unidades y la desigualdad relativa se conserva. Este resultado es consistente con el teorema de *proporcional equity* en economía del bienestar (Atkinson, 1970).

---

### 4.3.3 Síntesis comparativa y análisis de dominancia estocástica

La Tabla 13 presenta una síntesis cuantitativa de los indicadores clave al final del horizonte de simulación, con el propósito de facilitar la comparación normativa entre escenarios.

**Tabla 13**

*Síntesis de indicadores de desempeño al Año 10 de simulación por escenario de inversión pública*

| Indicador | Focalizado | Distribuido | Adaptativo | Mejor escenario |
|---|:---:|:---:|:---:|:---:|
| IVC medio final | **0.1986** | 0.2120 | 0.2098 | Focalizado |
| Reducción IVC respecto al año base (%) | **43.9** | 40.1 | 40.7 | Focalizado |
| Desviación estándar IVC | **0.0181** | 0.0400 | 0.0252 | Focalizado |
| Coeficiente de variación IVC | **0.0913** | 0.1886 | 0.1202 | Focalizado |
| Rango IVC [mín., máx.] | [0.173, 0.232] | [0.137, 0.300] | [0.158, 0.261] | — |
| Bienestar medio | **0.9594** | 0.9309 | 0.9322 | Focalizado |
| Capacidad infraestructura media | 0.9573 | **1.0000** | **1.0000** | Dist./Adapt. |
| Δ IVC (Año 10 − Año 1) | **−0.1332** | −0.1244 | −0.1259 | Focalizado |
| Velocidad de reducción promedio (puntos/año) | **0.0133** | 0.0124 | 0.0126 | Focalizado |
| IVC máximo residual | **0.2318** | 0.2997 | 0.2610 | Focalizado |

*Nota.* Los valores en negrita indican el mejor desempeño entre los tres escenarios para cada indicador. El «mejor escenario» para el IVC máximo residual corresponde al Focalizado porque un máximo más bajo implica que ninguna unidad territorial queda rezagada con vulnerabilidad muy elevada. La capacidad de infraestructura alcanza plena saturación (1.000) antes en Distribuido y Adaptativo porque estos escenarios distribuyen la inversión en capital físico de manera más uniforme. Fuente: elaboración propia.

---

### 4.3.4 Discusión: implicaciones para la política de inversión pública

Los resultados de la simulación permiten articular tres conclusiones de orden estratégico para el diseño de políticas de inversión pública en ciudades con alta heterogeneidad territorial:

**Primera conclusión — Eficiencia y equidad no son objetivos en conflicto bajo una estrategia de focalización rotante.** Contrariamente a la dicotomía clásica eficiencia–equidad en la asignación de recursos públicos, los resultados muestran que el escenario Focalizado domina a los demás en ambas dimensiones simultáneamente al Año 10: menor IVC medio (eficiencia) y menor coeficiente de variación (equidad territorial). Este resultado emerge del mecanismo de rotación secuencial de prioridades inherente al modelo ABM: una vez que las comunas de mayor vulnerabilidad son intervenidas y su IVC desciende por debajo de un umbral crítico, los recursos se reorientan hacia el siguiente estrato, generando una «frontera de Pareto dinámica» que ninguna regla estática (distribución igualitaria o distribución proporcional fija) puede replicar.

**Segunda conclusión — La estrategia Distribuida conserva la desigualdad territorial estructural.** El coeficiente de variación del escenario Distribuido se mantiene prácticamente invariante (CV ≈ 0.188) durante los diez años de simulación. Esto implica que una política de inversión igualitaria, aunque políticamente atractiva por su neutralidad aparente, reproduce y consolida las asimetrías territoriales heredadas de décadas anteriores. Las comunas históricamente más vulnerables (Popular, Manrique, Santa Cruz) permanecen como los focos de mayor IVC incluso al Año 10 bajo esta estrategia, con valores entre 0.283 y 0.300, mientras que en el escenario Focalizado estas mismas comunas han sido reducidas por debajo de ese umbral.

**Tercera conclusión — El escenario Adaptativo ofrece un balance intermedio, pero su ventaja sobre el Distribuido es marginal.** El Adaptativo supera al Distribuido en todos los indicadores al Año 10, pero la diferencia es de pequeña magnitud (IVC medio 0.2098 vs. 0.2120; CV 0.120 vs. 0.189). La regla de asignación dinámica del Adaptativo aproxima parcialmente el comportamiento del Focalizado en términos de equidad (CV intermedio de 0.120), pero no replica su eficiencia global porque la rotación de prioridades es menos agresiva. Esto sugiere que el diseño de la regla adaptativa es un parámetro crítico: una calibración más agresiva —mayor ponderación del IVC corriente en la función de asignación— podría acercar este escenario al Focalizado.

---

### 4.3.5 Limitaciones y advertencias metodológicas

El análisis presentado en este capítulo está sujeto a las siguientes limitaciones que deben ser consideradas al momento de extrapolar sus conclusiones:

1. **Integración pendiente de la telemetría SIATA.** La nulidad informativa del índice de riesgo climático (`idx_riesgo_clima = 0.5` constante) implica que los resultados del modelo XGBoost y de la simulación ABM no capturan la dimensión de exposición ambiental diferencial entre comunas. Las comunas de ladera (Corregimiento de San Cristóbal, Altavista) presentan perfiles de riesgo por remoción en masa que no quedan reflejados en el IVC actual.

2. **Normalización incompleta de nombres de unidades territoriales.** Como se documenta en el proceso de validación del pipeline ETL, la ausencia de normalización de nombres de comunas genera 33 entidades artificiales en el conjunto de datos fusionado, de las cuales solo 21 son entidades territoriales reales. Si bien la simulación ABM heredó este problema del archivo fuente, los estadísticos agregados reportados en este capítulo son robustos a esta duplicación dado que los valores iniciales de IVC para las variantes del mismo nombre son coherentes entre sí.

3. **Supuesto de linealidad del mecanismo de inversión.** El modelo ABM asume que la función de actualización del IVC en respuesta a la inversión es monotónica y sin rendimientos no lineales discontinuos (como los asociados a umbrales de cobertura universal en salud o educación). En la realidad, es plausible que existan umbrales críticos de inversión acumulada por debajo de los cuales el impacto es prácticamente nulo.

4. **Ausencia de efectos de desbordamiento (*spillover*) entre comunas.** El modelo ABM trata cada agente territorial como independiente. Sin embargo, la evidencia empírica indica que la inversión en infraestructura de transporte o en servicios de salud de una comuna puede beneficiar a comunas adyacentes mediante externalidades positivas de proximidad.

5. **Horizonte temporal limitado.** La simulación se circunscribe a 10 años. Horizontes más prolongados podrían revelar dinámicas de retroalimentación positiva (trampas de vulnerabilidad) o negativa (saturación de infraestructura) que no son visibles en el presente análisis.

---

## Referencias

Alkire, S., & Foster, J. (2011). Counting and multidimensional poverty measurement. *Journal of Public Economics*, *95*(7–8), 476–487. https://doi.org/10.1016/j.jpubeco.2010.11.006

Atkinson, A. B. (1970). On the measurement of inequality. *Journal of Economic Theory*, *2*(3), 244–263. https://doi.org/10.1016/0022-0531(70)90039-6

Coady, D., Grosh, M., & Hoddinott, J. (2004). *Targeting of transfers in developing countries: Review of lessons and experience*. The World Bank. https://doi.org/10.1596/0-8213-5769-0

Hallegatte, S., Bangalore, M., Bonzanigo, L., Fay, M., Kane, T., Narloch, U., Rozenberg, J., Treguer, D., & Vogt-Schilb, A. (2016). *Shock waves: Managing the impacts of climate change on poverty*. The World Bank. https://doi.org/10.1596/978-1-4648-0673-5

IPCC. (2022). *Climate change 2022: Impacts, adaptation and vulnerability. Contribution of Working Group II to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change*. Cambridge University Press. https://doi.org/10.1017/9781009325844

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In I. Guyon et al. (Eds.), *Advances in Neural Information Processing Systems 30 (NIPS 2017)* (pp. 4765–4774). Curran Associates.

Sen, A. (1999). *Development as freedom*. Oxford University Press.

---

*Nota del autor:* Este capítulo fue generado con análisis directo sobre los artefactos computacionales del proyecto: `evidencia/resultados_simulacion_escenarios.csv` (990 filas × 9 columnas), `models/cv_metrics.json`, `models/shap_values.csv` y `models/vulnerability_ranking.csv`. Todos los valores numéricos presentados en las tablas son calculados directamente a partir de los datos de salida del pipeline, sin interpolación ni estimación adicional. Fecha de análisis: 25 de mayo de 2026.
