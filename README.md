# Plataforma Integral de Toma de Decisiones para Ciudades Inteligentes

> **Prediccion de Vulnerabilidad Urbana e Inversion Estrategica mediante Big Data, Machine Learning y Modelado Basado en Agentes**
>
> Universidad de Antioquia · Trabajo de Grado 2026 · Facultad de Ingenieria

---

## Descripcion General

Esta plataforma integra tres paradigmas computacionales para apoyar la toma de decisiones de politica publica en ciudades colombianas:

| Componente | Tecnologia | Funcion |
|---|---|---|
| **Big Data / IoT** | Dask + Pandas | Procesa la red de estaciones pluviometricas SIATA (telemetria ambiental en tiempo real) |
| **Machine Learning** | XGBoost + SHAP | Predice el Indice de Vulnerabilidad Compuesto (IVC) por unidad territorial con validacion cruzada temporal (walk-forward) |
| **Modelado Basado en Agentes** | Mesa (Python) | Simula tres escenarios de asignacion presupuestal durante 10 anos para comparar estrategias de inversion publica |
| **Dashboard Interactivo** | Streamlit + Plotly + Folium | Visualiza predicciones, mapas corepleticos, importancia de variables y permite reentrenar el modelo con un solo clic |

El sistema fue disenado desde su arquitectura para ser **completamente replicable en cualquier ciudad de Colombia**: basta con editar el archivo `config.yaml` para adaptar la plataforma a Bogota, Cali, Barranquilla u otra ciudad, sin modificar una sola linea de codigo fuente.

---

## Tabla de Contenidos

1. [Requisitos Previos](#1-requisitos-previos)
2. [Estructura del Directorio](#2-estructura-del-directorio)
3. [Guia de Parametrizacion Espacial](#3-guia-de-parametrizacion-espacial-lo-mas-importante)
4. [Estructura de Datos Requerida](#4-estructura-de-datos-requerida)
5. [Ejecucion de la Plataforma](#5-ejecucion-de-la-plataforma)
6. [Ejecucion por Modulos](#6-ejecucion-por-modulos)
7. [Resultados y Artefactos](#7-resultados-y-artefactos)
8. [Rendimiento del Modelo](#8-rendimiento-del-modelo)
9. [Arquitectura de Software](#9-arquitectura-de-software)
10. [Licencia y Autores](#10-licencia-y-autores)

---

## 1. Requisitos Previos

### Python

Se requiere **Python 3.9 o superior**. Se recomienda usar un entorno virtual:

```bash
# Crear entorno virtual
python -m venv .venv

# Activar (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activar (Linux / macOS)
source .venv/bin/activate
```

### Dependencias

Instale todas las librerias con un solo comando:

```bash
pip install -r requirements.txt
```

**`requirements.txt`** (contenido completo):

```
# Core
numpy>=1.24
pandas>=2.0
openpyxl>=3.1
xlrd>=2.0

# Big Data
dask[dataframe]>=2024.1

# Machine Learning
scikit-learn>=1.4
xgboost>=2.0
shap>=0.44

# Modelado basado en agentes
mesa>=2.3

# Visualizacion
matplotlib>=3.8
plotly>=5.20
streamlit>=1.35
streamlit-folium>=0.20
folium>=0.16

# Geoespacial
geopandas>=1.0
pyproj>=3.6
shapely>=2.0

# Configuracion
pyyaml>=6.0

# Utilidades
tqdm>=4.66
python-dotenv>=1.0
```

> **Nota:** Si su entorno ya tiene algunas librerias instaladas, puede omitir versiones especificas. El proyecto fue desarrollado y probado con Python 3.14 en Windows 11.

---

## 2. Estructura del Directorio

```
model_strategic_investment_cities/
|
|-- config.yaml                     # << EDITAR PRIMERO >> Parametros de ciudad
|-- config_loader.py                # Lee config.yaml y expone variables tipadas
|-- requirements.txt                # Dependencias del proyecto
|-- README.md                       # Esta documentacion
|
|-- 03_dashboard_visualizacion.py   # Dashboard Streamlit (punto de entrada principal)
|-- run_ml_pipeline.py              # Ejecutor secuencial del pipeline ML
|-- run_abm_pipeline.py             # Ejecutor secuencial del pipeline ABM
|-- abm_direct_runner.py            # Runner alternativo para el ABM
|
|-- bigdata/                        # Modulo 1: Procesamiento de datos IoT/SIATA
|   |-- 01_siata_processor.py       # Pipeline Dask: limpieza y agregacion pluviometrica
|   `-- run_sample.py               # Modo de prueba con 10 estaciones
|
|-- ml/                             # Modulo 2: Machine Learning
|   |-- 00_inspect_columns.py       # Inspeccion rapida de columnas en datos crudos
|   |-- 02_vulnerability_indices.py # Construye los 7 indices de vulnerabilidad
|   `-- 03_xgboost_model.py         # Entrena XGBoost con CV temporal y genera SHAP
|
|-- abm/                            # Modulo 3: Modelado Basado en Agentes (Mesa)
|   |-- 04_urban_model.py           # Modelo ABM: 3 agentes x N territorios x 10 anos
|   `-- 05_scenario_analysis.py     # Genera informe HTML comparativo de escenarios
|
|-- pluviometrica/                  # Datos IoT crudos: estaciones pluviometricas SIATA
|   `-- Estacion_pluviometrica_*.csv  # Un CSV por estacion (formato SIATA)
|
|-- ModeloDatos/
|   `-- ModeloDatos/
|       `-- data/
|           |-- raw/                # Datos sociales crudos (ECV, IPM, IML en .xlsx)
|           `-- processed/          # Datos sociales procesados
|               |-- ECV_Procesado.xlsx
|               |-- IML_Procesado.xlsx
|               |-- IPM_Procesado.xlsx
|               `-- indices_comunas.csv  # Salida del pipeline ML (7 indices + IVC)
|
|-- data/
|   |-- processed/                  # Artefactos generados por los pipelines
|   |   |-- siata_agregado.csv      # Precipitacion agregada por estacion/ano/territorio
|   |   `-- simulation_results.csv  # Resultados de los 3 escenarios ABM
|   `-- geo/                        # Archivos geoespaciales (agregar aqui)
|       `-- comunas.geojson         # << OPCIONAL >> Geometrias para mapa Folium
|
|-- models/                         # Artefactos del modelo entrenado
|   |-- vulnerability_model.pkl     # Modelo XGBoost serializado
|   |-- cv_metrics.json             # Metricas de validacion cruzada temporal
|   |-- vulnerability_ranking.csv   # Ranking de IVC medio por unidad territorial
|   |-- shap_values.csv             # Valores SHAP por muestra y variable
|   |-- feature_importance.png      # Grafico de importancia (Gain)
|   `-- shap_importance.png         # Grafico de importancia SHAP
|
|-- reports/
|   `-- scenario_report.html        # Informe interactivo HTML del ABM
|
`-- logs/                           # Trazabilidad de ejecucion
    |-- ml_execution.log
    |-- abm_execution.log
    `-- bigdata_execution.log
```

---

## 3. Guia de Parametrizacion Espacial *(Lo mas importante)*

> **Este es el unico archivo que necesita editar para adaptar la plataforma a su ciudad.**

### 3.1 Abrir `config.yaml`

Toda la configuracion geografica y organizacional de la plataforma vive en `config.yaml`. El resto del sistema consume este archivo a traves de `config_loader.py`, por lo que ningun script de codigo contiene referencias geograficas fijas.

### 3.2 Seccion `ciudad` — Metadatos generales

```yaml
ciudad:
  nombre:             "Medellin"      # Nombre que aparece en el dashboard y reportes
  departamento:       "Antioquia"
  codigo_dane:        "05001"         # Codigo DANE del municipio
  unidad_territorial: "comuna"        # CLAVE: define el nombre de columna en todos los CSVs
  crs:                "EPSG:4326"     # Sistema de referencia de coordenadas
  centro:
    lat: 6.2442                       # Latitud del centroide de la ciudad (para centrar el mapa)
    lon: -75.5812
```

#### Ejemplos de adaptacion por ciudad

| Ciudad | `nombre` | `unidad_territorial` | `codigo_dane` | `centro.lat` | `centro.lon` |
|---|---|---|---|---|---|
| Medellin | `"Medellin"` | `"comuna"` | `"05001"` | `6.2442` | `-75.5812` |
| Bogota | `"Bogota"` | `"localidad"` | `"11001"` | `4.7110` | `-74.0721` |
| Cali | `"Cali"` | `"comuna"` | `"76001"` | `3.4516` | `-76.5320` |
| Barranquilla | `"Barranquilla"` | `"localidad"` | `"08001"` | `10.9685` | `-74.7813` |
| Bucaramanga | `"Bucaramanga"` | `"comuna"` | `"68001"` | `7.1193` | `-73.1227` |

**Efecto inmediato:** al cambiar `unidad_territorial: "localidad"`, el dashboard muestra automaticamente *"Prioridad por Localidad"*, los graficos de Plotly se etiquetan con el termino correcto y los popups del mapa Folium dicen *"Localidad:"*.

### 3.3 Seccion `rutas` — Directorios de datos

```yaml
rutas:
  datos_climaticos:  "pluviometrica"               # Carpeta con CSVs de estaciones IoT
  datos_sociales:    "ModeloDatos/ModeloDatos/data" # Carpeta raiz de datos sociales
  procesados:        "data/processed"               # Salidas de los pipelines
  modelos:           "models"                       # Artefactos ML (.pkl, .json, .csv)
  logs:              "logs"
  geo:               "data/geo"                     # Archivos geoespaciales (.geojson, .shp)
```

Todas las rutas son **relativas a la raiz del proyecto**. Puede cambiarlas si su estructura de carpetas es diferente.

### 3.4 Seccion `territorios` — Unidades territoriales con centroides

Lista cada unidad territorial con su identificador, nombre canonico y coordenadas WGS84. Estos centroides se usan para:
- Asignar cada estacion IoT a su territorio mas cercano (distancia haversine)
- Dibujar el mapa de burbujas si no hay archivo `.geojson`

```yaml
territorios:
  - id: 1
    nombre: "Popular"       # Debe coincidir EXACTAMENTE con los valores en los CSVs sociales
    lat: 6.2969
    lon: -75.5584
  - id: 2
    nombre: "Santa Cruz"
    lat: 6.2897
    lon: -75.5566
  # ... agregar todas las unidades territoriales de su ciudad
```

> **Advertencia:** el campo `nombre` debe ser identico (mayusculas, tildes, espacios) al valor de la columna de unidad territorial en los archivos Excel de datos sociales. Cualquier discrepancia hace que esa unidad quede sin datos en el cruce.

### 3.5 Seccion `catalogo_estaciones` — Coordenadas de estaciones IoT

Mapeo de ID de estacion a coordenadas geograficas. Se usa para asignar cada estacion climatica a su unidad territorial mas proxima:

```yaml
catalogo_estaciones:
  2:   {lat: 6.2501, lon: -75.5683}
  3:   {lat: 6.2897, lon: -75.5566}
  # ... una entrada por estacion conocida
```

Si una estacion no esta en el catalogo, el sistema le asigna un territorio de forma deterministica (hash del ID) y lo marca con `[HASH]` en el log para revision manual.

### 3.6 Archivo geoespacial (opcional, pero recomendado)

Para activar el mapa corepletico con poligonos reales:

1. Consiga el shapefile o GeoJSON de las unidades territoriales de su ciudad desde el portal de **Datos Abiertos Colombia** (datos.gov.co) o el geoportal de su municipio.
2. Guardelo en `data/geo/` con el nombre `<unidad_territorial>s.geojson` (ej. `comunas.geojson` o `localidades.geojson`).
3. Reinicie el dashboard: el mapa Folium se activara automaticamente.

Si no hay archivo geoespacial, el dashboard muestra un mapa de burbujas usando los centroides del `config.yaml`.

---

## 4. Estructura de Datos Requerida

### 4.1 Datos sociales (fuentes ECV, IPM, IML)

Los tres archivos Excel deben ubicarse en `ModeloDatos/ModeloDatos/data/processed/` con los nombres exactos:

| Archivo | Fuente | Contenido |
|---|---|---|
| `ECV_Procesado.xlsx` | Encuesta de Calidad de Vida | Tasas de escolaridad, seguridad alimentaria, convivencia |
| `IPM_Procesado.xlsx` | Indice de Pobreza Multidimensional | Hacinamiento, acceso a servicios, logro educativo |
| `IML_Procesado.xlsx` | Indicadores de Mercado Laboral | Tasas de desempleo, informalidad, participacion |

**Estructura minima requerida por archivo:**

```
| <unidad_territorial> | Año  | Variable_1 | Variable_2 | ... |
|----------------------|------|------------|------------|-----|
| Popular              | 2020 | 12.5       | 8.3        | ... |
| Santa Cruz           | 2020 | 15.1       | 9.7        | ... |
```

- La columna de unidad territorial puede llamarse `Comuna`, `Localidad`, `Corregimiento`, `UPZ`, `Barrio` u otras variantes — el pipeline las detecta y estandariza automaticamente al nombre configurado en `unidad_territorial`.
- La columna de tiempo puede llamarse `Año`, `Ano`, `Year`, `Periodo` o `Fecha`.
- El resto de columnas son variables de indicadores; el pipeline las detecta por patrones de nombre (regex) configurados en `ml/02_vulnerability_indices.py`.

### 4.2 Datos climaticos / IoT (estaciones pluviometricas)

Cada archivo en `pluviometrica/` debe seguir la convencion de nombre `Estacion_pluviometrica_<ID>_<fecha_inicio>_<fecha_fin>.csv` y contener las columnas:

```
codigo    : int   — ID de la estacion (debe coincidir con el ID en el nombre del archivo)
fecha_hora: str   — Datetime en formato 'YYYY-MM-DD HH:MM:SS'
p1        : float — Precipitacion en mm (acumulado 1 minuto)
p2        : float — Precipitacion alternativa (sensor secundario)
calidad   : int   — Bandera de calidad (1 = valido, otros = descartado)
```

El pipeline filtra automaticamente los registros con `calidad != 1` y agrega por estacion, ano y territorio: precipitacion media, maxima y percentil 95.

---

## 5. Ejecucion de la Plataforma

### Metodo recomendado: Dashboard interactivo

```bash
py -m streamlit run 03_dashboard_visualizacion.py
```

El dashboard se abre automaticamente en el navegador en `http://localhost:8501`.

```
Desde el dashboard puede:
  1. Seleccionar el perfil de ciudad (Medellin / Bogota / personalizado)
  2. Subir un config.yaml propio desde la barra lateral
  3. Explorar los indicadores, mapas y graficos de interpretabilidad
  4. Presionar "Ejecutar Pipeline Automatico" para reentrenar el modelo
```

#### Boton "Ejecutar Pipeline Automatico"

Al presionar este boton, el dashboard lanza en secuencia los tres modulos del pipeline sobre los datos de la ciudad activa:

```
[1/3]  bigdata/01_siata_processor.py     → agrega datos IoT por territorio y ano
[2/3]  ml/02_vulnerability_indices.py    → construye los 7 indices de vulnerabilidad
[3/3]  ml/03_xgboost_model.py           → entrena XGBoost y genera artefactos SHAP
```

Al finalizar, limpia el cache de Streamlit y recarga el dashboard con los nuevos modelos y datos. No es necesario reiniciar el servidor.

---

## 6. Ejecucion por Modulos

Si prefiere ejecutar cada etapa manualmente desde la terminal:

### Paso 1 — Procesamiento de datos IoT (Big Data)

```bash
# Modo muestra (10 estaciones, rapido, recomendado para pruebas)
py bigdata/run_sample.py

# Modo completo (todas las estaciones, puede tardar varios minutos)
py bigdata/01_siata_processor.py
```

Salida: `data/processed/siata_agregado.csv`

### Paso 2 — Construccion de indices de vulnerabilidad

```bash
py ml/02_vulnerability_indices.py
```

Salida: `ModeloDatos/ModeloDatos/data/processed/indices_comunas.csv`

Los 7 indices generados son:

| Indice | Variable | Fuente |
|---|---|---|
| `idx_desempleo` | Tension de desempleo | IML |
| `idx_habitat` | Deficit de habitat | ECV + IPM |
| `idx_educacion` | Vulnerabilidad educativa | ECV + IPM |
| `idx_riesgo_clima` | Riesgo climatico | SIATA (placeholder 0.5) |
| `idx_pobreza` | Pobreza multidimensional | IPM |
| `idx_tejido_social` | Fragilidad del tejido social | ECV |
| `ivc` | Indice de Vulnerabilidad Compuesto | Promedio ponderado (1–6) |

### Paso 3 — Entrenamiento del modelo XGBoost

```bash
py ml/03_xgboost_model.py
```

Salidas en `models/`:
- `vulnerability_model.pkl` — modelo serializado
- `cv_metrics.json` — metricas de validacion cruzada
- `vulnerability_ranking.csv` — ranking de IVC por territorio
- `shap_values.csv` — valores SHAP por muestra
- `feature_importance.png` y `shap_importance.png`

### Paso 4 — Simulacion ABM (opcional)

```bash
py abm/04_urban_model.py
```

Salida: `data/processed/simulation_results.csv`

### Paso 5 — Informe HTML de escenarios (opcional)

```bash
py abm/05_scenario_analysis.py
```

Salida: `reports/scenario_report.html` (informe interactivo autocontenido)

### Pipeline completo de una vez

```bash
# ML completo
py run_ml_pipeline.py

# ABM completo
py run_abm_pipeline.py
```

---

## 7. Resultados y Artefactos

### Dashboard (Streamlit)

| Pestana | Contenido |
|---|---|
| **Vision General** | 4 KPIs de ciudad, ranking horizontal de vulnerabilidad, donut de prioridades, evolucion temporal del IVC, tabla detallada por territorio |
| **Mapa Predictivo** | Mapa corepletico Folium (con `.geojson`) o mapa de burbujas Plotly (con centroides del config) |
| **Interpretabilidad** | XGBoost Gain Importance, SHAP mean absolute value, curva de R² por fold, tabla de metricas CV, hiperparametros |

### Informe ABM (`reports/scenario_report.html`)

Contiene 4 figuras comparativas de los 3 escenarios de inversion:

| Figura | Descripcion |
|---|---|
| Fig 1 | Evolucion del bienestar promedio durante 10 anos |
| Fig 2 | Heatmap de bienestar por territorio al ano 10 |
| Fig 3 | Perfil de asignacion presupuestal por escenario |
| Fig 4 | Capacidad de infraestructura en el tiempo |

Los tres escenarios simulados son:

- **A — Focalizado:** 80% del presupuesto a los 5 territorios mas vulnerables
- **B — Distribuido:** distribucion igualitaria entre todos los territorios
- **C — Adaptativo:** reponderacion cada 2 anos segun tasa de mejora observada

---

## 8. Rendimiento del Modelo

Resultados de la validacion cruzada temporal walk-forward sobre datos de Medellin (2007–2024):

| Fold | Anos entrenamiento | Anos prueba | RMSE | MAE | R² |
|---|---|---|---|---|---|
| 1 | 2007–2010 | 2010–2013 | 0.02606 | 0.01913 | 0.9224 |
| 2 | 2007–2013 | 2013–2016 | 0.01147 | 0.00896 | 0.9747 |
| 3 | 2007–2016 | 2016–2019 | 0.00866 | 0.00666 | 0.9821 |
| 4 | 2007–2019 | 2019–2022 | 0.01910 | 0.01324 | 0.9138 |
| 5 | 2007–2022 | 2022–2024 | 0.00982 | 0.00696 | 0.9782 |
| **Promedio** | | | **0.01502** | **0.01099** | **0.9542** |

**R² en datos completos: 0.9985** — el modelo explica el 95.4% de la varianza del IVC en validacion out-of-sample.

Hiperparametros optimos seleccionados por grid search interno:

```json
{
  "n_estimators":      200,
  "max_depth":         3,
  "learning_rate":     0.1,
  "subsample":         0.8,
  "colsample_bytree":  0.8,
  "min_child_weight":  1
}
```

---

## 9. Arquitectura de Software

### Patron de diseno: Single Source of Truth

Toda la configuracion de ciudad reside en `config.yaml`. El modulo `config_loader.py` la lee, valida y expone como objetos tipados (dataclasses). Ningun script contiene referencias geograficas fijas.

```
config.yaml
    |
    v
config_loader.py  (get_config() -> CityConfig)
    |
    +-- bigdata/01_siata_processor.py    (cfg.unidad_territorial, cfg.rutas)
    +-- ml/02_vulnerability_indices.py   (UT_COL, merge keys, rutas)
    +-- ml/03_xgboost_model.py           (UT_COL, groupby, rutas)
    +-- abm/04_urban_model.py            (UT, territorios dinamicos desde datos)
    +-- abm/05_scenario_analysis.py      (UT en pivot_table y merge)
    `-- 03_dashboard_visualizacion.py    (CIUDAD, UT_COL, UT_PLU en todos los textos)
```

### Flujo de datos

```
[Estaciones IoT SIATA]          [Encuestas Sociales ECV/IPM/IML]
        |                                       |
        v                                       v
01_siata_processor.py           02_vulnerability_indices.py
        |                                       |
        v                                       v
siata_agregado.csv              indices_comunas.csv
                                        |
                                        v
                              03_xgboost_model.py
                                        |
                          +-------------+-------------+
                          |             |             |
                          v             v             v
                vulnerability_   cv_metrics.    shap_values.
                model.pkl        json           csv
                          |
                          v
                04_urban_model.py
                          |
                          v
                simulation_results.csv
                          |
                          v
                05_scenario_analysis.py
                          |
                          v
                scenario_report.html
                          |
                          v
                03_dashboard_visualizacion.py  <-- punto de entrada del usuario
```

---

## 10. Licencia y Autores

### Autores

| Rol | Nombre | Institucion |
|---|---|---|
| Investigador principal | *(nombre del autor)* | Universidad de Antioquia |
| Director de trabajo de grado | *(nombre del director)* | Universidad de Antioquia |

### Citacion academica

Si utiliza esta plataforma en su investigacion, por favor cite:

```
[Autor], "[Titulo del trabajo de grado]",
Trabajo de Grado, Facultad de Ingenieria,
Universidad de Antioquia, Medellin, Colombia, 2026.
```

### Licencia

Este proyecto se distribuye bajo la **Licencia MIT** para uso academico e investigativo.

```
MIT License

Copyright (c) 2026 [Nombre del Autor] — Universidad de Antioquia

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

### Datos y fuentes

Los datos utilizados en el caso de estudio de Medellin provienen de fuentes publicas:

- **SIATA** (Sistema de Alerta Temprana de Medellin y el Valle de Aburra) — red de estaciones pluviometricas bajo licencia de uso publico
- **Encuesta de Calidad de Vida (ECV)** — Alcaldia de Medellin / DANE
- **Indice de Pobreza Multidimensional (IPM)** — DANE Colombia
- **Indicadores de Mercado Laboral (IML)** — DANE / Ministerio del Trabajo

---

## Apendice — Preguntas Frecuentes

**P: ¿Puedo usar esta plataforma sin datos de SIATA?**
R: Si. El indice de riesgo climatico (`idx_riesgo_clima`) usa un valor placeholder de `0.5` cuando no hay datos IoT disponibles. El modelo XGBoost se entrena igualmente con los 6 indices sociales restantes.

**P: ¿El modelo se puede usar con datos de un solo ano?**
R: El modelo fue disenado para series temporales (2007–2024). Con un solo ano no es posible la validacion cruzada temporal, pero se puede entrenar en modo `full_data` y usar las predicciones como referencia inicial.

**P: ¿Como agrego una nueva variable social?**
R: Agregue la columna al archivo Excel correspondiente (ECV, IPM o IML) y añada el patron regex en la seccion correspondiente de `ml/02_vulnerability_indices.py` (funciones `_match_cols`). No es necesario tocar el config.

**P: ¿Por que el mapa muestra burbujas en lugar de poligonos?**
R: El mapa corepletico completo requiere un archivo `.geojson` con las geometrias de las unidades territoriales. Descárguelo del geoportal de su ciudad y copielo en `data/geo/<unidad>s.geojson`. El dashboard lo detectara automaticamente al reiniciar.

**P: ¿Cuanto tiempo tarda el pipeline completo?**
R: En modo muestra (10 estaciones): ~30 segundos. En modo completo con todas las estaciones SIATA de Medellin: ~5–15 minutos dependiendo del hardware. El reentrenamiento XGBoost tarda tipicamente menos de 10 segundos.

---

*Generado automaticamente por el pipeline de documentacion — Universidad de Antioquia · 2026*
