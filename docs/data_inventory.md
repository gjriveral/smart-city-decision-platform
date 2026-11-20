# Inventario de datos (Prompt 0)

Fecha de cierre de este inventario: 20 de noviembre de 2026. No se descargó ningún archivo nuevo y no se solicitó nada a nombre del autor. Lo que no está en el repositorio se declara como ausente.

El Prompt 0 no se había entregado: no existe `Entregables_UrbanScience/Prompt0_…`. Este archivo cubre ese hueco.

Rutas locales respecto de la raíz del repositorio de la plataforma (`smart-city-decision-platform`).

## Índice de vulnerabilidad y Encuesta de Calidad de Vida

| Qué | Ruta | Años | Grano | ¿Real u oficial? | ¿Alcanza para calibrar o validar? |
| --- | --- | --- | --- | --- | --- |
| Índice compuesto (IVC) y seis subíndices | `ModeloDatos/ModeloDatos/data/processed/indices_comunas.csv` | 2007–2024 en el archivo; IVC completo solo 2013–2024 | 21 territorios (comunas y corregimientos), 378 filas; 252 con IVC no nulo y 126 sin IVC | Panel procesado a partir de la Encuesta de Calidad de Vida municipal, el Índice de Pobreza Multidimensional y un índice multidimensional local. No es una serie sintética. El microdato de la encuesta no está en el repo | Sí para el lado del índice: 2013–2024 está completo. 2007–2012 no se usa porque el IVC falta (no se imputa) |

Columnas: `Comuna`, `Año`, `idx_desempleo`, `idx_habitat`, `idx_educacion`, `idx_riesgo_clima`, `idx_pobreza`, `idx_tejido_social`, `ivc`.

No existe la carpeta `ModeloDatos/ModeloDatos/data/raw/`. El limpiador `ModeloDatos/ModeloDatos/ecv_cleaner.py` está en el repo; los microdatos de la encuesta, no.

## Inversión ejecutada

| Qué | Ruta | Años | Grano | ¿Real u oficial? | ¿Alcanza? |
| --- | --- | --- | --- | --- | --- |
| Agregado usado en la calibración | `data/external/inversion_ejecutada_oficial_territorio_anio.csv` | 2013–2020 | 21 territorios × 8 años = 168 filas, sin nulos y sin celdas imputadas | Oficial. Suma de la inversión ejecutada de año cerrado publicada por la Alcaldía de Medellín. No es sintética | Sí, junto con el índice observado, para calibrar γ en 2013–2018 (índice 2014–2019) y validar 2019–2020 (índice 2020–2021). No alcanza para 2021–2024 |
| Libros de 2013–2018 y 2020 | `data/external/_inv_raw/` (`2013.xlsx` … `2018.xlsx`, `2020.xlsx`) | Esos años | Proyectos georreferenciados, antes de sumar al territorio | Archivos oficiales de la Alcaldía. 2013–2015 y el resto de esa serie traen los datos en la hoja ALL, no en la ficha técnica | Fuente del agregado. No sustituyen los años que faltan |
| 2019 | `data/external/_inv_probe/2019.csv` | Vigencia 2019 (5.292 filas de proyecto; 23 nombres de comuna en el archivo crudo, porque incluye filas que no son territorio) | Proyecto | MEData, identificador 1-002-11-000278, CC BY-SA 4.0. El campo `CORTE` dice `31/12/20` en las 5.292 filas; la cobertura declarada es el año calendario 2019 | Se usó para 2019. La salvedad del corte queda registrada. No se reinterpretó como año 2020 |

Columnas del agregado: `territory`, `anio`, `inversion_pesos`, `fuente`, `ajuste_negativo`. El conteo de `ajuste_negativo` verdadero es 0. Las filas de ciudad completa («Toda la Ciudad») y de fortalecimiento institucional no entran en las 168 filas.

`data/external/inversion_historica_por_territorio.template.csv` y `inversion_historica_schema.json` son una plantilla de esquema. No contienen los pesos oficiales y no se usaron como serie de calibración.

La partición de la calibración es 6 de 8 transiciones (75/25): inversión 2013–2018 para ajustar, inversión 2019–2020 retenida. No es un 70/30.

## Lo que no está

- Inversión ejecutada georreferenciada, de año cerrado y legible por máquina, para 2021, 2022, 2023 y 2024. No se rellenó.
- POAI territorializado: no hay archivo en el repositorio.
- Presupuesto participativo por territorio y año: no hay serie. El manuscrito habla del procedimiento como institución; no hay montos.
- Medellín Cómo Vamos: no hay archivo.
- Población DANE por comuna o corregimiento: no hay serie. `config.yaml` guarda el código municipal DANE 05001. `data/geo/comunas.geojson` es el límite administrativo, no un censo. `download_geo_real.py` mapea códigos de límite, no habitantes.
- Lluvia minuto a minuto de SIATA (del orden de 20 GB): no está en el repo (`pluviometrica/` está ignorada). Sí están los derivados.

Las carpetas `Datos_*_Beta/` de otras ciudades, incluido `Datos_Bogotá_Beta/bogota_dummy.geojson`, no son la calibración de Medellín. `generar_datos_bogota.py` declara que esos archivos no son datos oficiales del DANE ni de la Alcaldía.

## Lluvia SIATA

| Qué | Ruta | Años | Grano | ¿Real u oficial? | ¿Alcanza? |
| --- | --- | --- | --- | --- | --- |
| Lluvia territorial anual | `data/processed/siata_territorio_anual.csv` | 2012–2026 | 21 territorios, 315 filas | Derivada de la red de pluviógrafos de SIATA. 2012 y 2026 son años parciales y no entran a los modelos | Sirve para el pronóstico y el control de calidad. No entra en la transición del índice: esa ecuación no tiene término de lluvia |
| Agregado por estación | `data/processed/siata_agregado.csv` | Estación-año | Estación | Misma red | Insumo intermedio |
| Catálogo y asignación | `data/external/pluvio_siata_catalogo.json`, `data/external/asignacion_estacion_UT_oficial.json` | Catálogo vigente en el repo | Estación → territorio | Lista oficial de estaciones más el catálogo | Asignación. No es una serie de inversión ni de población |

## Lista de descarga, por prioridad

No se descargó nada en este paso. Identificadores que ya están en el repo se repiten aquí para no perderlos.

1. **(A)** Inversión ejecutada georreferenciada, de año cerrado y legible por máquina, 2021–2024, en MEData o en Planeación. Sin este archivo la validación del índice no puede pasar de 2021. El identificador ya usado para 2019 es MEData `1-002-11-000278`. En ese archivo el campo de corte dice `31/12/20` y la vigencia declarada es 2019; cualquier archivo nuevo hay que leerlo con la misma cautela.
2. **(B)** POAI territorializado por comuna y corregimiento.
3. **(C)** Presupuesto participativo por territorio y año.
4. **(D)** Series de Medellín Cómo Vamos que se quieran contrastar con el IVC.
5. **(E)** Población DANE por comuna y corregimiento, con el año del censo o de la proyección.
