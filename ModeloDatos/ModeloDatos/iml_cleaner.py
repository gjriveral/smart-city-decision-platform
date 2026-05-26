import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger


def procesar_hoja_mercado(df_hoja):
    """
    Recibe un DataFrame de Pandas cargado desde el Excel de Mercado Laboral.
    Detecta la jerarquía (espacios) y extrae solo métricas porcentuales,
    ajustando dinámicamente el cambio de comunas.
    """
    records = []
    current_commune = None
    years = []
    current_parent_indicator = None

    lines = df_hoja.fillna('').values.tolist()

    for i, row in enumerate(lines):
        col1_raw = str(row[1])
        if col1_raw == '':
            continue

        col1 = col1_raw.strip()

        if col1 == "Concepto":
            years = [str(y).strip().replace('.0', '') for y in row[2:]
                     if str(y).strip().replace('.0', '').isdigit() and len(str(y).strip().replace('.0', '')) == 4]

            for j in range(i - 1, max(-1, i - 4), -1):
                prev_col1 = str(lines[j][1]).strip()
                if prev_col1 and prev_col1 not in ["Enero - Diciembre", "Medellín y 16 comunas",
                                                   "Medellín y 5 corregimientos"] and "población" not in prev_col1.lower() and "Regresar" not in prev_col1:
                    current_commune = prev_col1
                    break

            current_parent_indicator = None
            continue

        if not years or not current_commune:
            continue

        es_metadato = col1 in ["Concepto", "Ene - Dic"] or col1.startswith("Fuente:") or "Notas:" in col1 or "*" in col1

        if not es_metadato:
            espacios_al_inicio = len(col1_raw) - len(col1)

            if espacios_al_inicio > 0 and current_parent_indicator:
                indicator_name = f"{current_parent_indicator} - {col1}"
            else:
                indicator_name = col1
                current_parent_indicator = col1

            # 4. Extraer los valores
            for j, year in enumerate(years):
                if j + 2 < len(row):
                    val = str(row[j + 2]).strip()
                    if val and val.lower() not in ['nd', 'n.d.', '', 'nan']:
                        records.append({
                            'Comuna': current_commune,
                            'Año': int(year),
                            'Indicador': indicator_name,
                            'Valor': val
                        })

    df_long = pd.DataFrame(records)
    if df_long.empty:
        return pd.DataFrame()

    df_long['Valor'] = df_long['Valor'].str.replace(',', '.')
    df_long['Valor'] = pd.to_numeric(df_long['Valor'], errors='coerce')
    df_long['Valor'] = df_long['Valor'].replace(0, np.nan)
    df_long = df_long.dropna(subset=['Valor'])

    metricas_porcentuales = [
        '% población en edad de trabajar',
        'TGP', 'TO', 'TD', 'T.D. Abierto', 'T.D. Oculto',
        'Tasa de subempleo subjetivo',
        'Tasa de subempleo subjetivo - Insuficiencia de horas',
        'Tasa de subempleo subjetivo - Empleo inadecuado por competencias',
        'Tasa de subempleo subjetivo - Empleo inadecuado por ingresos',
        'Tasa de subempleo objetivo',
        'Tasa de subempleo objetivo - Insuficiencia de horas',
        'Tasa de subempleo objetivo - Empleo inadecuado por competencias',
        'Tasa de subempleo objetivo - Empleo inadecuado por ingresos',
        'T.Informalidad'
    ]

    df_long = df_long[df_long['Indicador'].isin(metricas_porcentuales)]

    df_wide = df_long.pivot_table(
        index=['Comuna', 'Año'],
        columns='Indicador',
        values='Valor',
        aggfunc='first'
    ).reset_index()

    df_wide.columns.name = None
    return df_wide


def run_iml_pipeline(input_path: str, output_path: str):
    """
    Función principal para orquestar la extracción de indicadores del mercado laboral.
    """
    logger.info(f"Cargando el archivo de Mercado Laboral desde: {input_path}")
    hojas_a_cargar = ["IML 16 Comunas ", "IML 5 Corregimientos"]

    try:
        dict_hojas = pd.read_excel(input_path, sheet_name=hojas_a_cargar, header=None)

        logger.info(f"Extrayendo datos de la hoja: {hojas_a_cargar[0]}...")
        df_comunas = procesar_hoja_mercado(dict_hojas[hojas_a_cargar[0]])

        logger.info(f"Extrayendo datos de la hoja: {hojas_a_cargar[1]}...")
        df_corregimientos = procesar_hoja_mercado(dict_hojas[hojas_a_cargar[1]])

        logger.info("Uniendo ambas zonas geográficas en un dataset maestro...")
        df_iml_maestro = pd.concat([df_comunas, df_corregimientos], ignore_index=True)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        df_iml_maestro.to_excel(output_path, index=False)

        logger.success(
            f"¡Procesamiento IML finalizado! Dimensiones: {df_iml_maestro.shape[0]}x{df_iml_maestro.shape[1]}")
        logger.info(f"Archivo guardado en: {output_path}")

        return df_iml_maestro

    except Exception as e:
        logger.error(f"Ocurrió un error procesando el archivo IML: {e}")
        raise


if __name__ == "__main__":
    RUTA_ENTRADA = "data/raw/HistoricoMercadoLaboral.xlsx"
    RUTA_SALIDA = "data/processed/IML_Procesado.xlsx"
    run_iml_pipeline(RUTA_ENTRADA, RUTA_SALIDA)