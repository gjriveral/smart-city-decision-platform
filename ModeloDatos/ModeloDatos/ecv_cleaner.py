import re
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger


def procesar_hoja_ecv(df_hoja):
    """
    Toma un DataFrame de pandas leído desde el Excel de la ECV y lo transforma
    de un formato de "bloques apilados" a un formato tabular (ancho).
    Ignora años no válidos, convierte los valores de 0 en nulos (NaN) y extrae
    solo el nombre de la comuna (sin el código numérico).
    """
    records = []
    current_indicator = None
    years = []

    def parse_year(val):
        val = str(val).strip()
        if val.endswith('.0'):
            val = val[:-2]
        if val.isdigit() and len(val) == 4:
            y = int(val)
            if 2000 <= y <= 2030:
                return str(y)
        return None

    lines = df_hoja.fillna('').values.tolist()

    for i, row in enumerate(lines):
        first_col = str(row[0]).strip()
        is_commune = bool(re.match(r'^\d+\.\s', first_col))

        if not is_commune:
            row_years = []
            for y in row[1:]:
                py = parse_year(y)
                if py:
                    row_years.append(py)

            if len(row_years) > 1:
                years = []
                for y in row[1:]:
                    py = parse_year(y)
                    if py:
                        years.append(py)
                    else:
                        years.append(None)

                def is_metadata(x):
                    return (
                        x in ("", "INDICADORES CALCULADOS", "INDICADORES PARA PUBLICAR")
                        or "tabla" in x.lower()
                        or "nota técnica" in x.lower()
                        or x.lower().startswith("total")
                    )

                if first_col and not is_metadata(first_col):
                    current_indicator = first_col
                else:
                    for j in range(i - 1, max(-1, i - 6), -1):
                        prev_first = str(lines[j][0]).strip()
                        is_prev_commune = bool(re.match(r'^\d+\.\s', prev_first))

                        if prev_first and not is_prev_commune and not is_metadata(prev_first):
                            current_indicator = prev_first
                            break
            continue

        if is_commune:
            if current_indicator and years:
                # AQUÍ ESTÁ EL CAMBIO:
                # Eliminamos los dígitos, el punto y los espacios al inicio (Ej: "1. Popular" -> "Popular")
                comuna_name = re.sub(r'^\d+\.\s*', '', first_col).strip()

                for j, year in enumerate(years):
                    if year is not None and j + 1 < len(row):
                        val = str(row[j + 1]).strip()
                        if val and val.upper() not in ['N.D.', 'N.D', ' ', '', 'NAN']:
                            records.append({
                                'Comuna': comuna_name,
                                'Año': int(year),
                                'Indicador': current_indicator,
                                'Valor': val
                            })

    df_long = pd.DataFrame(records)
    if df_long.empty:
        return pd.DataFrame()

    df_long['Valor'] = df_long['Valor'].str.replace(',', '.')
    df_long['Valor'] = pd.to_numeric(df_long['Valor'], errors='coerce')

    df_long['Valor'] = df_long['Valor'].replace(0, np.nan)
    df_long = df_long.dropna(subset=['Valor'])

    df_wide = df_long.pivot_table(
        index=['Comuna', 'Año'],
        columns='Indicador',
        values='Valor',
        aggfunc='first'
    ).reset_index()

    df_wide.columns.name = None
    return df_wide


def run_ecv_pipeline(input_path: str, output_path: str):
    """
    Función principal para orquestar la limpieza del dataset histórico de ECV.
    """
    logger.info(f"Cargando el archivo ECV desde: {input_path}")

    try:
        dict_hojas = pd.read_excel(input_path, sheet_name=None, header=None)
        nombres_hojas = list(dict_hojas.keys())
        logger.info(f"Hojas encontradas para procesar: {nombres_hojas}")

        logger.info(f"Extrayendo datos de la hoja: {nombres_hojas[0]}...")
        df_hoja1 = procesar_hoja_ecv(dict_hojas[nombres_hojas[0]])

        logger.info(f"Extrayendo datos de la hoja: {nombres_hojas[1]}...")
        df_hoja2 = procesar_hoja_ecv(dict_hojas[nombres_hojas[1]])

        logger.info("Uniendo hojas en un dataset maestro...")
        df_maestro_ecv = pd.merge(df_hoja1, df_hoja2, on=['Comuna', 'Año'], how='outer')

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        df_maestro_ecv.to_excel(output_path, index=False)

        logger.success(
            f"¡Procesamiento de ECV finalizado! Dimensiones: {df_maestro_ecv.shape[0]}x{df_maestro_ecv.shape[1]}")
        logger.info(f"Archivo guardado en: {output_path}")

        return df_maestro_ecv

    except Exception as e:
        logger.error(f"Ocurrió un error procesando el archivo ECV: {e}")
        raise


if __name__ == "__main__":
    RUTA_ENTRADA = "data/raw/HistoricoECV.xlsx"
    RUTA_SALIDA = "data/processed/ECV_Procesado.xlsx"
    run_ecv_pipeline(RUTA_ENTRADA, RUTA_SALIDA)