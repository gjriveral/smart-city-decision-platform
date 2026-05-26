import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger


def procesar_hoja_ipm_total(df_hoja):
    """
    Procesa la hoja del IPM total donde los años están como columnas.
    Utiliza la columna 'Código' para filtrar, pero guarda solo el nombre de la comuna.
    """
    records = []
    lines = df_hoja.fillna('').values.tolist()

    header_idx = -1
    year_indices = {}
    comuna_idx = -1
    codigo_idx = -1

    # 1. Encontrar la fila de cabecera y mapear las columnas
    for i, row in enumerate(lines):
        row_str = [str(x).strip() for x in row]
        if "Comuna - Corregimiento" in row_str or "Nombre" in row_str:
            header_idx = i
            for idx, val in enumerate(row_str):
                val_clean = val.replace('.0', '')
                if val_clean.isdigit() and len(val_clean) == 4:
                    year_indices[int(val_clean)] = idx
                elif "Comuna" in val or "Nombre" in val:
                    comuna_idx = idx
                elif "Código" in val or "Codigo" in val:
                    codigo_idx = idx
            break

    if header_idx == -1: return pd.DataFrame()

    # 2. Extraer los datos
    for row in lines[header_idx + 1:]:
        if comuna_idx >= len(row) or codigo_idx >= len(row) or codigo_idx == -1:
            continue

        comuna_name = str(row[comuna_idx]).strip()
        codigo = str(row[codigo_idx]).strip().replace('.0', '')

        # Validar si es una comuna/corregimiento real (Código mayor a 0)
        try:
            cod_int = int(codigo)
            is_valid = cod_int > 0
        except ValueError:
            is_valid = False

        if is_valid:
            # Guardar exclusivamente el nombre de la comuna
            comuna_formateada = comuna_name

            for year, idx in year_indices.items():
                if idx < len(row):
                    val = str(row[idx]).strip()
                    if val and val.lower() not in ['nd', 'n.d.', '', 'nan', ' ']:
                        records.append({
                            'Comuna': comuna_formateada,
                            'Año': year,
                            'Indicador': 'IPM Total',
                            'Valor': val
                        })

    return pd.DataFrame(records)


def procesar_hoja_dimensiones(df_hoja):
    """
    Procesa la hoja de dimensiones utilizando la columna 'Código' para
    filtrar, pero conserva solo el nombre de la comuna en el formato final.
    """
    records = []
    lines = df_hoja.fillna('').values.tolist()

    header_idx = -1

    # 1. Encontrar la cabecera
    for i, row in enumerate(lines):
        row_str = [str(x).strip() for x in row]
        if "Bajo logro educativo" in row_str:
            header_idx = i
            break

    if header_idx == -1: return pd.DataFrame()

    header_row = lines[header_idx]
    year_idx = -1
    comuna_idx = -1
    codigo_idx = -1
    indicadores_indices = {}

    # 2. Mapear las columnas
    for idx, val in enumerate(header_row):
        val_str = str(val).strip()
        if val_str.lower() == 'año':
            year_idx = idx
        elif val_str.lower() in ['nombre', 'comuna - corregimiento', 'comuna']:
            comuna_idx = idx
        elif val_str.lower() == 'código':
            codigo_idx = idx
        elif val_str and val_str not in ['Código', 'Regresar'] and "Pobreza" not in val_str:
            indicadores_indices[val_str] = idx

    # 3. Extraer los datos
    for row in lines[header_idx + 1:]:
        if year_idx >= len(row) or comuna_idx >= len(row) or codigo_idx >= len(row) or codigo_idx == -1:
            continue

        year_val = str(row[year_idx]).strip().replace('.0', '')
        comuna_name = str(row[comuna_idx]).strip()
        codigo = str(row[codigo_idx]).strip().replace('.0', '')

        try:
            cod_int = int(codigo)
            is_valid = cod_int > 0
        except ValueError:
            is_valid = False

        if is_valid and year_val.isdigit() and len(year_val) == 4:
            # Guardar exclusivamente el nombre de la comuna
            comuna_formateada = comuna_name

            for ind_name, idx in indicadores_indices.items():
                if idx < len(row):
                    val = str(row[idx]).strip()
                    if val and val.lower() not in ['nd', 'n.d.', '', 'nan', ' ']:
                        records.append({
                            'Comuna': comuna_formateada,
                            'Año': int(year_val),
                            'Indicador': f"{ind_name}",
                            'Valor': val
                        })

    return pd.DataFrame(records)


def run_ipm_pipeline(input_path: str, output_path: str):
    """
    Función principal para orquestar la extracción de indicadores del IPM.
    """
    logger.info(f"Cargando el archivo de IPM desde: {input_path}")
    hojas_a_cargar = ["IPM 2010 - 2024", "Dimensiones 2010 - 2024"]

    try:
        dict_hojas = pd.read_excel(input_path, sheet_name=hojas_a_cargar, header=None)

        logger.info(f"Extrayendo datos de la hoja: {hojas_a_cargar[0]}...")
        df_ipm_total = procesar_hoja_ipm_total(dict_hojas[hojas_a_cargar[0]])

        logger.info(f"Extrayendo datos de la hoja: {hojas_a_cargar[1]}...")
        df_dimensiones = procesar_hoja_dimensiones(dict_hojas[hojas_a_cargar[1]])

        logger.info("Uniendo ambos DataFrames en formato largo...")
        df_long = pd.concat([df_ipm_total, df_dimensiones], ignore_index=True)

        if df_long.empty:
            logger.warning("No se logró extraer ningún registro. Verifica las condiciones en las funciones.")
            return pd.DataFrame()

        logger.info("Aplicando limpieza numérica y pivoteando a formato ancho...")
        df_long['Valor'] = df_long['Valor'].str.replace(',', '.')
        df_long['Valor'] = pd.to_numeric(df_long['Valor'], errors='coerce')

        df_long['Valor'] = df_long['Valor'].replace(0, np.nan)
        df_long = df_long.dropna(subset=['Valor'])

        df_maestro_ipm = df_long.pivot_table(
            index=['Comuna', 'Año'],
            columns='Indicador',
            values='Valor',
            aggfunc='first'
        ).reset_index()

        df_maestro_ipm.columns.name = None

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        df_maestro_ipm.to_excel(output_path, index=False)

        logger.success(
            f"¡Procesamiento IPM finalizado! Dimensiones: {df_maestro_ipm.shape[0]}x{df_maestro_ipm.shape[1]}")
        logger.info(f"Archivo guardado en: {output_path}")

        return df_maestro_ipm

    except Exception as e:
        logger.error(f"Ocurrió un error procesando el archivo IPM: {e}")
        raise


if __name__ == "__main__":
    RUTA_ENTRADA = "data/raw/HistoricoIPM.xlsx"
    RUTA_SALIDA = "data/processed/IPM_Procesado.xlsx"

    run_ipm_pipeline(RUTA_ENTRADA, RUTA_SALIDA)