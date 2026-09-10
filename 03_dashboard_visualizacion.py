# © 2026 Gabriel Jaime Rivera León. Todos los derechos reservados.
# El diseño arquitectónico, modelado lógico, simulaciones y estructura de este
# software son obra original del autor. La lógica computacional implementada
# comprende pipelines ETL, índices de vulnerabilidad, ensamble XGBoost con
# validación temporal y simulación basada en agentes (ABM).

"""
03_dashboard_visualizacion.py
Dashboard de Ciudades Inteligentes — Plataforma Multi-Ciudad
Ejecutar con:  py -m streamlit run 03_dashboard_visualizacion.py

Funcionalidades:
  · Selector de perfil de ciudad (Medellín / Bogotá / Cali / Barranquilla /
    Bucaramanga / Pereira / Manizales / personalizado)
  · Carga de config.yaml personalizado vía sidebar
  · Todos los textos, etiquetas y popups son dinámicos (unidad territorial del config)
  · Botón "Ejecutar Pipeline Automático" que reentrena el modelo XGBoost
  · En ciudades demo: carga de datos reales (CSV/GeoJSON) + reentrenamiento
"""

import html as html_lib
import io
import os
import sys
import json
import pickle
import shutil
import warnings
import subprocess
import tempfile
import urllib.request
import base64
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── Detección de entorno cloud ─────────────────────────────────────────────────
# En Streamlit Community Cloud la carpeta de telemetría cruda nunca existe.
# Usamos esta flag para deshabilitar el botón de pipeline y evitar errores.
_IS_CLOUD = not (ROOT / "pluviometrica").exists()

# ── Paleta "Resiliencia y Territorio" ─────────────────────────────────────────
# Fuente única de verdad para tarjetas, Plotly y Folium. Esmeralda petróleo
# como acento primario; hueso/arena como lienzo; púrpura para alta
# vulnerabilidad (sin azules corporativos ni rojos alarmistas).
PALETTE = {
    "navy":        "#047857",   # primario: verde esmeralda / petróleo
    "indigo":      "#0F766E",   # secundario: teal territorial
    "indigo_mid":  "#4C1D95",   # acento geográfico: púrpura profundo
    "slate":       "#57534E",   # texto secundario (piedra)
    "danger":      "#4C1D95",   # vulnerabilidad alta (púrpura, no rojo)
    "warning":     "#CA8A04",   # vulnerabilidad media (ocre)
    "success":     "#65A30D",   # vulnerabilidad baja (verde lima)
    "grid":        "#E7E5E4",   # cuadrícula
    "neutral":     "#A8A29E",   # referencias / anotaciones
    "canvas":      "#F5F5F4",   # fondo hueso / arena
    "paper":       "#FFFFFF",   # fondo de gráficos
    "charcoal":    "#1F2937",   # texto de alto contraste
}
# Coropleta IVC: baja = amarillo territorial, alta = púrpura profundo.
# Plotly usa Viridis invertida (nativa); Folium usa YlGnBu (ColorBrewer).
IVC_COLORSCALE = "Viridis_r"
FOLIUM_IVC_FILL = "YlGnBu"
# Carto Positron exige API key (xyzservices muestra "API KEY REQUIRED").
# Esri Light Gray es un lienzo claro equivalente, sin clave.
BASEMAP_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
)
BASEMAP_ATTR = (
    "Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ"
)
PLOTLY_BASEMAP = "open-street-map"
INSTITUTIONAL_COLORWAY = [
    "#047857", "#0F766E", "#4C1D95", "#CA8A04", "#65A30D", "#57534E",
]

# ── Perfiles de ciudad predefinidos ───────────────────────────────────────────
# Cada demo beta apunta a su config_<ciudad>.yaml.
# El pipeline lee rutas absolutas desde ese YAML — no hay hardcode de
# Medellín en los loaders.
CITY_PROFILES = {
    "Medellin": {
        "label":  "Medellín (predeterminado)",
        "config": str(ROOT / "config.yaml"),
        "icon":   "terrain",
    },
    "Bogota": {
        "label":  "Bogotá (beta · localidades + barrios)",
        "config": str(ROOT / "config_bogota.yaml"),
        "icon":   "account_balance",
    },
    "Cali": {
        "label":  "Cali (demo · 22 comunas)",
        "config": str(ROOT / "config_cali.yaml"),
        "icon":   "location_city",
    },
    "Barranquilla": {
        "label":  "Barranquilla (demo · 5 localidades)",
        "config": str(ROOT / "config_barranquilla.yaml"),
        "icon":   "public",
    },
    "Bucaramanga": {
        "label":  "Bucaramanga (demo · 17 comunas)",
        "config": str(ROOT / "config_bucaramanga.yaml"),
        "icon":   "map",
    },
    "Pereira": {
        "label":  "Pereira (demo · 19 comunas + 2 sectores)",
        "config": str(ROOT / "config_pereira.yaml"),
        "icon":   "hub",
    },
    "Manizales": {
        "label":  "Manizales (demo · 11 comunas)",
        "config": str(ROOT / "config_manizales.yaml"),
        "icon":   "terrain",
    },
    "Personalizado": {
        "label":  "Ciudad personalizada (subir config.yaml)",
        "config": None,
        "icon":   "location_city",
    },
}

# ── Página ────────────────────────────────────────────────────────────────────
PRODUCT_NAME = "SMIC Analytics"
PRODUCT_TAGLINE = "Planificación estratégica de inversiones urbanas."
BRAND_DIR = ROOT / "assets" / "brand"
LOGO_PATH = next(
    (p for p in (ROOT / "SMIC_Analytics_logo.png", BRAND_DIR / "smic_logo.png") if p.exists()),
    BRAND_DIR / "smic_logo.png",
)
LOGO_MARK_PATH = BRAND_DIR / "smic_mark.png"
_PAGE_ICON = str(LOGO_PATH) if LOGO_PATH.exists() else ":material/insights:"

st.set_page_config(
    page_title="SMIC Analytics",
    page_icon=_PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# Íconos vectoriales Material Symbols embebidos como paths SVG (Apache-2.0,
# Google). Se incrustan directamente en el código — a diferencia de una
# fuente web externa (Google Fonts) o una etiqueta <link>, esto NO depende
# de ninguna descarga de red en el navegador del usuario, por lo que no
# puede fallar por firewall corporativo, proxy, o saneamiento de DOMPurify
# en st.html() (que elimina <link> pero no <svg> ni <path> inline).
MATERIAL_ICONS: dict[str, str] = {
    "terrain": "m40-240 240-320 180 240h300L560-586 460-454l-50-66 150-200 360 480H40Zm521-80Zm-361 0h160l-80-107-80 107Zm0 0h160-160Z",
    "account_balance": "M200-280v-280h80v280h-80Zm240 0v-280h80v280h-80ZM80-120v-80h800v80H80Zm600-160v-280h80v280h-80ZM80-640v-80l400-200 400 200v80H80Zm178-80h444-444Zm0 0h444L480-830 258-720Z",
    "location_city": "M120-120v-560h240v-80l120-120 120 120v240h240v400H120Zm80-80h80v-80h-80v80Zm0-160h80v-80h-80v80Zm0-160h80v-80h-80v80Zm240 320h80v-80h-80v80Zm0-160h80v-80h-80v80Zm0-160h80v-80h-80v80Zm0-160h80v-80h-80v80Zm240 480h80v-80h-80v80Zm0-160h80v-80h-80v80Z",
    "info": "M440-280h80v-240h-80v240Zm68.5-331.5Q520-623 520-640t-11.5-28.5Q497-680 480-680t-28.5 11.5Q440-657 440-640t11.5 28.5Q463-600 480-600t28.5-11.5ZM480-80q-83 0-156-31.5T197-197q-54-54-85.5-127T80-480q0-83 31.5-156T197-763q54-54 127-85.5T480-880q83 0 156 31.5T763-763q54 54 85.5 127T880-480q0 83-31.5 156T763-197q-54 54-127 85.5T480-80Zm0-80q134 0 227-93t93-227q0-134-93-227t-227-93q-134 0-227 93t-93 227q0 134 93 227t227 93Zm0-320Z",
    "bar_chart": "M640-160v-280h160v280H640Zm-240 0v-640h160v640H400Zm-240 0v-440h160v440H160Z",
    "leaderboard": "M160-200h160v-320H160v320Zm240 0h160v-560H400v560Zm240 0h160v-240H640v240ZM80-120v-480h240v-240h320v320h240v400H80Z",
    "pie_chart": "M520-520h278q-15-110-91.5-186.5T520-798v278Zm-80 358v-636q-121 15-200.5 105.5T160-480q0 122 79.5 212.5T440-162Zm80 0q110-14 187-91t91-187H520v278Zm-40-318Zm0 400q-83 0-156-31.5T197-197q-54-54-85.5-127T80-480q0-83 31.5-156T197-763q54-54 127-85.5T480-880q83 0 155.5 31.5t127 86q54.5 54.5 86 127T880-480q0 82-31.5 155T763-197.5q-54 54.5-127 86T480-80Z",
    "show_chart": "m140-220-60-60 300-300 160 160 284-320 56 56-340 384-160-160-240 240Z",
    "table_chart": "M760-120H200q-33 0-56.5-23.5T120-200v-560q0-33 23.5-56.5T200-840h560q33 0 56.5 23.5T840-760v560q0 33-23.5 56.5T760-120ZM200-640h560v-120H200v120Zm100 80H200v360h100v-360Zm360 0v360h100v-360H660Zm-80 0H380v360h200v-360Z",
    "map": "m600-120-240-84-186 72q-20 8-37-4.5T120-170v-560q0-13 7.5-23t20.5-15l212-72 240 84 186-72q20-8 37 4.5t17 33.5v560q0 13-7.5 23T812-192l-212 72Zm-40-98v-468l-160-56v468l160 56Zm80 0 120-40v-474l-120 46v468Zm-440-10 120-46v-468l-120 40v474Zm440-458v468-468Zm-320-56v468-468Z",
    "query_stats": "m105-399-65-47 200-320 120 140 160-260 120 180 135-214 65 47-198 314-119-179-152 247-121-141-145 233Zm475 159q42 0 71-29t29-71q0-42-29-71t-71-29q-42 0-71 29t-29 71q0 42 29 71t71 29ZM784-80 676-188q-21 14-45.5 21t-50.5 7q-75 0-127.5-52.5T400-340q0-75 52.5-127.5T580-520q75 0 127.5 52.5T760-340q0 26-7 50.5T732-244l108 108-56 56Z",
    "rule": "m576-160-56-56 104-104-104-104 56-56 104 104 104-104 56 56-104 104 104 104-56 56-104-104-104 104Zm79-360L513-662l56-56 85 85 170-170 56 57-225 226ZM80-280v-80h360v80H80Zm0-320v-80h360v80H80Z",
    "hub": "M155-75q-35-35-35-85t35-85q35-35 85-35 14 0 26 3t23 8l57-71q-28-31-39-70t-5-78l-81-27q-17 25-43 40t-58 15q-50 0-85-35T0-580q0-50 35-85t85-35q50 0 85 35t35 85v8l81 28q20-36 53.5-61t75.5-32v-87q-39-11-64.5-42.5T360-840q0-50 35-85t85-35q50 0 85 35t35 85q0 42-26 73.5T510-724v87q42 7 75.5 32t53.5 61l81-28v-8q0-50 35-85t85-35q50 0 85 35t35 85q0 50-35 85t-85 35q-32 0-58.5-15T739-515l-81 27q6 39-5 77.5T614-340l57 70q11-5 23-7.5t26-2.5q50 0 85 35t35 85q0 50-35 85t-85 35q-50 0-85-35t-35-85q0-20 6.5-38.5T624-232l-57-71q-41 23-87.5 23T392-303l-56 71q11 15 17.5 33.5T360-160q0 50-35 85t-85 35q-50 0-85-35Zm-35-465q17 0 28.5-11.5T160-580q0-17-11.5-28.5T120-620q-17 0-28.5 11.5T80-580q0 17 11.5 28.5T120-540Zm148.5 408.5Q280-143 280-160t-11.5-28.5Q257-200 240-200t-28.5 11.5Q200-177 200-160t11.5 28.5Q223-120 240-120t28.5-11.5Zm240-680Q520-823 520-840t-11.5-28.5Q497-880 480-880t-28.5 11.5Q440-857 440-840t11.5 28.5Q463-800 480-800t28.5-11.5ZM480-360q42 0 71-29t29-71q0-42-29-71t-71-29q-42 0-71 29t-29 71q0 42 29 71t71 29Zm268.5 228.5Q760-143 760-160t-11.5-28.5Q737-200 720-200t-28.5 11.5Q680-177 680-160t11.5 28.5Q703-120 720-120t28.5-11.5Zm120-420Q880-563 880-580t-11.5-28.5Q857-620 840-620t-28.5 11.5Q800-597 800-580t11.5 28.5Q823-540 840-540t28.5-11.5ZM480-840ZM120-580Zm360 120Zm360-120ZM240-160Zm480 0Z",
    "target": "M324-111.5Q251-143 197-197t-85.5-127Q80-397 80-480t31.5-156Q143-709 197-763t127-85.5Q397-880 480-880t156 31.5Q709-817 763-763t85.5 127Q880-563 880-480t-31.5 156Q817-251 763-197t-127 85.5Q563-80 480-80t-156-31.5ZM707-253q93-93 93-227t-93-227q-93-93-227-93t-227 93q-93 93-93 227t93 227q93 93 227 93t227-93Zm-397-57q-70-70-70-170t70-170q70-70 170-70t170 70q70 70 70 170t-70 170q-70 70-170 70t-170-70Zm283-57q47-47 47-113t-47-113q-47-47-113-47t-113 47q-47 47-47 113t47 113q47 47 113 47t113-47Zm-169.5-56.5Q400-447 400-480t23.5-56.5Q447-560 480-560t56.5 23.5Q560-513 560-480t-23.5 56.5Q513-400 480-400t-56.5-23.5Z",
    "verified": "m344-60-76-128-144-32 14-148-98-112 98-112-14-148 144-32 76-128 136 58 136-58 76 128 144 32-14 148 98 112-98 112 14 148-144 32-76 128-136-58-136 58Zm34-102 102-44 104 44 56-96 110-26-10-112 74-84-74-86 10-112-110-24-58-96-102 44-104-44-56 96-110 24 10 112-74 86 74 84-10 114 110 24 58 96Zm102-318Zm-42 142 226-226-56-58-170 170-86-84-56 56 142 142Z",
    "trending_down": "M640-240v-80h104L536-526 376-366 80-664l56-56 240 240 160-160 264 264v-104h80v240H640Z",
    "trending_up": "m136-240-56-56 296-298 160 160 208-206H640v-80h240v240h-80v-104L536-320 376-480 136-240Z",
    "speed": "M480-316.5q38-.5 56-27.5l224-336-336 224q-27 18-28.5 55t22.5 61q24 24 62 23.5Zm0-483.5q59 0 113.5 16.5T696-734l-76 48q-33-17-68.5-25.5T480-720q-133 0-226.5 93.5T160-400q0 42 11.5 83t32.5 77h552q23-38 33.5-79t10.5-85q0-36-8.5-70T766-540l48-76q30 47 47.5 100T880-406q1 57-13 109t-41 99q-11 18-30 28t-40 10H204q-21 0-40-10t-30-28q-26-45-40-95.5T80-400q0-83 31.5-155.5t86-127Q252-737 325-768.5T480-800Zm7 313Z",
    "warning": "m40-120 440-760 440 760H40Zm138-80h604L480-720 178-200Zm330.5-51.5Q520-263 520-280t-11.5-28.5Q497-320 480-320t-28.5 11.5Q440-297 440-280t11.5 28.5Q463-240 480-240t28.5-11.5ZM440-360h80v-200h-80v200Zm40-100Z",
    "priority_high": "M480-120q-33 0-56.5-23.5T400-200q0-33 23.5-56.5T480-280q33 0 56.5 23.5T560-200q0 33-23.5 56.5T480-120Zm-80-240v-480h160v480H400Z",
    "public": "M324-111.5Q251-143 197-197t-85.5-127Q80-397 80-480t31.5-156Q143-709 197-763t127-85.5Q397-880 480-880t156 31.5Q709-817 763-763t85.5 127Q880-563 880-480t-31.5 156Q817-251 763-197t-127 85.5Q563-80 480-80t-156-31.5ZM440-162v-78q-33 0-56.5-23.5T360-320v-40L168-552q-3 18-5.5 36t-2.5 36q0 121 79.5 212T440-162Zm276-102q41-45 62.5-100.5T800-480q0-98-54.5-179T600-776v16q0 33-23.5 56.5T520-680h-80v80q0 17-11.5 28.5T400-560h-80v80h240q17 0 28.5 11.5T600-440v120h40q26 0 47 15.5t29 40.5Z",
    "gps_fixed": "M440-42v-80q-125-14-214.5-103.5T122-440H42v-80h80q14-125 103.5-214.5T440-838v-80h80v80q125 14 214.5 103.5T838-520h80v80h-80q-14 125-103.5 214.5T520-122v80h-80Zm238-240q82-82 82-198t-82-198q-82-82-198-82t-198 82q-82 82-82 198t82 198q82 82 198 82t198-82Zm-311-85q-47-47-47-113t47-113q47-47 113-47t113 47q47 47 47 113t-47 113q-47 47-113 47t-113-47Zm169.5-56.5Q560-447 560-480t-23.5-56.5Q513-560 480-560t-56.5 23.5Q400-513 400-480t23.5 56.5Q447-400 480-400t56.5-23.5ZM480-480Z",
    "balance": "M80-120v-80h360v-447q-26-9-45-28t-28-45H240l120 280q0 50-41 85t-99 35q-58 0-99-35t-41-85l120-280h-80v-80h247q12-35 43-57.5t70-22.5q39 0 70 22.5t43 57.5h247v80h-80l120 280q0 50-41 85t-99 35q-58 0-99-35t-41-85l120-280H593q-9 26-28 45t-45 28v447h360v80H80Zm585-320h150l-75-174-75 174Zm-520 0h150l-75-174-75 174Zm335-280q17 0 28.5-11.5T520-760q0-17-11.5-28.5T480-800q-17 0-28.5 11.5T440-760q0 17 11.5 28.5T480-720Z",
    "looks_one": "M480-280h80v-400H400v80h80v320ZM200-120q-33 0-56.5-23.5T120-200v-560q0-33 23.5-56.5T200-840h560q33 0 56.5 23.5T840-760v560q0 33-23.5 56.5T760-120H200Zm0-80h560v-560H200v560Zm0-560v560-560Z",
    "looks_two": "M360-280h240v-80H440v-80h80q33 0 56.5-23.5T600-520v-80q0-33-23.5-56.5T520-680H360v80h160v80h-80q-33 0-56.5 23.5T360-440v160ZM200-120q-33 0-56.5-23.5T120-200v-560q0-33 23.5-56.5T200-840h560q33 0 56.5 23.5T840-760v560q0 33-23.5 56.5T760-120H200Zm0-80h560v-560H200v560Zm0-560v560-560Z",
    "looks_3": "M360-280h160q33 0 56.5-23.5T600-360v-60q0-26-17-43t-43-17q26 0 43-17t17-43v-60q0-33-23.5-56.5T520-680H360v80h160v80h-80v80h80v80H360v80ZM200-120q-33 0-56.5-23.5T120-200v-560q0-33 23.5-56.5T200-840h560q33 0 56.5 23.5T840-760v560q0 33-23.5 56.5T760-120H200Zm0-80h560v-560H200v560Zm0-560v560-560Z",
    "looks_4": "M520-280h80v-400h-80v160h-80v-160h-80v240h160v160ZM200-120q-33 0-56.5-23.5T120-200v-560q0-33 23.5-56.5T200-840h560q33 0 56.5 23.5T840-760v560q0 33-23.5 56.5T760-120H200Zm0-80h560v-560H200v560Zm0-560v560-560Z",
    "looks_5": "M360-280h160q33 0 56.5-23.5T600-360v-80q0-33-23.5-56.5T520-520h-80v-80h160v-80H360v240h160v80H360v80ZM200-120q-33 0-56.5-23.5T120-200v-560q0-33 23.5-56.5T200-840h560q33 0 56.5 23.5T840-760v560q0 33-23.5 56.5T760-120H200Zm0-80h560v-560H200v560Zm0-560v560-560Z",
    "cloud_upload": "M260-160q-91 0-155.5-63T40-377q0-78 47-139t123-78q25-92 100-149t170-57q117 0 198.5 81.5T760-520q69 8 114.5 59.5T920-340q0 75-52.5 127.5T740-160H520q-33 0-56.5-23.5T440-240v-206l-64 62-56-56 160-160 160 160-56 56-64-62v206h220q42 0 71-29t29-71q0-42-29-71t-71-29h-60v-80q0-83-58.5-141.5T480-720q-83 0-141.5 58.5T280-520h-20q-58 0-99 41t-41 99q0 58 41 99t99 41h100v80H260Zm220-280Z",
    "account_tree": "M600-120v-120H440v-400h-80v120H80v-320h280v120h240v-120h280v320H600v-120h-80v320h80v-120h280v320H600ZM160-760v160-160Zm520 400v160-160Zm0-400v160-160Zm0 160h120v-160H680v160Zm0 400h120v-160H680v160ZM160-600h120v-160H160v160Z",
    "menu_book": "M560-564v-68q33-14 67.5-21t72.5-7q26 0 51 4t49 10v64q-24-9-48.5-13.5T700-600q-38 0-73 9.5T560-564Zm0 220v-68q33-14 67.5-21t72.5-7q26 0 51 4t49 10v64q-24-9-48.5-13.5T700-380q-38 0-73 9t-67 27Zm0-110v-68q33-14 67.5-21t72.5-7q26 0 51 4t49 10v64q-24-9-48.5-13.5T700-490q-38 0-73 9.5T560-454ZM260-320q47 0 91.5 10.5T440-278v-394q-41-24-87-36t-93-12q-36 0-71.5 7T120-692v396q35-12 69.5-18t70.5-6Zm260 42q44-21 88.5-31.5T700-320q36 0 70.5 6t69.5 18v-396q-33-14-68.5-21t-71.5-7q-47 0-93 12t-87 36v394Zm-40 118q-48-38-104-59t-116-21q-42 0-82.5 11T100-198q-21 11-40.5-1T40-234v-482q0-11 5.5-21T62-752q46-24 96-36t102-12q58 0 113.5 15T480-740q51-30 106.5-45T700-800q52 0 102 12t96 36q11 5 16.5 15t5.5 21v482q0 23-19.5 35t-40.5 1q-37-20-77.5-31T700-240q-60 0-116 21t-104 59ZM280-494Z",
    "explore": "m300-300 280-80 80-280-280 80-80 280Zm180-120q-25 0-42.5-17.5T420-480q0-25 17.5-42.5T480-540q25 0 42.5 17.5T540-480q0 25-17.5 42.5T480-420Zm0 340q-83 0-156-31.5T197-197q-54-54-85.5-127T80-480q0-83 31.5-156T197-763q54-54 127-85.5T480-880q83 0 156 31.5T763-763q54 54 85.5 127T880-480q0 83-31.5 156T763-197q-54 54-127 85.5T480-80Zm0-80q133 0 226.5-93.5T800-480q0-133-93.5-226.5T480-800q-133 0-226.5 93.5T160-480q0 133 93.5 226.5T480-160Zm0-320Z",
    "upload_file": "M440-200h80v-167l64 64 56-57-160-160-160 160 57 56 63-63v167ZM240-80q-33 0-56.5-23.5T160-160v-640q0-33 23.5-56.5T240-880h320l240 240v480q0 33-23.5 56.5T720-80H240Zm280-520v-200H240v640h480v-440H520ZM240-800v200-200 640-640Z",
    "checklist": "M222-200 80-342l56-56 85 85 170-170 56 57-225 226Zm0-320L80-662l56-56 85 85 170-170 56 57-225 226Zm298 240v-80h360v80H520Zm0-320v-80h360v80H520Z",
    "dataset": "M280-280h160v-160H280v160Zm240 0h160v-160H520v160ZM280-520h160v-160H280v160Zm240 0h160v-160H520v160ZM200-120q-33 0-56.5-23.5T120-200v-560q0-33 23.5-56.5T200-840h560q33 0 56.5 23.5T840-760v560q0 33-23.5 56.5T760-120H200Zm0-80h560v-560H200v560Zm0-560v560-560Z",
    "play_arrow": "M320-200v-560l440 280-440 280Zm80-280Zm0 134 210-134-210-134v268Z",
}


def icon(name: str, size: str = "1.15em", color: str = "currentColor") -> str:
    """Devuelve el HTML de un ícono vectorial SVG embebido (sin emojis y
    sin depender de ninguna fuente/recurso externo — ver MATERIAL_ICONS).

    Uso: st.markdown(f"{icon('info')} Texto", unsafe_allow_html=True)
    """
    path_d = MATERIAL_ICONS.get(name, MATERIAL_ICONS["info"])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960" '
        f'style="width:{size};height:{size};vertical-align:-0.15em;'
        f'fill:{color};flex-shrink:0;" aria-hidden="true">'
        f'<path d="{path_d}"/></svg>'
    )


@st.cache_data(show_spinner=False)
def _logo_data_uri() -> str | None:
    """Data URI del logotipo para incrustarlo en HTML (misma caja que el eslogan)."""
    if not LOGO_PATH.exists():
        return None
    b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


LOGO_DATA_URI = _logo_data_uri()


def render_brand_header() -> None:
    """Kicker de eslogan bajo la barra global (el logo vive en la topbar)."""
    slogan = html_lib.escape(PRODUCT_TAGLINE)
    st.html(
        '<div class="brand-kicker">'
        '<span class="brand-kicker-bar" aria-hidden="true"></span>'
        "<h2>Planificación estratégica<br>de inversiones urbanas.</h2>"
        "</div>"
        '<div class="brand-lockup" style="padding-bottom:0;margin-top:.55rem;"></div>'
        f'<span class="visually-hidden">{html_lib.escape(PRODUCT_NAME)} — {slogan}</span>'
    )


def render_section_title(
    title: str,
    icon_name: str = "bar_chart",
    help_title: str | None = None,
    help_body: str | None = None,
) -> None:
    """Título de sección con ícono vectorial y, opcionalmente, un botón de
    ayuda técnica tipo "info" (evita saturar el gráfico con texto
    explicativo). El botón se renderiza sin etiqueta —solo el ícono— y
    alineado verticalmente al centro del título, siguiendo el patrón
    estándar de "ayuda contextual" (icon-only, esquina superior derecha
    de la sección)."""
    if help_body:
        col_t, col_h = st.columns([0.95, 0.05], vertical_alignment="center")
        with col_t:
            st.html(f'<p class="section-title">{icon(icon_name)} {title}</p>')
        with col_h:
            with st.popover("", icon=":material/info:", help="Más información"):
                if help_title:
                    st.markdown(f"**{help_title}**")
                st.caption(help_body)
    else:
        st.html(f'<p class="section-title">{icon(icon_name)} {title}</p>')


def render_kpi_card(
    label: str,
    value: str,
    sub: str = "",
    icon_name: str | None = None,
    delta: str | None = None,
    delta_positive: bool = True,
    help_text: str | None = None,
) -> None:
    """Renderiza una tarjeta KPI con el estilo institucional único (.kpi-card).

    Se usa en TODAS las secciones del dashboard (Vision General, sidebar,
    Interpretabilidad, Simulacion ABM) para garantizar identidad visual
    consistente, en lugar de mezclar st.metric() nativo con tarjetas HTML.
    """
    label_icon = icon(icon_name, size="1rem", color="white") if icon_name else ""
    delta_html = ""
    if delta:
        delta_cls = "kpi-delta-pos" if delta_positive else "kpi-delta-neg"
        delta_html = f'<div class="kpi-sub {delta_cls}">{delta}</div>'
    title_attr = (
        f' title="{html_lib.escape(help_text, quote=True)}"' if help_text else ""
    )
    st.html(
        f'<div class="kpi-card"{title_attr}>'
        f'<div class="kpi-label">{label_icon}{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'{delta_html}'
        f'</div>'
    )
    if help_text:
        # Caption mínimo: el (i) nativo de Streamlit encapsula la definición
        # técnica sin recargar la tarjeta ni el gráfico principal.
        st.caption("Definición", help=help_text)


def apply_chart_theme(fig) -> None:
    """Aplica el tema claro institucional a cualquier figura Plotly.

    No altera series de datos: solo fondo, tipografía, grilla y colorway.
    """
    fig.update_layout(
        plot_bgcolor=PALETTE["paper"],
        paper_bgcolor=PALETTE["paper"],
        font=dict(color=PALETTE["charcoal"], size=12),
        colorway=INSTITUTIONAL_COLORWAY,
    )
    fig.update_xaxes(showgrid=True, gridcolor=PALETTE["grid"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=PALETTE["grid"], zeroline=False)


def city_kpi_specs(df_ranking, df_latest, ut_col: str, ut_plu: str) -> list[tuple]:
    """Especificación única de KPIs de ciudad (resumen y mapa).

    Solo ensambla etiquetas ya calculadas; no recalcula el modelo.
    """
    if df_ranking is None or df_latest is None or df_latest.empty:
        return []
    mean_ivc = (
        df_latest["ivc_pred"].mean()
        if "ivc_pred" in df_latest.columns
        else df_ranking["mean_ivc"].mean()
    )
    max_ivc = df_ranking["mean_ivc"].max()
    min_ivc = df_ranking["mean_ivc"].min()
    n_terr = (
        df_ranking[ut_col].nunique()
        if ut_col in df_ranking.columns
        else len(df_ranking)
    )
    alta_count = (
        int((df_latest["prioridad"] == "Alta").sum())
        if "prioridad" in df_latest.columns
        else "—"
    )
    worst = (
        df_ranking.iloc[0][ut_col]
        if ut_col in df_ranking.columns
        else "—"
    )
    return [
        ("Índice promedio (ciudad)", f"{mean_ivc:.3f}",
         "Media territorial del panel", "speed",
         "IVC medio de la ciudad en [0, 1], donde mayor implica más vulnerabilidad. "
         "Un valor de 0,50 indica posición intermedia en el panel; 0,46 es "
         "ligeramente inferior (mejor relativo). No es un umbral normativo "
         "nacional: se interpreta frente al resto de "
         f"{ut_plu.lower()} de la misma ciudad y el mismo año."),
        ("Máximo territorial", f"{max_ivc:.3f}",
         f"{ut_col} más vulnerable: {worst}", "warning",
         "Máximo del IVC medio histórico entre unidades territoriales. Identifica "
         "el territorio más crítico del ranking, no un pronóstico puntual."),
        (f"{ut_plu} prioritarias", str(alta_count),
         "Prioridad Alta (tercil superior)", "priority_high",
         "Conteo de unidades en el tercil superior del IVC estimado del último "
         "año. Alta / Media / Baja son cortes relativos (terciles), no umbrales "
         "absolutos de política."),
        (f"Total {ut_plu} analizadas", str(n_terr),
         f"Mínimo observado: {min_ivc:.3f}", "public",
         f"Cardinalidad del panel territorial. El mínimo ({min_ivc:.3f}) es el "
         "IVC medio histórico más bajo; valores cercanos a 0 indican menor "
         "privación relativa, no ausencia de necesidad."),
    ]


def render_kpi_row(specs: list[tuple]) -> None:
    """Fila de tarjetas KPI con el mismo estilo en todas las pestañas."""
    if not specs:
        return
    cols = st.columns(len(specs))
    for col, spec in zip(cols, specs):
        label, value, sub, ic = spec[0], spec[1], spec[2], spec[3]
        help_text = spec[4] if len(spec) > 4 else None
        with col:
            render_kpi_card(label, value, sub, icon_name=ic, help_text=help_text)


# ── CSS ───────────────────────────────────────────────────────────────────────
# NOTA: se usa st.html() (no st.markdown) porque el parser de Markdown de
# Streamlit puede interpretar una línea en blanco dentro de un bloque
# <style>...</style> como el fin del bloque HTML crudo, filtrando el resto
# del CSS como texto plano visible. st.html() inyecta el HTML/CSS sin pasar
# por el parser de Markdown, evitando ese problema por completo.
st.html("""
<style>
/* Tokens SMIC — mismos nombres y valores que assets/splash/splash.css :root */
:root{
  color-scheme:light;
  --smic-emerald:#047857;
  --smic-teal:#0F766E;
  --smic-canvas:#F5F5F4;
  --smic-sidebar:#E5E7EB;
  --smic-paper:#ffffff;
  --smic-charcoal:#1F2937;
  --smic-ink:#111827;
  --smic-stone:#57534E;
  --smic-border:#E5E7EB;
  --smic-font:"Segoe UI","San Francisco",Roboto,Helvetica,Arial,sans-serif;
  --smic-fs-lede:1.05rem;
  --smic-fs-body:.95rem;
  --smic-fs-label:.92rem;
  --smic-fs-caption:.78rem;
  --smic-lh:1.5;
  --smic-space-2:.5rem;
  --smic-space-3:.75rem;
  --smic-space-4:1rem;
  --smic-space-5:1.25rem;
  --smic-radius:12px;
  --smic-focus-ring:2px solid var(--smic-emerald);
  --smic-focus-offset:2px;
  --smic-motion:.15s ease;
  --smic-shadow:0 4px 14px rgba(4,120,87,.18);
}
html, body { color-scheme: light; }
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stToolbar"]{
  background-color:var(--smic-canvas) !important;
  color:var(--smic-charcoal) !important;
}
section[data-testid="stSidebar"]{
  background-color:var(--smic-sidebar) !important;
}

.brand-lockup{border-bottom:1px solid #047857;padding-bottom:.85rem;margin-bottom:1.2rem;}
.main-header,.brand-title{font-size:2rem;font-weight:800;color:#047857;
  letter-spacing:.01em;line-height:1.15;margin:0 0 .2rem 0;
  border-bottom:none;padding-bottom:0;}
.sub-header,.brand-tagline{font-size:.95rem;font-weight:400;color:#57534E;
  letter-spacing:.03em;line-height:1.45;margin:0;}
.brand-kicker{
  display:flex;
  align-items:center;
  gap:1rem;
  margin:0;
  max-width:38rem;
  box-sizing:border-box;
  border-left:none;
  padding:0;
}
.brand-kicker-bar{
  display:block;
  flex:0 0 4px;
  width:4px;
  align-self:stretch;
  background:#007B5E;
  border-radius:2px;
}
.brand-kicker h2,
.brand-kicker p{
  margin:0 !important;
  padding:0;
  font-size:1.2rem !important;
  font-weight:600 !important;
  color:#333 !important;
  letter-spacing:.01em;
  line-height:1.35 !important;
}
.sidebar-brand-lockup{
  display:flex;
  flex-direction:column;
  align-items:center;
  text-align:center;
  width:100%;
  margin:0 0 1.05rem 0;
  padding:0 0 1rem 0;
  border-bottom:1px solid #047857;
  box-sizing:border-box;
}
.sidebar-brand-lockup img{
  display:block;
  width:168px;
  height:auto;
  margin:0 auto;
}
.sidebar-brand{
  display:flex;
  flex-direction:column;
  align-items:center;
  width:100%;
  box-sizing:border-box;
  margin:.1rem 0 1.05rem 0;
  padding:.35rem 0 1rem;
  border-bottom:1px solid #047857;
  text-align:center;
}
.sidebar-brand-title{font-size:1.2rem;font-weight:800;color:#047857;
  letter-spacing:.02em;margin:0;line-height:1.2;}
.sidebar-brand-tagline{
  display:block;
  width:168px;
  margin:.4rem auto 0 auto;
  padding:0;
  text-align:center;
  font-size:.78rem;
  font-weight:700 !important;
  color:#111827 !important;
  letter-spacing:0;
  line-height:1.35;
}
section[data-testid="stSidebar"] [data-testid="stHtml"]{
  width:100%;
}
section[data-testid="stSidebar"] [data-testid="stHtml"] p{
  text-align:center !important;
  margin:0 auto !important;
  color:#111827 !important;
  font-weight:700 !important;
}

/* ── Tarjeta KPI unificada — usada en TODAS las secciones ────────────────── */
.kpi-card{background:linear-gradient(135deg,#047857 0%,#0F766E 100%);
  border-radius:12px;padding:1.2rem 1.4rem;color:white;
  box-shadow:0 4px 14px rgba(4,120,87,.22);height:100%;}
.kpi-label{font-size:.78rem;opacity:.9;text-transform:uppercase;letter-spacing:.05em;
  display:flex;align-items:center;gap:.35rem;}
.kpi-value{font-size:2rem;font-weight:700;line-height:1.25;margin-top:.15rem;}
.kpi-sub{font-size:.75rem;opacity:.8;margin-top:.25rem;}
.kpi-delta-pos{color:#a5d6a7;font-weight:600;}
.kpi-delta-neg{color:#ef9a9a;font-weight:600;}

.section-title{font-size:1.1rem;font-weight:700;color:#047857;
  margin-top:.5rem;margin-bottom:.3rem;display:flex;align-items:center;gap:.4rem;}
.pipeline-box{background:#e8f5e9;border-left:4px solid #388e3c;
  border-radius:0 8px 8px 0;padding:1rem 1.2rem;font-size:.85rem;}
div[data-testid="stTabs"] button{font-size:1rem;font-weight:600;}

/* ── Botón "Info" (popover de ayuda contextual) ──────────────────────────
   Ícono-solo, circular y translúcido: se apoya visualmente junto al
   título/gráfico en lugar de un botón rectangular con etiqueta "Info"
   que rompía la alineación y saturaba la interfaz. */
div[data-testid="stPopover"] button{
  border-radius:50%!important;
  border:1px solid transparent!important;
  background:transparent!important;
  min-width:2.1rem!important;
  width:2.1rem;
  height:2.1rem;
  padding:0!important;
  display:flex;align-items:center;justify-content:center;
  box-shadow:none!important;
  transition:background-color .15s ease, border-color .15s ease;
}
div[data-testid="stPopover"] button:hover{
  background:#d1fae5!important;
  border-color:#6ee7b7!important;
}
div[data-testid="stPopover"] button span,
div[data-testid="stPopover"] button p{
  color:#047857!important;
  margin:0!important;
  font-size:1.15rem!important;
}

/* ── Tarjeta introductoria (Vision General) ──────────────────────────────── */
.intro-card{background:#ecfdf5;border:1px solid #a7f3d0;border-left:5px solid #047857;
  border-radius:0 10px 10px 0;padding:1.1rem 1.4rem;margin-bottom:1.1rem;}
.intro-card h4{color:#047857;margin:0 0 .4rem 0;font-size:1.05rem;
  display:flex;align-items:center;gap:.4rem;}
.intro-card p{color:#37474f;font-size:.88rem;line-height:1.55;margin:.3rem 0;}

/* ── Tarjeta de recomendación (política ganadora ABM) ───────────────────── */
.winner-card{background:#ecfdf5;border:1px solid #a7f3d0;border-left:5px solid #047857;
  border-radius:0 12px 12px 0;padding:1.1rem 1.4rem;margin:.8rem 0 1rem 0;}
.winner-card h4{color:#047857;margin:0 0 .35rem 0;font-size:1.05rem;
  display:flex;align-items:center;gap:.4rem;}
.winner-card .winner-label{font-size:1.15rem;font-weight:700;color:#047857;margin:.2rem 0;}
.winner-card p{color:#37474f;font-size:.88rem;line-height:1.5;margin:.25rem 0 0 0;}
.winner-card .winner-metrics{display:flex;flex-wrap:wrap;gap:.75rem 1.25rem;
  margin-top:.55rem;font-size:.82rem;color:#455a64;}
.winner-card .winner-metrics strong{color:#047857;}

/* ── Modo sidebar (Decisión / Admin) ─────────────────────────────────────── */
.mode-hint{font-size:var(--smic-fs-caption);color:var(--smic-stone);margin:.2rem 0 .6rem 0;}

/* ── Landing: barra + hero como una sola composición ─────────────────── */
.smic-topbar-marker{display:none;}
div[data-testid="stVerticalBlock"]:has(> div .smic-topbar-marker):not(:has(> div [data-testid="stVerticalBlock"] .smic-topbar-marker)){
  position:sticky;top:0;z-index:200;
  background:transparent;
  border-bottom:none;
  box-shadow:none;
  padding:.45rem .15rem .35rem;
  margin:0;
  gap:.05rem !important;
}
div[data-testid="stVerticalBlock"]:has(> div .smic-topbar-marker):not(:has(> div [data-testid="stVerticalBlock"] .smic-topbar-marker))
  > div[data-testid="stHorizontalBlock"]{
  align-items:center !important;
}
div[data-testid="stVerticalBlock"]:has(> div .smic-topbar-marker):not(:has(> div [data-testid="stVerticalBlock"] .smic-topbar-marker))
  > div[data-testid="stHorizontalBlock"] > div:last-child{
  display:flex;
  justify-content:flex-end;
  align-items:center;
}
.smic-back-footer-marker{display:none;}
div[data-testid="stVerticalBlock"]:has(> div .smic-back-footer-marker):not(:has(> div [data-testid="stVerticalBlock"] .smic-back-footer-marker)){
  position:fixed;
  right:1.5rem;
  bottom:1.35rem;
  left:auto;
  width:auto;
  z-index:220;
  background:transparent;
  padding:0;
  margin:0;
}
div[data-testid="stVerticalBlock"]:has(> div .smic-back-footer-marker):not(:has(> div [data-testid="stVerticalBlock"] .smic-back-footer-marker))
  [data-testid="stHtml"]{
  display:none !important;
}
div[data-testid="stVerticalBlock"]:has(> div .smic-back-footer-marker):not(:has(> div [data-testid="stVerticalBlock"] .smic-back-footer-marker))
  [data-testid="stButton"]{
  margin:0;
}
div[data-testid="stVerticalBlock"]:has(> div .smic-back-footer-marker):not(:has(> div [data-testid="stVerticalBlock"] .smic-back-footer-marker))
  [data-testid="stButton"] button{
  color:#047857 !important;
  font-weight:600 !important;
  font-size:.92rem !important;
  min-height:2.65rem !important;
  padding:.45rem 1.35rem .45rem 1.05rem !important;
  border-radius:999px !important;
  border:1px solid #d1d5db !important;
  background:#ffffff !important;
  box-shadow:none !important;
  cursor:pointer !important;
}
div[data-testid="stVerticalBlock"]:has(> div .smic-back-footer-marker):not(:has(> div [data-testid="stVerticalBlock"] .smic-back-footer-marker))
  [data-testid="stButton"] button:hover{
  background:#ecfdf5 !important;
  border-color:#007B5E !important;
  color:#007B5E !important;
}
div[data-testid="stVerticalBlock"]:has(> div .smic-back-footer-marker):not(:has(> div [data-testid="stVerticalBlock"] .smic-back-footer-marker))
  [data-testid="stButton"] button:focus-visible{
  outline:2px solid #007B5E !important;
  outline-offset:2px !important;
}
section.main .block-container{
  padding-bottom:4.75rem;
}
div[data-testid="stVerticalBlock"]:has(> div .smic-topbar-marker):not(:has(> div [data-testid="stVerticalBlock"] .smic-topbar-marker))
  > div[data-testid="stHorizontalBlock"] > div:last-child [data-testid="stVerticalBlock"]{
  align-items:flex-end;
  width:100%;
}
div[data-testid="stVerticalBlock"]:has(> div .smic-topbar-marker):not(:has(> div [data-testid="stVerticalBlock"] .smic-topbar-marker))
  [data-testid="stSegmentedControl"]{
  margin-left:auto;
}
div[data-testid="stVerticalBlock"]:has(> div .smic-topbar-marker):not(:has(> div [data-testid="stVerticalBlock"] .smic-topbar-marker))
  [data-testid="stWidgetLabel"]{
  display:none !important;
}
div[data-testid="stVerticalBlock"]:has(> div .smic-topbar-marker):not(:has(> div [data-testid="stVerticalBlock"] .smic-topbar-marker))
  img{
  display:block;
  max-height:8rem;
  width:auto;
  margin:0;
}
div[data-testid="stVerticalBlock"]:has(> div .smic-topbar-marker):not(:has(> div [data-testid="stVerticalBlock"] .smic-topbar-marker))
  [data-testid="stSegmentedControl"] > div{
  background:#f8faf9;
  border:1px solid #e5e7eb;
  border-radius:999px;
  padding:3px;
  gap:2px;
  box-shadow:inset 0 1px 2px rgba(31,41,55,.04);
}
div[data-testid="stVerticalBlock"]:has(> div .smic-topbar-marker):not(:has(> div [data-testid="stVerticalBlock"] .smic-topbar-marker))
  [data-testid="stSegmentedControl"] button,
div[data-testid="stVerticalBlock"]:has(> div .smic-topbar-marker):not(:has(> div [data-testid="stVerticalBlock"] .smic-topbar-marker))
  [data-testid="stSegmentedControl"] [role="radio"]{
  border-radius:999px !important;
  border:1px solid transparent !important;
  font-size:.86rem !important;
  font-weight:600 !important;
  min-height:2.4rem !important;
  padding:.25rem 1.15rem !important;
  transition:background-color .18s ease, color .18s ease, border-color .18s ease,
    box-shadow .18s ease;
}
div[data-testid="stVerticalBlock"]:has(> div .smic-topbar-marker):not(:has(> div [data-testid="stVerticalBlock"] .smic-topbar-marker))
  [data-testid="stSegmentedControl"] button:hover,
div[data-testid="stVerticalBlock"]:has(> div .smic-topbar-marker):not(:has(> div [data-testid="stVerticalBlock"] .smic-topbar-marker))
  [data-testid="stSegmentedControl"] [role="radio"]:hover{
  background:#ecfdf5 !important;
  color:#007B5E !important;
}
div[data-testid="stVerticalBlock"]:has(> div .smic-topbar-marker):not(:has(> div [data-testid="stVerticalBlock"] .smic-topbar-marker))
  [data-testid="stSegmentedControl"] [aria-checked="true"],
div[data-testid="stVerticalBlock"]:has(> div .smic-topbar-marker):not(:has(> div [data-testid="stVerticalBlock"] .smic-topbar-marker))
  [data-testid="stSegmentedControl"] button[kind="primary"]{
  background:#ffffff !important;
  border-color:#007B5E !important;
  color:#007B5E !important;
  box-shadow:0 1px 3px rgba(0,123,94,.16);
}
div[data-testid="stVerticalBlock"]:has(> div .smic-topbar-marker):not(:has(> div [data-testid="stVerticalBlock"] .smic-topbar-marker)) [data-testid="stSegmentedControl"] *:focus-visible,
div[data-testid="stVerticalBlock"]:has(> div .smic-topbar-marker):not(:has(> div [data-testid="stVerticalBlock"] .smic-topbar-marker)) [role="radiogroup"] *:focus-visible,
div[data-testid="stVerticalBlock"]:has(> div .smic-topbar-marker):not(:has(> div [data-testid="stVerticalBlock"] .smic-topbar-marker)) [data-baseweb="radio"] *:focus-visible{
  outline:2px solid #007B5E !important;
  outline-offset:2px !important;
}
.visually-hidden{
  position:absolute;width:1px;height:1px;padding:0;margin:-1px;
  overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0;
}

div[data-testid="stHtml"]:has(.smic-hero),
div[data-testid="stElementContainer"]:has(.smic-hero){
  display:flex;
  justify-content:center;
}
.smic-hero{
  width:min(100%, 36rem);
  margin:1.85rem auto 1.35rem auto;
  padding:0 1.5rem;
  text-align:center;
  display:flex;
  flex-direction:column;
  align-items:center;
  font-family:var(--smic-font);
}
.smic-hero-eyebrow{
  margin:0;
  font-size:.68rem;
  font-weight:700;
  letter-spacing:.18em;
  text-transform:uppercase;
  color:#007B5E;
  line-height:1;
}
.smic-hero-rule{
  display:block;
  width:1.85rem;
  height:3px;
  margin:.7rem 0 .95rem 0;
  background:#007B5E;
  border-radius:999px;
}
.smic-hero-title{
  margin:0 !important;
  padding:0;
  max-width:24rem;
  font-family:var(--smic-font) !important;
  font-size:1.5rem !important;
  font-weight:650 !important;
  font-style:normal !important;
  color:#1f2937 !important;
  letter-spacing:-.02em;
  line-height:1.25 !important;
  text-align:center !important;
}
.smic-hero-lede{
  margin:1rem 0 0 0 !important;
  max-width:31rem;
  font-family:var(--smic-font);
  font-size:1.02rem;
  font-weight:400;
  line-height:1.68;
  color:#57534e;
  text-align:center !important;
}

.role-context{
  max-width:31rem;
  margin:0 auto;
  font-family:var(--smic-font);
  font-size:1.02rem;
  font-weight:400;
  line-height:1.68;
  color:#57534e;
  text-align:center;
}

/* Pie del sidebar: descarga de identidad, no CTA funcional */
section[data-testid="stSidebar"] [data-testid="stDownloadButton"] button{
  background:transparent !important;
  border:none !important;
  box-shadow:none !important;
  color:var(--smic-stone) !important;
  font-size:var(--smic-fs-caption) !important;
  font-weight:500 !important;
  padding:.15rem .2rem !important;
  min-height:0 !important;
  justify-content:flex-start !important;
}
section[data-testid="stSidebar"] [data-testid="stDownloadButton"] button:hover{
  color:var(--smic-emerald) !important;
  background:#ecfdf5 !important;
}
section[data-testid="stSidebar"] [data-testid="stDownloadButton"] button:focus-visible{
  outline:var(--smic-focus-ring) !important;
  outline-offset:var(--smic-focus-offset) !important;
}

/* ── Barra de confianza / branding ───────────────────────────────────────── */
.trust-bar{display:flex;flex-wrap:wrap;gap:.45rem .7rem;align-items:center;
  background:#fff;border:1px solid #e7e5e4;border-radius:10px;
  padding:.55rem .9rem;margin:0 0 1rem 0;font-size:.8rem;color:#455a64;}
.trust-bar .chip{background:#d1fae5;color:#047857;border-radius:999px;
  padding:.15rem .65rem;font-weight:600;font-size:.75rem;}
.trust-bar .chip-muted{background:#f5f5f4;color:#57534E;border-radius:999px;
  padding:.15rem .65rem;font-size:.75rem;}

/* ── Banner de objetivo (post-onboarding) ────────────────────────────────── */
.goal-banner{background:#ecfdf5;border:1px solid #a7f3d0;border-left:4px solid #047857;
  border-radius:0 10px 10px 0;padding:.85rem 1.1rem;margin:0 0 1rem 0;}
.goal-banner b{color:#047857;}
.goal-banner p{margin:.2rem 0 0 0;font-size:.85rem;color:#37474f;line-height:1.45;}

/* ══════════════════════════════════════════════════════════════════════
   DISEÑO RESPONSIVO — la plataforma debe verse y usarse correctamente en
   escritorio, tablet y teléfono. Streamlit ya apila st.columns() de forma
   automática por debajo de ~640px de ancho de viewport; estas reglas
   afinan tipografía, espaciados, áreas táctiles y elementos que sí
   requieren control manual (iframes de Folium, tablas, pestañas largas).
   ══════════════════════════════════════════════════════════════════════ */

/* Nunca permitir scroll horizontal de página completa: cualquier iframe
   (mapa Folium) o imagen debe respetar el ancho de su contenedor. */
iframe, img{max-width:100%;}
div[data-testid="stDataFrame"], div[data-testid="stTable"]{max-width:100%;overflow-x:auto;}

/* ── Tablet (≤ 992px): reduce ligeramente tipografía y paddings ─────────── */
@media (max-width: 992px){
  .main-header,.brand-title{font-size:1.65rem;}
  .kpi-value{font-size:1.7rem;}
  .kpi-card{padding:1rem 1.15rem;}
  .role-context{font-size:1rem;max-width:100%;}
}

/* ── Móvil (≤ 640px): pantalla angosta, todo apilado en una columna ─────── */
@media (max-width: 640px){
  .main-header,.brand-title{font-size:1.3rem;line-height:1.3;}
  .sub-header,.brand-tagline{font-size:.82rem;letter-spacing:.02em;}
  .sidebar-brand-title{font-size:1.05rem;}
  .section-title{font-size:.95rem;}
  .pipeline-box{font-size:.8rem;padding:.8rem 1rem;}

  .kpi-card{padding:.85rem 1rem;border-radius:10px;}
  .kpi-value{font-size:1.35rem;}
  .kpi-label{font-size:.68rem;}
  .kpi-sub{font-size:.66rem;}

  .intro-card{padding:.9rem 1rem;}
  .intro-card h4{font-size:.92rem;}
  .intro-card p{font-size:.8rem;}

  .scenario-badge{padding:.6rem .8rem;}
  .scenario-badge b{font-size:.85rem;}
  .scenario-badge p{font-size:.75rem;}

  /* Pestañas principales: en pantallas angostas los 4 títulos no caben
     en una sola fila; se reduce el texto y se habilita scroll horizontal
     suave (con inercia táctil) en vez de comprimirlos ilegiblemente. */
  div[data-testid="stTabs"] button{font-size:.78rem;padding:.4rem .5rem;}
  div[data-testid="stTabs"] [data-baseweb="tab-list"]{
    overflow-x:auto;-webkit-overflow-scrolling:touch;flex-wrap:nowrap;
  }
  div[data-testid="stTabs"] [data-baseweb="tab-list"] button{flex-shrink:0;}

  .role-context,.smic-hero-lede{font-size:.95rem;line-height:1.55;margin:.75rem auto 0 auto;text-align:center;}
  .smic-hero{margin:1.5rem auto .5rem auto;}
  .smic-hero-title{font-size:1.28rem !important;}
  div[data-testid="stVerticalBlock"]:has(> div .smic-topbar-marker):not(:has(> div [data-testid="stVerticalBlock"] .smic-topbar-marker)){
    padding:.55rem 0 .7rem;
  }
  div[data-testid="stVerticalBlock"]:has(> div .smic-back-footer-marker):not(:has(> div [data-testid="stVerticalBlock"] .smic-back-footer-marker)){
    right:1rem;
    bottom:1rem;
  }
}

/* ── Dispositivos táctiles (móvil/tablet): áreas de toque ≥ 44px según
   pautas de accesibilidad WCAG/Material, sin agrandar el botón en
   escritorio (donde el cursor es preciso). ─────────────────────────────── */
@media (hover: none) and (pointer: coarse){
  div[data-testid="stPopover"] button{
    min-width:2.75rem!important;width:2.75rem;height:2.75rem;
  }
  div[data-testid="stPopover"] button span,
  div[data-testid="stPopover"] button p{font-size:1.3rem!important;}
}

@media (prefers-reduced-motion: reduce){
  div[data-testid="stVerticalBlock"]:has(> div .smic-topbar-marker):not(:has(> div [data-testid="stVerticalBlock"] .smic-topbar-marker))
    button{
    transition:none !important;
  }
}
</style>
""")

from splash_screen import maybe_render_splash
from ui_chrome import (
    render_app_topbar,
    render_role_context,
    maybe_render_service_hub,
    render_dashboard_back_footer,
)

maybe_render_splash()
render_app_topbar(logo_path=LOGO_PATH, product_name=PRODUCT_NAME)
render_role_context()
maybe_render_service_hub()
render_dashboard_back_footer()


# ══════════════════════════════════════════════════════════════════════════════
# GUÍA DE USO — modal de ayuda paso a paso
# ══════════════════════════════════════════════════════════════════════════════

@st.dialog("Guía de Uso de la Plataforma", width="large")
def _show_help_dialog():
    tab_intro, tab_nav, tab_city, tab_gloss = st.tabs([
        ":material/explore: Qué es la plataforma",
        ":material/route: Cómo navegar",
        ":material/location_city: Cargar tu ciudad",
        ":material/menu_book: Glosario",
    ])

    # ── Pestaña 1: Descripción general y arquitectura ──────────────────────
    with tab_intro:
        st.markdown(
            f"{icon('explore', color='#047857')} **Problema de planificación.** "
            "Bajo restricción fiscal, la inversión pública urbana debe "
            "asignarse entre unidades territoriales con privación "
            "desigualmente distribuida. La pregunta operativa es: "
            "*¿en qué comuna, localidad o zona debería invertirse primero, "
            "con qué evidencia se justifica esa prioridad, y qué "
            "trayectoria seguiría la vulnerabilidad según la regla de "
            "reparto del presupuesto?* El marco responde con un índice "
            "compuesto, un modelo predictivo auditado (XGBoost + SHAP) y "
            "una simulación de agentes a diez años.",
            unsafe_allow_html=True,
        )
        st.markdown("**Las tres piezas centrales:**")
        st.markdown(
            f"{icon('leaderboard', color='#047857')} **Índice de "
            "vulnerabilidad** — convierte encuestas y estadísticas "
            "(desempleo, pobreza, vivienda, educación, tejido social, "
            "clima) en un solo número por zona, entre 0 (nada vulnerable) "
            "y 1 (muy vulnerable). Mayor = peor.",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"{icon('hub', color='#047857')} **Modelo predictivo** — "
            "aprende de los datos históricos para estimar la vulnerabilidad "
            "futura de cada zona y explica qué factores pesan más "
            "(detalle técnico en la pestaña correspondiente).",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"{icon('account_tree', color='#047857')} **Comparación de "
            "políticas** — proyecta 10 años qué pasaría con la "
            "vulnerabilidad de cada zona bajo tres formas distintas de "
            "repartir el presupuesto público.",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown(
            f"{icon('account_tree', color='#047857')} **Cómo está "
            "construida (arquitectura):**",
            unsafe_allow_html=True,
        )
        st.caption(
            "Escrita íntegramente en Python y organizada en 5 capas que se "
            "comunican mediante archivos (no hay base de datos ni servidor "
            "externo): **Configuración** (`config.yaml`) → **Datos de "
            "entrada** (Excel/CSV/GeoJSON) → **Procesamiento ETL** "
            "(cálculo de índices) → **Modelado** (XGBoost + ABM) → "
            "**Presentación** (este dashboard). Ningún dato de Medellín "
            "está escrito directamente en el código — todo se lee desde "
            "`config.yaml` a través de `config_loader.py` (principio de "
            "\"única fuente de verdad\"), por lo que la plataforma es "
            "replicable en cualquier ciudad sin modificar el código fuente."
        )
        st.caption(
            "**Stack tecnológico:** Python · Pandas/NumPy · scikit-learn · "
            "XGBoost · SHAP · simulación de agentes · Streamlit · "
            "Plotly · Folium/streamlit-folium · GeoPandas."
        )
        st.markdown(
            f"{icon('checklist', color='#047857')} **¿Qué te permite "
            "hacer?** Diagnosticar la vulnerabilidad social de cada zona, "
            "priorizar territorios objetivamente, predecir y auditar el "
            "modelo (no es una \"caja negra\"), simular políticas públicas "
            "antes de ejecutarlas, y replicar todo el análisis en "
            "cualquier ciudad editando únicamente `config.yaml`.",
            unsafe_allow_html=True,
        )

    # ── Pestaña 2: Cómo navegar el dashboard ────────────────────────────────
    with tab_nav:
        st.markdown(
            f"{icon('looks_one', color='#047857')} **Selecciona una ciudad** "
            "en el panel lateral izquierdo (Medellín, Bogotá, Cali, "
            "Barranquilla, Bucaramanga, Pereira, Manizales, o sube tu propio "
            "`config.yaml`). Todos los textos y cálculos se recalculan "
            "automáticamente para la unidad territorial de esa ciudad "
            "(comuna, localidad, etc.).",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"{icon('looks_two', color='#047857')} **Pestaña “Resumen ejecutivo”**: "
            "revisa los indicadores clave (KPIs), el ranking histórico de "
            "vulnerabilidad, la distribución de prioridades y el detalle por "
            "índice social de la última medición disponible.",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"{icon('looks_3', color='#047857')} **Pestaña “Mapa de priorización”**: "
            "explora el mapa interactivo — pasa el cursor sobre cada zona para "
            "ver su índice de vulnerabilidad estimado y la prioridad de inversión "
            "asignada por el modelo predictivo.",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"{icon('looks_4', color='#047857')} **Pestaña “Detalle técnico”**: "
            "entiende *qué factores* explican más la vulnerabilidad de cada zona "
            "(importancia de variables y atribución SHAP), además de las métricas "
            "de validación del modelo.",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"{icon('looks_5', color='#047857')} **Pestaña “Comparar políticas "
            "(10 años)”**: contrasta tres heurísticas de asignación — "
            "**Focalizado (optimización algorítmica)**, **Distribuido "
            "(igualitario)** y **Adaptativo (híbrido)** — proyectadas a "
            "una década bajo restricción fiscal constante.",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.caption(
            "Busca el ícono de información junto a los títulos de sección y "
            "los gráficos para ver explicaciones técnicas detalladas sin "
            "sobrecargar la visualización principal."
        )

    # ── Pestaña 3: Cómo cargar los datos de una ciudad nueva ────────────────
    with tab_city:
        st.markdown(
            f"{icon('upload_file', color='#047857')} **1. Configura tu "
            "ciudad** — edita `config.yaml` (o duplícalo) con el nombre de "
            "la ciudad, la unidad territorial (`comuna`, `localidad`, "
            "`corregimiento`...), el centro del mapa y la lista completa de "
            "territorios con sus coordenadas. Luego, en el panel lateral, "
            "selecciona **\"Ciudad personalizada\"** y sube tu archivo "
            "`.yaml` directamente desde el navegador.",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"{icon('dataset', color='#047857')} **2. Consigue las bases "
            "de datos** — necesitas indicadores de 5 dimensiones sociales "
            "(desempleo, hábitat, educación, pobreza, tejido social) y, "
            "opcionalmente, datos de lluvia y un mapa de comunas (GeoJSON).",
            unsafe_allow_html=True,
        )
        st.caption(
            "Fuentes típicas en Colombia: **DANE** (Encuesta de Calidad de "
            "Vida, GEIH, Índice de Pobreza Multidimensional), **IDEAM** "
            "(datos climáticos), y el **geoportal de tu alcaldía** (mapa de "
            "comunas/localidades)."
        )
        st.markdown(
            f"{icon('checklist', color='#047857')} **3. Ubica los "
            "archivos** — coloca `ECV_Procesado.xlsx`, `IML_Procesado.xlsx` "
            "e `IPM_Procesado.xlsx` en "
            "`ModeloDatos/ModeloDatos/data/processed/`, con las columnas "
            "`Comuna`/`Año` y al menos un indicador por dimensión.",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"{icon('play_arrow', color='#047857')} "
            "**4. Ejecuta el pipeline** — con un clic, desde el botón "
            "**\"Ejecutar Pipeline Automático\"** en el panel lateral, o "
            "paso a paso por terminal (`01_siata_processor.py` → "
            "`02_vulnerability_indices.py` → `03_xgboost_model.py`).",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.info(
            "La guía completa, con las fuentes de datos exactas, el "
            "formato detallado de cada archivo y la solución de problemas "
            "frecuentes, está en el archivo **`GUIA_NUEVA_CIUDAD.md`** en "
            "la raíz del proyecto.",
            icon=":material/menu_book:",
        )

    # ── Pestaña 4: Glosario de términos y siglas ────────────────────────────
    with tab_gloss:
        st.caption(
            "Referencia rápida. El glosario completo está en la sección 2 "
            "y 3 de `GUIA_NUEVA_CIUDAD.md`."
        )
        with st.expander("Términos generales y de datos", icon=":material/menu_book:"):
            st.markdown(
                "- **Unidad territorial**: forma en que se subdivide la "
                "ciudad (comuna, localidad, corregimiento...).\n"
                "- **IVC (Índice de Vulnerabilidad Compuesto)**: número "
                "entre 0 y 1 que resume la vulnerabilidad social de un "
                "territorio; combina los 6 índices sociales.\n"
                "- **Índice** (`idx_*`): número entre 0 y 1 que resume "
                "varias variables de un mismo tema (ej. pobreza).\n"
                "- **Pipeline**: secuencia automatizada de pasos que "
                "transforma datos crudos en resultados finales.\n"
                "- **Normalización Min-Max**: técnica que lleva cualquier "
                "variable a una escala común de 0 a 1.\n"
                "- **Prioridad de inversión**: clasificación Alta/Media/"
                "Baja asignada según el IVC predicho de cada territorio."
            )
        with st.expander("Modelo predictivo (XGBoost)", icon=":material/hub:"):
            st.markdown(
                "- **R² (coeficiente de determinación)**: qué proporción "
                "de la variabilidad real explica el modelo (más cerca de "
                "1 es mejor).\n"
                "- **RMSE / MAE**: error promedio de predicción, en la "
                "misma escala del IVC (0 a 1); más bajo es mejor.\n"
                "- **Validación cruzada temporal (walk-forward)**: el "
                "modelo siempre se entrena con años pasados y se evalúa "
                "con años futuros.\n"
                "- **Gain Importance**: fracción de la reducción de error "
                "atribuible a cada variable en el ensamble XGBoost (suman 1). "
                "Ejemplo: *Pobreza Gain: 0.3819* = 38,19 % de la capacidad "
                "predictiva; no es un peso del IVC ni un porcentaje de pobreza.\n"
                "- **SHAP** (*SHapley Additive exPlanations*): reparte, de "
                "forma aditiva y axiomáticamente justa, la contribución de "
                "cada variable a cada predicción individual. El ranking "
                "headline usa la media |SHAP| fuera de muestra (CV temporal).\n"
                "- **Sobreajuste (Overfitting)**: cuando el modelo "
                "\"memoriza\" los datos de entrenamiento en vez de "
                "aprender patrones generales."
            )
        with st.expander("Simulación de agentes (ABM)", icon=":material/account_tree:"):
            st.markdown(
                "- **Agente**: entidad autónoma de la simulación "
                "(población, infraestructura, gobierno) que sigue reglas "
                "propias, implementada en Python puro.\n"
                "- **Escenario Focalizado — Optimización algorítmica**: "
                "concentra una fracción ρ del presupuesto en las cinco "
                "unidades de mayor prioridad (peso SHAP o, en su defecto, "
                "IVC) y reparte el remanente de forma uniforme.\n"
                "- **Escenario Distribuido — Igualitario**: asigna a cada "
                "unidad la misma fracción del presupuesto (1/N), sin "
                "priorizar por IVC ni por SHAP.\n"
                "- **Escenario Adaptativo — Híbrido**: repondera según el "
                "IVC corriente penalizado por la eficiencia reciente de "
                "mejora (φ = 1/(1+5g)). Es una heurística independiente, "
                "no una mezcla mecánica de las dos reglas anteriores.\n"
                "- **Bienestar (wellbeing)**: calidad de vida simulada de "
                "la población de un territorio.\n"
                "- **Coeficiente de Variación (CV)**: desigualdad del IVC "
                "entre territorios; bajo = reducción más pareja."
            )
        with st.expander("Siglas y acrónimos frecuentes", icon=":material/menu_book:"):
            st.markdown(
                "- **ABM**: Agent-Based Model (Modelado Basado en Agentes)\n"
                "- **CV**: *Cross-Validation* (validación cruzada, en el "
                "modelo predictivo) **o** *Coeficiente de Variación* "
                "(en la simulación ABM) — revisa el contexto\n"
                "- **DANE**: Departamento Administrativo Nacional de "
                "Estadística\n"
                "- **ECV**: Encuesta de Calidad de Vida\n"
                "- **ETL**: Extract, Transform, Load (Extraer, "
                "Transformar, Cargar)\n"
                "- **IDEAM**: Instituto de Hidrología, Meteorología y "
                "Estudios Ambientales\n"
                "- **IML**: Indicadores de Mercado Laboral\n"
                "- **IPM**: Índice de Pobreza Multidimensional\n"
                "- **IVC**: Índice de Vulnerabilidad Compuesto\n"
                "- **KPI**: Key Performance Indicator (Indicador Clave de "
                "Desempeño)\n"
                "- **R² / RMSE / MAE**: métricas de precisión del modelo "
                "predictivo\n"
                "- **SHAP**: SHapley Additive exPlanations\n"
                "- **SIATA**: Sistema de Alerta Temprana de Medellín y el "
                "Valle de Aburrá\n"
                "- **UT**: Unidad Territorial\n"
                "- **XGBoost**: eXtreme Gradient Boosting\n"
                "- **YAML**: formato de texto usado en `config.yaml`"
            )

    st.markdown("---")
    if st.button("Entendido, cerrar guía", type="primary", width="stretch"):
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ONBOARDING 60s — wizard orientado a decisión (primera visita / botón sidebar)
# ══════════════════════════════════════════════════════════════════════════════

POLICY_GOALS = {
    "media": {
        "label": "Reducir la vulnerabilidad media de la ciudad",
        "tab":   "Comparar políticas (10 años)",
        "hint":  "Mira la tarjeta de recomendación y el índice medio al año 2034.",
    },
    "equidad": {
        "label": "Mejorar la equidad entre territorios",
        "tab":   "Comparar políticas (10 años)",
        "hint":  "Compara el CV territorial: cuanto más bajo, más pareja es la mejora.",
    },
    "peor": {
        "label": "Atender primero el territorio más crítico",
        "tab":   "Mapa de priorización",
        "hint":  "Las zonas en púrpura tienen prioridad Alta; el ranking del resumen lista el top-N.",
    },
}


@st.dialog("SMIC Analytics en 60 segundos", width="large")
def _show_onboarding_dialog(ciudad: str, ut_plu: str):
    """Wizard corto: ciudad → objetivo de política → dónde mirar primero."""
    st.markdown(
        f"**Paso 1 — Ciudad activa:** {ciudad or '—'}  \n"
        f"Unidad de análisis: {ut_plu.lower()}."
    )
    st.caption(
        "Si necesitas otra ciudad, cámbiala en el panel lateral y vuelve a abrir "
        "esta guía."
    )
    st.markdown("---")
    st.markdown("**Paso 2 — ¿Qué quieres lograr en esta sesión?**")
    goal_key = st.radio(
        "Objetivo de política",
        options=list(POLICY_GOALS.keys()),
        format_func=lambda k: POLICY_GOALS[k]["label"],
        key="onboard_goal_choice",
        label_visibility="collapsed",
        help=(
            "El objetivo no altera el modelo: solo sugiere en qué pestaña "
            "concentrar la lectura (mapa vs. simulación a 10 años)."
        ),
    )
    goal = POLICY_GOALS[goal_key]
    st.info(
        f"**Siguiente paso sugerido:** pestaña «{goal['tab']}».  \n{goal['hint']}",
        icon=":material/route:",
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Empezar con este objetivo", type="primary", width="stretch"):
            st.session_state["onboarding_done"] = True
            st.session_state["policy_goal"] = goal_key
            st.rerun()
    with c2:
        if st.button("Omitir por ahora", width="stretch"):
            st.session_state["onboarding_done"] = True
            st.session_state.pop("policy_goal", None)
            st.rerun()


def _pdf_font_paths() -> tuple[Path, Path | None]:
    """Fuente Unicode para la ficha PDF (tildes / ñ)."""
    regular = ROOT / "assets" / "fonts" / "DejaVuSans.ttf"
    bold = ROOT / "assets" / "fonts" / "DejaVuSans-Bold.ttf"
    if regular.exists():
        return regular, bold if bold.exists() else None
    for path in (
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    ):
        if path.exists():
            return path, None
    raise FileNotFoundError(
        "No se encontró fuente TTF Unicode para generar el PDF "
        "(assets/fonts/DejaVuSans.ttf)."
    )


@st.cache_data(show_spinner=False)
def build_brand_manual_pdf(md_text: str) -> bytes:
    """Genera el Manual de Marca SMIC en PDF (bytes) a partir del Markdown."""
    from datetime import datetime
    from fpdf import FPDF
    import re

    raw = md_text
    font_reg, font_bold = _pdf_font_paths()

    class BrandPDF(FPDF):
        def header(self):
            if self.page_no() == 1:
                return
            self.set_font("Ficha", size=8)
            self.set_text_color(4, 120, 87)
            self.cell(0, 6, "SMIC  ·  Manual de marca", align="L")
            self.ln(8)

        def footer(self):
            self.set_y(-12)
            self.set_font("Ficha", size=8)
            self.set_text_color(87, 83, 78)
            self.cell(
                0, 8,
                f"SMIC Analytics  ·  Identidad visual  ·  {datetime.now():%Y-%m-%d}  ·  "
                f"Página {self.page_no()}/{{nb}}",
                align="C",
            )

    pdf = BrandPDF(format="A4", unit="mm")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.add_font("Ficha", "", str(font_reg))
    has_bold = font_bold is not None
    if has_bold:
        pdf.add_font("Ficha", "B", str(font_bold))

    logo = LOGO_PATH
    if logo.exists():
        pdf.image(str(logo), x=87, y=12, w=36)
        pdf.set_y(52)

    def set_bold(size: float) -> None:
        pdf.set_font("Ficha", "B" if has_bold else "", size=size)

    def set_reg(size: float) -> None:
        pdf.set_font("Ficha", "", size=size)

    def _plain(text: str) -> str:
        text = re.sub(r"`([^`]*)`", r"\1", text)
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)
        return text

    def write_md_line(line: str, size: float = 10) -> None:
        set_reg(size)
        pdf.set_text_color(31, 41, 55)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 5.0, _plain(line))

    in_code = False
    for raw_line in raw.splitlines():
        line = raw_line.rstrip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            set_reg(8)
            pdf.set_text_color(87, 83, 78)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 4.2, line if line else " ")
            continue
        if not line.strip() or line.strip() == "---":
            pdf.ln(1.6)
            continue
        if line.startswith("|"):
            cells = [_plain(c.strip()) for c in line.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            set_reg(8.5)
            pdf.set_text_color(51, 65, 85)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 4.4, "  ·  ".join(cells))
            continue
        if line.startswith("# "):
            set_bold(18)
            pdf.set_text_color(4, 120, 87)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 9, _plain(line[2:].strip()))
            pdf.ln(2)
            continue
        if line.startswith("## "):
            pdf.ln(3)
            set_bold(13)
            pdf.set_text_color(4, 120, 87)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 7, _plain(line[3:].strip()))
            pdf.ln(1.5)
            continue
        if line.startswith("#### "):
            pdf.ln(1.5)
            set_bold(10)
            pdf.set_text_color(6, 78, 59)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 5.5, _plain(line[5:].strip()))
            pdf.ln(0.6)
            continue
        if line.startswith("### "):
            pdf.ln(2)
            set_bold(11)
            pdf.set_text_color(6, 78, 59)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 6, _plain(line[4:].strip()))
            pdf.ln(1)
            continue
        if line.startswith("> "):
            set_bold(11)
            pdf.set_text_color(4, 120, 87)
            pdf.set_x(pdf.l_margin + 4)
            pdf.multi_cell(pdf.epw - 4, 6, _plain(line[2:].strip()))
            pdf.ln(1)
            continue
        if line.startswith("- [ ] ") or line.startswith("- [x] "):
            write_md_line("-  " + line[6:])
            continue
        if line.startswith("- "):
            write_md_line("•  " + line[2:])
            continue
        if len(line) > 2 and line[0].isdigit() and line[1] in ".)":
            write_md_line(line)
            continue
        write_md_line(line)

    return bytes(pdf.output())


@st.cache_data(show_spinner=False, hash_funcs={Path: str})
def build_committee_ficha_pdf(
    *,
    ciudad: str,
    ut: str,
    ut_col: str,
    ut_plu: str,
    year: int | None,
    df_ranking: pd.DataFrame | None,
    df_latest: pd.DataFrame,
    metrics: dict | None,
    abm_path: Path,
    scenario_labels: dict,
    scenario_descriptions: dict,
    abm_base: int = 2024,
    abm_final: int = 2034,
    n_top: int = 5,
) -> bytes:
    """Genera la ficha de priorización en PDF (bytes) para comité / POT."""
    from datetime import datetime
    from fpdf import FPDF

    font_reg, font_bold = _pdf_font_paths()

    class FichaPDF(FPDF):
        def footer(self):
            self.set_y(-12)
            self.set_font("Ficha", size=8)
            self.set_text_color(84, 110, 122)
            self.cell(
                0, 8,
                f"SMIC Analytics · UdeA 2026 · {ciudad} · Página {self.page_no()}/{{nb}}",
                align="C",
            )

    pdf = FichaPDF(format="A4", unit="mm")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.add_font("Ficha", "", str(font_reg))
    if font_bold is not None:
        pdf.add_font("Ficha", "B", str(font_bold))
        has_bold = True
    else:
        has_bold = False

    def set_bold(size: float):
        pdf.set_font("Ficha", "B" if has_bold else "", size=size)

    def set_reg(size: float):
        pdf.set_font("Ficha", "", size=size)

    def _reset_x():
        pdf.set_x(pdf.l_margin)

    def h1(text: str):
        _reset_x()
        set_bold(16)
        pdf.set_text_color(26, 35, 126)
        pdf.multi_cell(0, 8, text)
        pdf.ln(2)

    def h2(text: str):
        pdf.ln(2)
        _reset_x()
        set_bold(12)
        pdf.set_text_color(26, 35, 126)
        pdf.multi_cell(0, 7, text)
        pdf.ln(1)

    def body(text: str, size: float = 10):
        _reset_x()
        set_reg(size)
        pdf.set_text_color(55, 71, 79)
        pdf.multi_cell(0, 5.2, text)
        pdf.ln(1)

    def kv_table(rows: list[tuple[str, str]], col1: float = 95):
        set_reg(9)
        pdf.set_text_color(27, 31, 59)
        usable = pdf.epw  # ancho útil entre márgenes
        c1 = min(col1, usable * 0.62)
        c2 = usable - c1
        for i, (k, v) in enumerate(rows):
            if pdf.get_y() > pdf.h - 25:
                pdf.add_page()
            _reset_x()
            fill = i % 2 == 0
            if fill:
                pdf.set_fill_color(244, 246, 251)
            # Texto truncado a ancho de celda para evitar desborde
            pdf.cell(c1, 6.2, f" {k}"[:55], border=0, fill=fill)
            pdf.cell(c2, 6.2, f" {v}"[:55], border=0, fill=fill)
            pdf.ln(6.2)
        _reset_x()
        pdf.ln(2)

    h1(f"Ficha de priorización territorial — {ciudad}")
    body(
        f"Producto: SMIC Analytics  ·  Generada: {datetime.now():%Y-%m-%d %H:%M}\n"
        f"Unidad territorial: {ut}  ·  Año de referencia: {year or '—'}",
        size=9,
    )
    pdf.set_fill_color(232, 245, 233)
    pdf.set_draw_color(165, 214, 167)
    _reset_x()
    set_reg(9)
    pdf.set_text_color(27, 94, 32)
    pdf.multi_cell(
        0, 5,
        "Esta ficha resume evidencia cuantitativa para discusión de comité. "
        "No sustituye el juicio humano ni el proceso de concertación territorial.",
        fill=True,
        border=1,
    )
    pdf.ln(3)

    # ── 1. Indicadores ───────────────────────────────────────────────────────
    h2("1. Indicadores clave")
    if df_ranking is not None and not df_ranking.empty:
        mean_ivc = float(df_ranking["mean_ivc"].mean())
        max_ivc = float(df_ranking["mean_ivc"].max())
        min_ivc = float(df_ranking["mean_ivc"].min())
        worst = (
            df_ranking.iloc[0][ut_col]
            if ut_col in df_ranking.columns
            else "—"
        )
        n_terr = (
            df_ranking[ut_col].nunique()
            if ut_col in df_ranking.columns
            else len(df_ranking)
        )
        alta = (
            int((df_latest["prioridad"] == "Alta").sum())
            if not df_latest.empty and "prioridad" in df_latest.columns
            else "—"
        )
        kv_table([
            ("Índice medio histórico (ciudad)", f"{mean_ivc:.4f}"),
            ("Máximo territorial", f"{max_ivc:.4f} ({worst})"),
            ("Mínimo territorial", f"{min_ivc:.4f}"),
            (f"{ut_plu} con prioridad Alta", str(alta)),
            (f"Total {ut_plu.lower()} en el análisis", str(n_terr)),
        ])

        h2(f"2. Top {n_top} {ut_plu.lower()} más vulnerables (histórico)")
        top_rows = []
        for i, (_, r) in enumerate(df_ranking.head(n_top).iterrows(), 1):
            name = r[ut_col] if ut_col in r.index else r.iloc[0]
            top_rows.append((f"#{i}  {name}", f"{float(r['mean_ivc']):.4f}"))
        kv_table(top_rows, col1=120)
    else:
        body("Sin ranking disponible.")

    # ── 3. Prioridad estimada ────────────────────────────────────────────────
    if not df_latest.empty and "ivc_pred" in df_latest.columns and ut_col in df_latest.columns:
        h2(f"3. Prioridad estimada — año {year}")
        show = (
            df_latest[[ut_col, "ivc_pred", "prioridad"]]
            .sort_values("ivc_pred", ascending=False)
            .head(10)
        )
        kv_table([
            (f"{r[ut_col]}", f"{float(r['ivc_pred']):.4f}  ·  {r['prioridad']}")
            for _, r in show.iterrows()
        ], col1=90)

    # ── 4. Política recomendada ──────────────────────────────────────────────
    h2(f"4. Política recomendada al horizonte {abm_final}")
    if abm_path.exists():
        try:
            df_abm = load_abm_results(str(abm_path))
            base_ivc = float(df_abm["ivc_initial"].mean())
            df_end = df_abm[df_abm["calendar_year"] == abm_final]
            rows = []
            for sc in scenario_labels:
                sub = df_end[df_end["scenario"] == sc]["ivc_current"]
                if sub.empty:
                    continue
                m = float(sub.mean())
                s = float(sub.std()) if len(sub) > 1 else 0.0
                cv = (s / m) if m > 0 else 0.0
                red = (base_ivc - m) / base_ivc * 100
                rows.append((sc, m, cv, red))
            if rows:
                rows.sort(key=lambda t: (t[1], t[2]))
                sc, m, cv, red = rows[0]
                _reset_x()
                set_bold(11)
                pdf.set_text_color(27, 94, 32)
                pdf.multi_cell(0, 6, scenario_labels[sc])
                body(
                    f"Índice medio proyectado: {m:.4f}\n"
                    f"Reducción vs línea base {abm_base}: {red:.1f} %\n"
                    f"CV territorial: {cv:.3f}\n"
                    f"Criterio: menor índice medio; desempate por menor CV.\n"
                    f"{scenario_descriptions.get(sc, '')}"
                )
                h2("Comparativo de las tres políticas")
                kv_table([
                    (
                        scenario_labels[sc2],
                        f"índice {m2:.4f}  ·  CV {cv2:.3f}  ·  Δ {red2:.1f}%",
                    )
                    for sc2, m2, cv2, red2 in sorted(rows, key=lambda t: t[1])
                ], col1=100)
            else:
                body("Simulación sin datos agregables.")
        except Exception as exc:
            body(f"Simulación no disponible ({type(exc).__name__}).")
    else:
        body("Archivo de simulación no encontrado.")

    # ── 5. Confianza ─────────────────────────────────────────────────────────
    if metrics:
        h2("5. Confianza del modelo predictivo")
        kv_table([
            ("R² validación cruzada temporal", f"{metrics.get('mean_R2', float('nan')):.4f}"),
            ("RMSE CV", f"{metrics.get('mean_RMSE', float('nan')):.4f}"),
        ])

    h2("6. Nota metodológica")
    body(
        "• Índice de vulnerabilidad en escala 0–1 (mayor = peor).\n"
        "• Prioridades Alta/Media/Baja = terciles del índice estimado del último año.\n"
        "• La proyección a 10 años asume presupuesto agregado constante entre políticas.\n"
        "• Fuente de datos sociales: series municipales procesadas "
        "(ECV/IPM/IML según configuración)."
    )

    out = pdf.output()
    # fpdf2 puede devolver bytearray
    return bytes(out)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — creación de config demo para Bogotá
# ══════════════════════════════════════════════════════════════════════════════

def _create_bogota_demo(path: Path):
    """Genera un config.yaml de demo para Bogotá si no existe."""
    if path.exists():
        return
    content = """\
ciudad:
  nombre: "Bogotá"
  departamento: "Cundinamarca"
  codigo_dane: "11001"
  unidad_territorial: "localidad"
  crs: "EPSG:4326"
  centro:
    lat: 4.7110
    lon: -74.0721
rutas:
  datos_climaticos: "pluviometrica"
  datos_sociales: "ModeloDatos/ModeloDatos/data"
  procesados: "data/processed"
  modelos: "models"
  logs: "logs"
  geo: "data/geo"
territorios:
  - {id: 1,  nombre: "Usaquen",          lat: 4.7016, lon: -74.0317}
  - {id: 2,  nombre: "Chapinero",        lat: 4.6464, lon: -74.0603}
  - {id: 3,  nombre: "Santa Fe",         lat: 4.5980, lon: -74.0752}
  - {id: 4,  nombre: "San Cristobal",    lat: 4.5643, lon: -74.0768}
  - {id: 5,  nombre: "Usme",             lat: 4.4795, lon: -74.1278}
  - {id: 6,  nombre: "Tunjuelito",       lat: 4.5680, lon: -74.1280}
  - {id: 7,  nombre: "Bosa",             lat: 4.6197, lon: -74.1960}
  - {id: 8,  nombre: "Kennedy",          lat: 4.6280, lon: -74.1536}
  - {id: 9,  nombre: "Fontibon",         lat: 4.6747, lon: -74.1470}
  - {id: 10, nombre: "Engativa",         lat: 4.7010, lon: -74.1140}
  - {id: 11, nombre: "Suba",             lat: 4.7590, lon: -74.0840}
  - {id: 12, nombre: "Barrios Unidos",   lat: 4.6696, lon: -74.0800}
  - {id: 13, nombre: "Teusaquillo",      lat: 4.6513, lon: -74.0830}
  - {id: 14, nombre: "Martires",         lat: 4.6080, lon: -74.0951}
  - {id: 15, nombre: "Antonio Narino",   lat: 4.5958, lon: -74.1039}
  - {id: 16, nombre: "Puente Aranda",    lat: 4.6272, lon: -74.1110}
  - {id: 17, nombre: "Candelaria",       lat: 4.5966, lon: -74.0737}
  - {id: 18, nombre: "Rafael Uribe",     lat: 4.5639, lon: -74.1091}
  - {id: 19, nombre: "Ciudad Bolivar",   lat: 4.5200, lon: -74.1560}
  - {id: 20, nombre: "Sumapaz",          lat: 4.1933, lon: -74.3594}
catalogo_estaciones: {}
"""
    path.write_text(content, encoding="utf-8")


_create_bogota_demo(Path(CITY_PROFILES["Bogota"]["config"]))

# Perfiles demo (datos sintéticos de portabilidad) que admiten carga de datos reales
DEMO_CITY_PROFILES = {
    "Bogota",
    "Cali",
    "Barranquilla",
    "Bucaramanga",
    "Pereira",
    "Manizales",
}

_REQUIRED_IDX_FEATURES = [
    "idx_desempleo",
    "idx_habitat",
    "idx_educacion",
    "idx_riesgo_clima",
    "idx_pobreza",
    "idx_tejido_social",
]


def _normalize_uploaded_indices(df: pd.DataFrame, ut_col: str) -> pd.DataFrame:
    """Normaliza un CSV de índices al esquema del pipeline (UT_COL + ivc)."""
    out = df.copy()
    # UT
    aliases = {
        "comuna", "localidad", "corregimiento", "upz",
        "unidad_territorial", "unidad territorial",
    }
    found = ut_col if ut_col in out.columns else None
    if found is None:
        for c in out.columns:
            key = c.lower().replace(" ", "_")
            if key in aliases or c.lower() in aliases:
                found = c
                break
    if found is None:
        raise ValueError(
            "No se encontró columna de unidad territorial. "
            f"Esperada una de: {ut_col}, Unidad_Territorial, Comuna, Localidad."
        )
    if found != ut_col:
        out = out.rename(columns={found: ut_col})
    if "Año" not in out.columns and "anio" in {c.lower() for c in out.columns}:
        anio = next(c for c in out.columns if c.lower() in ("anio", "año", "year"))
        out = out.rename(columns={anio: "Año"})
    if "Año" not in out.columns:
        raise ValueError("Falta la columna obligatoria 'Año'.")
    if "ivc" not in out.columns and "IVC" in out.columns:
        out = out.rename(columns={"IVC": "ivc"})
    missing = [c for c in _REQUIRED_IDX_FEATURES + ["ivc"] if c not in out.columns]
    if missing:
        raise ValueError(
            "Faltan columnas obligatorias: " + ", ".join(missing)
        )
    return out


def _install_city_real_data(
    *,
    abs_rutas,
    ut_col: str,
    ut_label: str,
    indices_bytes: bytes,
    geo_bytes: bytes | None,
    abm_bytes: bytes | None,
) -> dict:
    """
    Escribe artefactos reales en la carpeta de la ciudad activa y deja listo
    el reentrenamiento XGBoost.
    """
    import io as _io

    processed = Path(abs_rutas.datos_sociales) / "processed"
    models = Path(abs_rutas.modelos)
    geo_dir = Path(abs_rutas.geo)
    evidencia = Path(abs_rutas.procesados).parent / "evidencia"
    for d in (processed, models, geo_dir, evidencia, Path(abs_rutas.logs)):
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(_io.BytesIO(indices_bytes), encoding="utf-8-sig")
    df = _normalize_uploaded_indices(df, ut_col)
    out_idx = processed / "indices_comunas.csv"
    df.to_csv(out_idx, index=False, encoding="utf-8-sig")

    # Ranking histórico rápido (el XGBoost lo regenera)
    ranking = (
        df.groupby(ut_col, as_index=False)["ivc"]
        .mean()
        .rename(columns={"ivc": "mean_ivc"})
        .sort_values("mean_ivc", ascending=False)
    )
    ranking.to_csv(models / "vulnerability_ranking.csv", index=False, encoding="utf-8-sig")

    n_geo = 0
    if geo_bytes:
        # Acepta GeoJSON FeatureCollection; normaliza nombre de unión
        payload = json.loads(geo_bytes.decode("utf-8"))
        if payload.get("type") != "FeatureCollection":
            raise ValueError("El GeoJSON debe ser un FeatureCollection.")
        for ft in payload.get("features") or []:
            props = ft.setdefault("properties", {})
            name = (
                props.get(ut_col)
                or props.get("Unidad_Territorial")
                or props.get("nombre")
                or props.get("NOMBRE")
                or props.get("comnombre")
                or props.get("NOMBRES_CO")
                or props.get("LOCNOMBRE")
            )
            if name:
                props["Unidad_Territorial"] = str(name)
                props["nombre"] = str(name)
                props.setdefault("tipo", ut_label)
        geo_path = geo_dir / "localidades.geojson"
        geo_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        n_geo = len(payload.get("features") or [])

    n_abm = 0
    if abm_bytes:
        abm_df = pd.read_csv(_io.BytesIO(abm_bytes), encoding="utf-8-sig")
        abm_path = evidencia / "resultados_simulacion_escenarios.csv"
        abm_df.to_csv(abm_path, index=False, encoding="utf-8-sig")
        n_abm = len(abm_df)

    return {
        "n_rows": len(df),
        "n_uts": int(df[ut_col].nunique()),
        "years": (int(df["Año"].min()), int(df["Año"].max())),
        "n_geo": n_geo,
        "n_abm": n_abm,
        "indices_path": str(out_idx),
    }


def _run_xgboost_for_config(config_path: str) -> tuple[bool, str]:
    """Reentrena el modelo con el config de la ciudad activa."""
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "PRIORIZA_CONFIG": str(Path(config_path).resolve()),
    }
    script = str(ROOT / "ml" / "03_xgboost_model.py")
    try:
        proc = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(ROOT),
            env=env,
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        return False, "El reentrenamiento superó el tiempo máximo (15 min)."
    except Exception as ex:
        return False, f"Error al lanzar XGBoost: {ex}"

    out = (proc.stdout or "")[-4000:]
    err = (proc.stderr or "")[-2000:]
    if proc.returncode != 0:
        return False, f"XGBoost falló (código {proc.returncode}).\n{err or out}"
    return True, out


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — modo Decisión (funcionario) vs Admin (pipeline / métricas)
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    if LOGO_DATA_URI:
        st.html(
            f'<div class="sidebar-brand-lockup">'
            f'<img src="{LOGO_DATA_URI}" alt="SMIC Analytics" width="168" height="168"/>'
            f'<p class="sidebar-brand-tagline">'
            f'Planificación estratégica<br>de inversiones urbanas.</p>'
            f'</div>'
        )
    elif LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=168)
        st.html(
            '<div class="sidebar-brand">'
            '<p class="sidebar-brand-tagline">'
            "Planificación estratégica<br>de inversiones urbanas.</p>"
            "</div>"
        )
    else:
        st.html(
            f'<div class="sidebar-brand">'
            f'<p class="sidebar-brand-title">{PRODUCT_NAME}</p>'
            f'<p class="sidebar-brand-tagline">{PRODUCT_TAGLINE}</p>'
            f'</div>'
        )
    sidebar_mode = st.session_state.get("sidebar_mode", "Decisión")
    if sidebar_mode not in ("Decisión", "Admin"):
        sidebar_mode = "Decisión"
    st.html(
        '<p class="mode-hint">'
        + (
            "Vista para priorizar territorios y comparar políticas."
            if sidebar_mode == "Decisión"
            else "Herramientas técnicas: pipeline y métricas de validación."
        )
        + "</p>"
    )
    st.markdown("---")

    st.markdown("### Ciudad")

    # ── Selector de perfil ────────────────────────────────────────────────────
    profile_keys   = list(CITY_PROFILES.keys())
    profile_labels = [CITY_PROFILES[k]["label"] for k in profile_keys]
    selected_idx   = st.selectbox(
        "Perfil de ciudad",
        range(len(profile_keys)),
        format_func=lambda i: profile_labels[i],
        key="city_profile_idx",
        help=(
            "Seleccione la ciudad activa. Toda la nomenclatura (comuna, "
            "localidad u otra unidad), los datos y el mapa se leen de su "
            "config.yaml. No altera el código fuente."
        ),
    )
    selected_profile = profile_keys[selected_idx]

    # ── Carga personalizada ───────────────────────────────────────────────────
    custom_config_path = None
    if selected_profile == "Personalizado":
        uploaded = st.file_uploader(
            "Subir config.yaml",
            type=["yaml", "yml"],
            help="El archivo debe seguir la estructura de config.yaml del proyecto.",
        )
        if uploaded is not None:
            tmp_dir  = Path(tempfile.mkdtemp())
            tmp_cfg  = tmp_dir / "config.yaml"
            tmp_cfg.write_bytes(uploaded.read())
            custom_config_path = str(tmp_cfg)
            st.success("config.yaml cargado correctamente.")
        else:
            st.info("Sube un archivo config.yaml para continuar.")

    # ── Determinar ruta de config activa ─────────────────────────────────────
    # IMPORTANTE: si el perfil es "Personalizado" y aún no se ha subido un
    # archivo, NO se debe caer de vuelta al config.yaml de Medellín — antes
    # este fallback silencioso hacía que la plataforma siguiera mostrando
    # "Medellín" aunque el usuario hubiera elegido "Ciudad personalizada".
    awaiting_upload = selected_profile == "Personalizado" and custom_config_path is None
    if selected_profile == "Personalizado":
        active_config_path = custom_config_path
    else:
        active_config_path = CITY_PROFILES[selected_profile]["config"]

    # ── Cargar config ─────────────────────────────────────────────────────────
    if awaiting_upload:
        UT, UT_COL, UT_PLU, CIUDAD = "comuna", "Comuna", "Comunas", None
        _abs      = None
        config_ok = False
    else:
        try:
            from config_loader import get_config
            cfg    = get_config(active_config_path)
            UT     = cfg.unidad_territorial                   # "comuna" / "localidad"
            UT_COL = UT.title()                               # "Comuna" / "Localidad"
            UT_PLU = UT_COL + "s"                             # "Comunas" / "Localidades"
            CIUDAD = cfg.ciudad.nombre
            _abs   = cfg.rutas_absolutas()
            config_ok = True
        except Exception as cfg_err:
            st.error(f"Error en config.yaml: {cfg_err}")
            UT, UT_COL, UT_PLU, CIUDAD = "comuna", "Comuna", "Comunas", "Ciudad"
            _abs       = None
            config_ok  = False

    # ── Indicador de ciudad ───────────────────────────────────────────────────
    if awaiting_upload:
        st.html(
            f'{icon("cloud_upload", size="2rem", color="#90a4ae")}'
            f'&nbsp;&nbsp;<strong style="font-size:1rem;color:#90a4ae;">'
            f'Esperando archivo</strong>'
        )
        st.caption(
            "Sube tu `config.yaml` arriba para activar el dashboard con la "
            "información de tu ciudad."
        )
    else:
        city_icon = CITY_PROFILES[selected_profile].get("icon", "location_city")
        st.html(
            f'{icon(city_icon, size="2rem", color="#047857")}'
            f'&nbsp;&nbsp;<strong style="font-size:1rem">{CIUDAD}</strong>'
        )
        st.markdown(f"**Unidad territorial:** `{UT}`")
        if selected_profile not in ("Medellin", "Personalizado") and config_ok:
            st.caption(
                "Demo de portabilidad: índices/SHAP sintéticos · cartografía "
                "según fuente municipal/OSM (ver carpeta Datos_*_Beta)."
            )
    st.markdown("---")

    # ── Carga de datos REALES (ciudades demo) — siempre visible ───────────────
    run_city_real_data = False
    _up_indices = _up_geo = _up_abm = None
    _is_demo_city = (
        config_ok
        and selected_profile in DEMO_CITY_PROFILES
        and not awaiting_upload
    )
    if _is_demo_city:
        st.markdown("### Datos reales")
        st.caption(
            "Reemplaza la demo sintética con archivos oficiales de la ciudad."
        )
        with st.expander(
            "Adjuntar archivos y actualizar el modelo",
            expanded=True,
        ):
            st.markdown(
                f"**Archivos para {CIUDAD}** — arrástralos o usa *Browse files*:"
            )
            st.markdown(
                f"1. **Obligatorio** — `indices_comunas.csv`  \n"
                f"   Columnas: `Año`, `{UT_COL}` o `Unidad_Territorial`, "
                f"`idx_desempleo`, `idx_habitat`, `idx_educacion`, "
                f"`idx_riesgo_clima`, `idx_pobreza`, `idx_tejido_social`, `ivc`.\n\n"
                f"2. **Recomendado** — GeoJSON de {UT_PLU.lower()}  \n"
                f"   FeatureCollection con nombre en `Unidad_Territorial` / "
                f"`nombre` / `{UT_COL}`.\n\n"
                f"3. **Opcional** — `resultados_simulacion_escenarios.csv` (ABM)."
            )
            _up_indices = st.file_uploader(
                "1. indices_comunas.csv (obligatorio)",
                type=["csv"],
                key=f"real_idx_{selected_profile}",
                help="Panel histórico de vulnerabilidad por territorio y año.",
            )
            _up_geo = st.file_uploader(
                f"2. GeoJSON de {UT_PLU.lower()} (recomendado)",
                type=["geojson", "json"],
                key=f"real_geo_{selected_profile}",
                help="Cartografía oficial de las unidades territoriales.",
            )
            _up_abm = st.file_uploader(
                "3. Resultados ABM (opcional)",
                type=["csv"],
                key=f"real_abm_{selected_profile}",
                help="Simulación de escenarios a 10 años, si ya la tienes.",
            )
            _ready = _up_indices is not None
            if _ready:
                st.success(
                    "CSV obligatorio listo. Pulsa el botón para cargar y reentrenar.",
                    icon=":material/check_circle:",
                )
            else:
                st.warning(
                    "Sube `indices_comunas.csv` para habilitar la ejecución.",
                    icon=":material/upload_file:",
                )
            run_city_real_data = st.button(
                "Ejecutar carga y actualizar modelo",
                type="primary",
                icon=":material/play_arrow:",
                width="stretch",
                disabled=not _ready,
                key=f"btn_run_real_{selected_profile}",
                help=(
                    "Instala los archivos en Datos_*_Beta y reentrena "
                    "XGBoost + SHAP para esta ciudad."
                ),
            )
        st.markdown("---")

    # ── Filtros (modo Decisión — y también Admin para no perder contexto) ────
    st.markdown("### Filtros de visualización")
    _filter_placeholder = st.empty()

    st.markdown("---")

    with st.expander(
        "Guía metodológica (paso a paso)",
        expanded=False,
        icon=":material/route:",
    ):
        st.markdown(
            "**Flujo de lectura recomendado para el tomador de decisión.** "
            "La plataforma no sustituye el juicio institucional: ordena "
            "evidencia territorial para que la asignación presupuestaria "
            "pueda auditarse."
        )
        st.markdown(
            "1. **Ciudad.** En *Perfil de ciudad* elija el municipio. "
            "La nomenclatura (comuna, localidad u otra unidad), el mapa y "
            "los archivos se leen de su `config.yaml`.\n\n"
            "2. **Recorte temporal.** Fije el *Año de análisis* y el "
            f"recorte *Top {UT_PLU} vulnerables* si desea limitar el ranking.\n\n"
            "3. **Resumen ejecutivo.** Lea los indicadores clave, el ranking "
            "histórico y la distribución de prioridades (terciles relativos, "
            "no umbrales nacionales).\n\n"
            "4. **Mapa de priorización.** Contraste la estimación del último "
            "año con la cartografía; el color más intenso señala mayor IVC.\n\n"
            "5. **Detalle técnico.** Audite *Gain Importance* (utilidad en el "
            "entrenamiento XGBoost) y *SHAP* (contribución aditiva fuera de "
            "muestra). No son pesos del IVC.\n\n"
            "6. **Comparar políticas (10 años).** Contraste las tres "
            "heurísticas de asignación — Focalizado (optimización "
            "algorítmica), Distribuido (igualitario) y Adaptativo (híbrido) — "
            "bajo restricción fiscal constante.\n\n"
            "7. **Ficha PDF.** Descargue el expediente para consejo o POT "
            "cuando requiera dejar constancia del recorte de evidencia."
        )
        st.caption(
            "Pulse *Guía de uso de la plataforma* para el glosario, la "
            "carga de una ciudad nueva y la arquitectura del marco."
        )

    # ── GUÍA DE USO + ONBOARDING ──────────────────────────────────────────────
    show_help_guide = st.button(
        "Guía de uso de la plataforma",
        icon=":material/help:",
        width="stretch",
        help="Abre una guía paso a paso sobre cómo interpretar y usar el dashboard.",
    )
    show_onboarding = st.button(
        "Empezar en 60 segundos",
        icon=":material/route:",
        width="stretch",
        help="Guía corta: elige un objetivo de política y te indicamos dónde mirar.",
    )

    # ── ADMIN: pipeline + métricas (oculto en Decisión) ───────────────────────
    run_pipeline = False
    if sidebar_mode == "Admin":
        st.markdown("---")
        st.markdown("### Automatización")
        if selected_profile in DEMO_CITY_PROFILES:
            st.caption(
                "En ciudades demo, usa **Cargar datos reales de la ciudad** "
                "(arriba) para instalar CSV/GeoJSON y reentrenar el modelo. "
                "El pipeline SIATA completo aplica a Medellín."
            )
        if _IS_CLOUD:
            st.info(
                "**Modo nube activo.**\n\n"
                "El reentrenamiento con telemetría SIATA (20 GB) no está "
                "disponible aquí. Sí puedes cargar CSV/GeoJSON reales en "
                "ciudades demo y actualizar el modelo predictivo.",
                icon=":material/cloud_done:",
            )
        elif selected_profile not in DEMO_CITY_PROFILES:
            run_pipeline = st.button(
                "Ejecutar pipeline automático",
                type="primary",
                icon=":material/play_arrow:",
                width="stretch",
                help=(
                    "Ejecuta en secuencia:\n"
                    "1. Procesador SIATA (pipeline ETL / bigdata)\n"
                    "2. Índices de vulnerabilidad (ETL + agregación estadística)\n"
                    "3. Reentrenamiento XGBoost (motor predictivo)\n\n"
                    "Al finalizar, el dashboard se recarga con los nuevos datos."
                ),
            )
        _admin_metrics_placeholder = st.empty()
    else:
        _admin_metrics_placeholder = None

    st.markdown("---")
    st.caption("Trabajo de Grado 2026 · UdeA")
    _brand_md = ROOT / "SMIC_Brand_Manual.md"
    if _brand_md.exists():
        try:
            _brand_pdf = build_brand_manual_pdf(_brand_md.read_text(encoding="utf-8"))
            st.download_button(
                label="Manual de marca (PDF)",
                data=_brand_pdf,
                file_name="SMIC_Brand_Manual.pdf",
                mime="application/pdf",
                icon=":material/menu_book:",
                type="secondary",
                help=(
                    "Identidad visual y de comunicación de SMIC "
                    "(naming, paleta, tipografía). No es la ficha de priorización."
                ),
                key="dl_brand_manual_sidebar",
            )
        except (OSError, ValueError, TypeError, UnicodeDecodeError):
            st.caption("Manual de marca no disponible en PDF.")

if show_help_guide:
    _show_help_dialog()
elif show_onboarding and not awaiting_upload:
    _show_onboarding_dialog(CIUDAD, UT_PLU)

if awaiting_upload:
    # Se corta la ejecución aquí para evitar que el resto del dashboard
    # (KPIs, mapa, modelo) caiga de vuelta a los datos locales de Medellín
    # mientras el usuario no haya subido su propio config.yaml.
    render_brand_header()
    st.html(
        f'<p class="section-title">{icon("cloud_upload", color="#047857")} '
        f'Ciudad personalizada</p>'
    )
    st.info(
        "Aún no se ha cargado ninguna ciudad. Sube un archivo `config.yaml` "
        "válido en el panel lateral (sección **Configuración de Ciudad**) "
        "para activar el dashboard con la información de tu propio territorio.",
        icon=":material/cloud_upload:",
    )
    st.caption(
        "El archivo debe seguir la misma estructura que `config.yaml` "
        "(ciudad, unidad territorial, rutas de datos y lista de territorios)."
    )
    st.stop()


# Aviso visible en el área principal para ciudades demo
if config_ok and selected_profile in DEMO_CITY_PROFILES:
    st.info(
        f"**{CIUDAD} (demo).** Los indicadores de prueba ya están cargados. "
        "Si cuentas con CSV/GeoJSON oficiales, adjúntalos en **Datos reales** "
        "y pulsa **Ejecutar carga y actualizar modelo**.",
        icon=":material/upload_file:",
    )


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS REALES + REENTRENAMIENTO (ciudades demo)
# ══════════════════════════════════════════════════════════════════════════════

if run_city_real_data and config_ok and _up_indices is not None:
    st.markdown("---")
    st.markdown(f"### Actualización de datos reales — {CIUDAD}")

    with st.status("Instalando archivos en la carpeta de la ciudad…", expanded=True) as status:
        try:
            summary = _install_city_real_data(
                abs_rutas=_abs,
                ut_col=UT_COL,
                ut_label=UT,
                indices_bytes=_up_indices.getvalue(),
                geo_bytes=_up_geo.getvalue() if _up_geo is not None else None,
                abm_bytes=_up_abm.getvalue() if _up_abm is not None else None,
            )
            st.write(
                f"Índices: **{summary['n_rows']}** filas · "
                f"**{summary['n_uts']}** {UT_PLU.lower()} · "
                f"años {summary['years'][0]}–{summary['years'][1]}"
            )
            if summary["n_geo"]:
                st.write(f"GeoJSON: **{summary['n_geo']}** polígonos")
            if summary["n_abm"]:
                st.write(f"ABM: **{summary['n_abm']}** filas de simulación")
            status.update(label="Archivos instalados", state="complete")
        except Exception as ex:
            status.update(label="Error al instalar archivos", state="error")
            st.error(f"No se pudieron instalar los archivos: {ex}")
            summary = None

    if summary is not None:
        with st.status("Reentrenando modelo XGBoost + SHAP…", expanded=True) as status:
            ok, log_txt = _run_xgboost_for_config(active_config_path)
            if ok:
                status.update(label="Modelo actualizado", state="complete")
                if log_txt.strip():
                    with st.expander("Salida del reentrenamiento", expanded=False):
                        st.code(log_txt, language="text")
                st.success(
                    f"Datos reales de **{CIUDAD}** cargados y modelo actualizado. "
                    "Recargando el dashboard…"
                )
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
            else:
                status.update(label="Error en el reentrenamiento", state="error")
                st.error(log_txt)
                st.warning(
                    "Los archivos CSV/GeoJSON quedaron instalados, pero el modelo "
                    "no se actualizó. Revisa el mensaje de error (columnas, años "
                    "mínimos para CV temporal, etc.)."
                )
    st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# EJECUTAR PIPELINE (si el botón fue presionado)
# ══════════════════════════════════════════════════════════════════════════════

if run_pipeline:
    pipeline_steps = [
        ("Procesador SIATA",           str(ROOT / "bigdata" / "01_siata_processor.py")),
        ("Indices de Vulnerabilidad",  str(ROOT / "ml"      / "02_vulnerability_indices.py")),
        ("Reentrenamiento XGBoost",    str(ROOT / "ml"      / "03_xgboost_model.py")),
    ]

    st.markdown("---")
    st.markdown("### Pipeline en ejecucion")
    overall_ok = True

    for step_name, script_path in pipeline_steps:
        with st.status(f"Ejecutando: {step_name}...", expanded=True) as status:
            try:
                proc = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=str(ROOT),
                    env={**os.environ, "PYTHONPATH": str(ROOT)},
                    timeout=600,
                )
                stdout = proc.stdout or ""
                stderr = proc.stderr or ""

                if proc.returncode == 0:
                    # st.status ya renderiza un icono nativo segun 'state'
                    # (check verde / cruz roja) — no se requiere prefijo manual.
                    status.update(label=f"{step_name} — completado", state="complete")
                    if stdout.strip():
                        with st.expander("Salida del script", expanded=False):
                            st.code(stdout[-4000:], language="text")
                else:
                    status.update(label=f"{step_name} — error", state="error")
                    st.error(f"El script terminó con código {proc.returncode}")
                    if stderr.strip():
                        st.code(stderr[-3000:], language="text")
                    overall_ok = False
                    break

            except subprocess.TimeoutExpired:
                status.update(label=f"{step_name} — tiempo agotado", state="error")
                st.error("El script tardó más de 10 minutos. Cancela y verifica los datos.")
                overall_ok = False
                break
            except Exception as ex:
                status.update(label=f"{step_name} — error", state="error")
                st.error(f"Error inesperado: {ex}")
                overall_ok = False
                break

    if overall_ok:
        st.success(
            "Pipeline completado. Los modelos han sido reentrenados con los datos "
            f"de {CIUDAD}. Limpiando cache y recargando..."
        )
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    else:
        st.warning(
            "El pipeline se detuvo con errores. Revisa los mensajes anteriores "
            "y verifica que los datos de entrada existen en las rutas del config."
        )
    st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# RUTAS DE ARTEFACTOS (dinámicas desde config)
# ══════════════════════════════════════════════════════════════════════════════

if config_ok:
    MODELS_DIR   = Path(_abs.modelos)
    DATA_PROC    = Path(_abs.datos_sociales) / "processed"
    PROC_DIR     = Path(_abs.procesados)
    GEO_DIR      = Path(_abs.geo)
    _SOCIAL_ROOT = Path(_abs.datos_sociales)
else:
    MODELS_DIR = ROOT / "models"
    DATA_PROC  = ROOT / "ModeloDatos" / "ModeloDatos" / "data" / "processed"
    PROC_DIR   = ROOT / "data" / "processed"
    GEO_DIR    = ROOT / "data" / "geo"
    _SOCIAL_ROOT = DATA_PROC.parent


def _first_existing(*candidates: Path | None) -> Path | None:
    for path in candidates:
        if path is not None and Path(path).exists():
            return Path(path)
    return None


MODEL_PATH   = MODELS_DIR / "vulnerability_model.pkl"
METRICS_PATH = _first_existing(
    MODELS_DIR / "cv_metrics.json",
) or (MODELS_DIR / "cv_metrics.json")
RANKING_PATH = _first_existing(
    MODELS_DIR / "vulnerability_ranking.csv",
    DATA_PROC / "vulnerability_ranking.csv",
    _SOCIAL_ROOT / "vulnerability_ranking.csv",
) or (MODELS_DIR / "vulnerability_ranking.csv")
# Ranking headline SHAP (oficial): estabilidad OOS por CV temporal.
# NO usar shap_values.csv para "determinante dominante" — ese archivo es
# atribución in-sample del modelo reentrenado al 100 % (ver comentario en
# la sección Interpretabilidad).
SHAP_STABILITY_PATH = _first_existing(
    MODELS_DIR / "shap_importance_cv_stability.csv",
) or (MODELS_DIR / "shap_importance_cv_stability.csv")
# Detalle exploratorio por territorio-año (nunca ranking headline):
SHAP_PATH = _first_existing(
    MODELS_DIR / "shap_values.csv",
    _SOCIAL_ROOT / "shap_values.csv",
) or (MODELS_DIR / "shap_values.csv")
INDICES_PATH = _first_existing(
    DATA_PROC / "indices_comunas.csv",
    _SOCIAL_ROOT / "indices_comunas.csv",
    PROC_DIR / "indices_comunas.csv",
) or (DATA_PROC / "indices_comunas.csv")

GEO_CANDIDATES = [
    GEO_DIR / f"{UT}s.geojson",
    GEO_DIR / "localidades.geojson",
    GEO_DIR / "comunas.geojson",
    GEO_DIR / "bogota_localidades.geojson",
    ROOT / "Datos_Bogotá_Beta" / "geo" / "localidades.geojson",
    ROOT / "Datos_Bogotá_Beta" / "bogota_localidades.geojson",
    GEO_DIR / f"{UT}s_{CIUDAD.lower()}.geojson" if CIUDAD else GEO_DIR / f"{UT}s.geojson",
    GEO_DIR / "bogota_dummy.geojson",
    GEO_DIR / f"{UT}s.gpkg",
    ROOT / "Datos_Bogotá_Beta" / "bogota_dummy.geojson",
    ROOT / "data" / "comunas.geojson",
    ROOT / "data" / "comunas_medellin.geojson",
]
# Demos multi-ciudad: <slug>_oficial.geojson en la carpeta Datos_*_Beta
if CIUDAD and config_ok:
    _slug = (
        CIUDAD.lower()
        .replace("á", "a").replace("é", "e").replace("í", "i")
        .replace("ó", "o").replace("ú", "u").replace(" ", "_")
    )
    _demo_root = Path(_abs.geo).parent if _abs is not None else ROOT
    GEO_CANDIDATES[2:2] = [
        _demo_root / f"{_slug}_oficial.geojson",
        GEO_DIR / f"{_slug}_oficial.geojson",
    ]


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS (con cache por ruta de config)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Cargando datos...")
def load_indices(path: str, ut_col: str) -> pd.DataFrame | None:
    if not Path(path).exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    # Normaliza el nombre de la columna de UT al esperado
    aliases = {
        "comuna", "localidad", "corregimiento", "upz",
        "unidad_territorial", "unidad territorial",
    }
    col_found = ut_col if ut_col in df.columns else None
    if col_found is None:
        for c in df.columns:
            if c.lower().replace(" ", "_") in aliases or c.lower() in aliases:
                col_found = c
                break
    if col_found and col_found != ut_col:
        df = df.rename(columns={col_found: ut_col})
    if "ivc" not in df.columns and "IVC" in df.columns:
        df = df.rename(columns={"IVC": "ivc"})
    if "Año" not in df.columns:
        raise ValueError("indices_comunas.csv no contiene la columna 'Año'")
    years = df["Año"].values.astype(float)
    df["año_normalizado"] = (years - years.min()) / max(years.max() - years.min(), 1)
    return df


@st.cache_data(show_spinner=False)
def load_ranking(path: str, ut_col: str) -> pd.DataFrame | None:
    if not Path(path).exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    aliases = {"comuna", "localidad", ut_col.lower(), "unidad_territorial"}
    col_found = next(
        (c for c in df.columns if c.lower().replace(" ", "_") in aliases),
        df.columns[0],
    )
    if col_found != ut_col:
        df = df.rename(columns={col_found: ut_col})
    return df


@st.cache_data(show_spinner=False)
def load_shap(path: str) -> pd.DataFrame | None:
    if not Path(path).exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_metrics(path: str) -> dict | None:
    if not Path(path).exists():
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not Path(path).exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError, OSError):
        return None


@st.cache_resource(show_spinner="Ajustando modelo de prueba...")
def train_fallback_model(indices_path: str, model_path: str, feature_cols: tuple) -> object | None:
    """Entrena un XGBoost ligero con el panel de prueba si no hay .pkl."""
    from xgboost import XGBRegressor

    df = pd.read_csv(indices_path, encoding="utf-8-sig")
    if "ivc" not in df.columns and "IVC" in df.columns:
        df = df.rename(columns={"IVC": "ivc"})
    if "Año" in df.columns:
        years = df["Año"].values.astype(float)
        df["año_normalizado"] = (years - years.min()) / max(years.max() - years.min(), 1)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing or "ivc" not in df.columns:
        return None
    model = XGBRegressor(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        n_jobs=1,
        verbosity=0,
    )
    model.fit(df[list(feature_cols)].values, df["ivc"].values)
    out = Path(model_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(out, "wb") as f:
            pickle.dump(model, f)
    except OSError:
        pass
    return model


def _resolve_abm_ut_col(df: pd.DataFrame) -> str | None:
    aliases = {
        "comuna", "localidad", "corregimiento", "upz", "unidad_territorial",
    }
    for c in df.columns:
        if str(c).lower().replace(" ", "_") in aliases:
            return c
    return None


@st.cache_data(show_spinner=False)
def load_abm_results(path: str, base_year: int = 2024) -> pd.DataFrame:
    """Carga y transforma el CSV de simulación ABM a años calendario."""
    df = pd.read_csv(path)
    if "year" not in df.columns:
        raise ValueError("simulation_results.csv no contiene la columna 'year'")
    df["calendar_year"] = int(base_year) + df["year"]
    return df


@st.cache_data(show_spinner=False)
def load_geodata(candidates: list[str], cache_bust: tuple = ()) -> tuple:
    """
    cache_bust: tupla (ruta, mtime) para invalidar caché cuando el GeoJSON
    se actualiza en disco (p. ej. tras actualizar_geo_medellin.py).
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        from shapely.validation import make_valid

        for path in candidates:
            if Path(path).exists():
                gdf = gpd.read_file(path)
                # Garantizar EPSG:4326 (WGS84) para Folium.
                if gdf.crs is None:
                    gdf = gdf.set_crs(epsg=4326)
                elif gdf.crs.to_epsg() != 4326:
                    gdf = gdf.to_crs(epsg=4326)

                # Rellenar anillos interiores por feature (huecos visibles en Folium)
                def _fill_feature_holes(geom):
                    if geom is None or geom.is_empty:
                        return geom
                    g = make_valid(geom)
                    parts = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
                    filled = []
                    for poly in parts:
                        if not hasattr(poly, "interiors") or not list(poly.interiors):
                            filled.append(poly)
                            continue
                        holes = [Polygon(r) for r in poly.interiors]
                        filled.append(make_valid(unary_union([poly, *holes])))
                    if not filled:
                        return g
                    return filled[0] if len(filled) == 1 else unary_union(filled)

                gdf = gdf.copy()
                gdf["geometry"] = gdf.geometry.map(_fill_feature_holes)
                # ~22 m a esta latitud: recorta GeoJSON de varios MB para Folium
                # sin alterar la lectura de la coropleta urbana.
                gdf["geometry"] = gdf.geometry.simplify(0.0002, preserve_topology=True)
                return gdf, f"local:{Path(path).name}"
    except ImportError:
        pass
    return None, "none"


# ── Cargar artefactos ─────────────────────────────────────────────────────────
errors: list[str] = []

df_indices = df_ranking = df_shap = df_shap_stability = metrics = model = None

try:
    df_indices = load_indices(str(INDICES_PATH), UT_COL)
except Exception as e:
    errors.append(f"indices_comunas.csv: {e}")

try:
    df_ranking = load_ranking(str(RANKING_PATH), UT_COL)
except Exception as e:
    errors.append(f"vulnerability_ranking.csv: {e}")

try:
    # Fuente oficial del ranking de determinantes (media |SHAP| entre folds OOS)
    df_shap_stability = load_shap(str(SHAP_STABILITY_PATH))
except Exception as e:
    errors.append(f"shap_importance_cv_stability.csv: {e}")

try:
    # Solo detalle exploratorio territorio-año; no alimentar ranking headline
    df_shap = load_shap(str(SHAP_PATH))
except Exception as e:
    errors.append(f"shap_values.csv (exploratorio): {e}")

try:
    metrics = load_metrics(str(METRICS_PATH))
except Exception as e:
    errors.append(f"cv_metrics.json: {e}")

try:
    model = load_model(str(MODEL_PATH))
except Exception as e:
    errors.append(f"vulnerability_model.pkl: {e}")
    model = None

if df_ranking is None and df_indices is not None:
    _rank_col = UT_COL if UT_COL in df_indices.columns else None
    if _rank_col is None:
        for c in df_indices.columns:
            if c.lower().replace(" ", "_") in {
                "comuna", "localidad", "unidad_territorial",
            }:
                _rank_col = c
                break
    if _rank_col and "ivc" in df_indices.columns:
        df_ranking = (
            df_indices.groupby(_rank_col, as_index=False)["ivc"]
            .mean()
            .rename(columns={"ivc": "mean_ivc", _rank_col: UT_COL})
            .sort_values("mean_ivc", ascending=False)
        )

_geo_paths = [str(c) for c in GEO_CANDIDATES]
_geo_bust = tuple(
    (p, Path(p).stat().st_mtime if Path(p).exists() else 0.0) for p in _geo_paths
)
# La cartografía (varios MB) solo se materializa en la pestaña del mapa.

# ── Constantes ABM ────────────────────────────────────────────────────────────
ABM_BASE_YEAR  = 2024
ABM_N_STEPS    = 10
ABM_FINAL_YEAR = ABM_BASE_YEAR + ABM_N_STEPS          # 2034
# Preferir simulación de la ciudad activa (p. ej. Bogotá beta); fallback Medellín
_abm_candidates = []
if config_ok:
    _abm_candidates.append(
        Path(_abs.procesados).parent / "evidencia" / "resultados_simulacion_escenarios.csv"
    )
_abm_candidates.append(ROOT / "evidencia" / "resultados_simulacion_escenarios.csv")
ABM_PATH = next((p for p in _abm_candidates if p.exists()), _abm_candidates[-1])
SCENARIO_LABELS = {
    "focalizado":  "Focalizado — Optimización Algorítmica",
    "distribuido": "Distribuido — Igualitario",
    "adaptativo":  "Adaptativo — Híbrido",
}
# Descripciones conceptuales del glosario ABM (tesis doctoral, tono formal).
SCENARIO_DESCRIPTIONS = {
    "focalizado": (
        "Optimización algorítmica: concentra una fracción ρ del presupuesto "
        "anual en las cinco unidades de mayor prioridad (peso SHAP o, si no "
        "hay señal, IVC) y reparte el remanente de forma uniforme. Busca el "
        "mayor impacto en los territorios de mayor atribución de vulnerabilidad."
    ),
    "distribuido": (
        "Igualitario: asigna a cada unidad territorial la misma fracción del "
        "presupuesto (1/N), sin priorizar por IVC ni por SHAP. Es el "
        "contrapunto normativo de equidad nominal en la asignación."
    ),
    "adaptativo": (
        "Híbrido: repondera el presupuesto en proporción al IVC corriente "
        "penalizado por la eficiencia reciente de mejora "
        "(φ = 1/(1+5g)). No es una mezcla mecánica de las dos reglas "
        "anteriores, sino una heurística independiente que reasigna hacia "
        "territorios vulnerables con menor tasa de recuperación observada."
    ),
}
# Colores derivados de la paleta institucional (PALETTE) — misma familia
# cromática usada en el resto del dashboard, garantizando cohesión visual.
SCENARIO_COLORS = {
    "focalizado":  PALETTE["navy"],
    "distribuido": PALETTE["indigo"],
    "adaptativo":  PALETTE["indigo_mid"],
}
SCENARIO_BAND = {
    "focalizado":  "rgba(4,120,87,0.14)",
    "distribuido": "rgba(15,118,110,0.14)",
    "adaptativo":  "rgba(76,29,149,0.12)",
}
SCENARIO_ICONS = {
    "focalizado":  "gps_fixed",
    "distribuido": "balance",
    "adaptativo":  "hub",
}

# ── Constantes de features ─────────────────────────────────────────────────────
FEATURE_COLS = [
    "idx_desempleo", "idx_habitat", "idx_educacion",
    "idx_riesgo_clima", "idx_pobreza", "idx_tejido_social", "año_normalizado",
]


def _score_year_panel(df_src, year, mdl, feature_cols):
    """Filtra el panel al año pedido y calcula ivc_pred + terciles de prioridad."""
    if df_src is None or year is None or "Año" not in df_src.columns:
        return pd.DataFrame(), 0.0, 0.0
    panel = df_src[df_src["Año"] == year].copy()
    if panel.empty:
        return panel, 0.0, 0.0
    missing = [c for c in feature_cols if c not in panel.columns]
    if mdl is not None and not missing:
        try:
            panel["ivc_pred"] = mdl.predict(panel[list(feature_cols)].values)
        except (ValueError, TypeError, AttributeError):
            panel["ivc_pred"] = panel["ivc"] if "ivc" in panel.columns else np.nan
    elif "ivc" in panel.columns:
        panel["ivc_pred"] = panel["ivc"]
    else:
        panel["ivc_pred"] = np.nan
    q33 = float(panel["ivc_pred"].quantile(0.33))
    q66 = float(panel["ivc_pred"].quantile(0.66))
    if not np.isfinite(q33) or not np.isfinite(q66):
        panel["prioridad"] = "Media"
        return panel, 0.0, 0.0
    edges = [-np.inf, q33, q66, np.inf]
    uniq = []
    for edge in edges:
        if not uniq or edge != uniq[-1]:
            uniq.append(edge)
    nlab = len(uniq) - 1
    labels_by_n = {1: ["Media"], 2: ["Baja", "Alta"], 3: ["Baja", "Media", "Alta"]}
    panel["prioridad"] = pd.cut(
        panel["ivc_pred"],
        bins=uniq,
        labels=labels_by_n.get(nlab, ["Media"] * max(nlab, 1)),
        include_lowest=True,
    )
    return panel, q33, q66


FEATURE_LABELS = {
    "idx_desempleo":     "Desempleo",
    "idx_habitat":       "Hábitat / Vivienda",
    "idx_educacion":     "Educación",
    "idx_riesgo_clima":  "Riesgo climático (telemetría)",
    "idx_pobreza":       "Pobreza",
    "idx_tejido_social": "Tejido social",
    "año_normalizado":   "Tendencia temporal",
}
PRIORITY_COLORS = {
    "Alta":  PALETTE["danger"],
    "Media": PALETTE["warning"],
    "Baja":  PALETTE["success"],
}

# Nombre legible + descripción técnica de cada hiperparámetro de XGBoost,
# usados para traducir el dict crudo de `best_params` (cv_metrics.json) en
# una tabla comprensible en la sección "Hiperparámetros del modelo final".
HYPERPARAM_LABELS = {
    "n_estimators":     ("n_estimators",     "Número de árboles (rondas de boosting) del ensamble."),
    "max_depth":        ("max_depth",        "Profundidad máxima de cada árbol; controla la complejidad."),
    "learning_rate":    ("learning_rate",    "Tasa de aprendizaje (eta): peso de cada árbol nuevo sobre la predicción."),
    "subsample":        ("subsample",        "Fracción de observaciones muestreadas para entrenar cada árbol."),
    "colsample_bytree": ("colsample_bytree", "Fracción de variables muestreadas para entrenar cada árbol."),
    "min_child_weight": ("min_child_weight", "Peso mínimo requerido en un nodo hijo; regulariza contra sobreajuste."),
    "reg_alpha":        ("reg_alpha",        "Regularización L1 sobre los pesos de las hojas."),
    "reg_lambda":       ("reg_lambda",       "Regularización L2 sobre los pesos de las hojas."),
    "gamma":            ("gamma",            "Reducción mínima de pérdida exigida para crear una nueva partición."),
}

# ── Predicciones ──────────────────────────────────────────────────────────────
if (
    model is None
    and df_indices is not None
    and INDICES_PATH.exists()
):
    try:
        model = train_fallback_model(
            str(INDICES_PATH),
            str(MODEL_PATH),
            tuple(FEATURE_COLS),
        )
    except (ValueError, TypeError, OSError) as _fit_err:
        errors.append(f"modelo de prueba: {_fit_err}")

if df_indices is not None and "Año" in df_indices.columns:
    latest_year = int(df_indices["Año"].max())
else:
    latest_year = None
df_latest = pd.DataFrame()
q33 = q66 = 0.0


# ── Rellenar filtros en sidebar ────────────────────────────────────────────────
with _filter_placeholder.container():
    if df_indices is not None and "Año" in df_indices.columns:
        all_years     = sorted(df_indices["Año"].unique(), reverse=True)
        selected_year = st.selectbox(
            "Año de análisis",
            all_years,
            index=0,
            help=(
                "Año del panel que alimenta ranking, mapa y tabla de índices. "
                "El IVC está en [0, 1]: 0,50 es intermedio; un descenso a "
                "0,46 indica mejora relativa, no un umbral absoluto de política."
            ),
        )
    else:
        selected_year = latest_year

    if df_ranking is not None:
        n_top = st.slider(
            f"Top {UT_PLU} vulnerables",
            5,
            max(5, len(df_ranking)),
            10,
            help=(
                "Número de unidades que se muestran en el ranking del resumen. "
                "El orden es por IVC medio histórico (mayor = más vulnerable)."
            ),
        )
    else:
        n_top = 10

if df_indices is not None and selected_year is not None and "Año" in df_indices.columns:
    df_latest, q33, q66 = _score_year_panel(
        df_indices, selected_year, model, FEATURE_COLS
    )
    latest_year = int(selected_year)

# Métricas del modelo solo en modo Admin (no saturan la vista de decisión)
if _admin_metrics_placeholder is not None and metrics:
    with _admin_metrics_placeholder.container():
        st.html(f'<p class="section-title">{icon("target")} Métricas del modelo</p>')
        render_kpi_card(
            "R² CV promedio", f"{metrics['mean_R2']:.4f}",
            "Validación cruzada temporal", icon_name="target",
            help_text=(
                "R² medio en validación walk-forward (el modelo se entrena con "
                "años pasados y se evalúa en años futuros). Cercano a 1 indica "
                "mejor ajuste fuera de muestra; no debe confundirse con el R² "
                "in-sample del corpus completo."
            ),
        )
        st.html("<div style='height:.5rem'></div>")
        render_kpi_card(
            "RMSE CV promedio", f"{metrics['mean_RMSE']:.4f}",
            "Error medio de predicción", icon_name="show_chart",
            help_text=(
                "Raíz del error cuadrático medio del IVC en los pliegues de "
                "prueba. Está en la misma escala [0, 1] del índice: un RMSE "
                "de 0,01 equivale a un error típico de un punto porcentual "
                "del rango del IVC."
            ),
        )
        st.html("<div style='height:.5rem'></div>")
        render_kpi_card(
            "R² datos completos", f"{metrics['full_data_R2']:.4f}",
            "Ajuste in-sample", icon_name="verified",
            help_text=(
                "R² al reentrenar sobre todo el panel. Suele ser más alto que "
                "el R² de CV porque no es fuera de muestra; se reporta como "
                "referencia de ajuste, no como evidencia de generalización."
            ),
        )


# ══════════════════════════════════════════════════════════════════════════════
# ENCABEZADO PRINCIPAL (dinámico)
# ══════════════════════════════════════════════════════════════════════════════
render_brand_header()

with st.expander(
    "Objetivo de la plataforma y problemática de planificación urbana",
    expanded=True,
    icon=":material/info:",
):
    st.markdown(
        f"**Objetivo principal.** {PRODUCT_NAME} operacionaliza un marco integral "
        f"de gestión de datos para **priorizar la inversión pública a escala "
        f"intraurbana** en {CIUDAD}. Convierte encuestas socioeconómicas, "
        f"indicadores territoriales y, cuando existe, telemetría ambiental "
        f"en un índice de vulnerabilidad compuesto (IVC) por {UT.lower()}, "
        f"un ranking auditado y una proyección a diez años de tres "
        f"heurísticas de asignación presupuestaria."
    )
    st.markdown(
        f"**Problemática que atiende.** La planificación urbana opera bajo "
        f"restricción fiscal y con privación desigualmente distribuida entre "
        f"{UT_PLU.lower()}: pobreza, hábitat, educación, desempleo, tejido "
        f"social y exposición climática no coinciden en el mismo mapa. "
        f"Decidir *dónde* y *con qué regla* invertir cada año no admite una "
        f"solución puramente contable; requiere evidencia comparable, "
        f"validación temporal del pronóstico y escenarios contrafactuales "
        f"explícitos. El marco no sustituye el juicio del decisor: reduce "
        f"la opacidad de la priorización y permite contrastar políticas "
        f"antes de ejecutarlas."
    )
    st.markdown(
        "**Arquitectura analítica (tres capas).** (i) Índices dimensionales "
        "con corrección de polaridad, agregados en el IVC. (ii) Modelo "
        "XGBoost con validación cruzada temporal (*walk-forward*) y "
        "atribución SHAP fuera de muestra —diagnóstico, no ponderación "
        "normativa del índice—. (iii) Simulación basada en agentes "
        "(Python puro) que compara **Focalizado (optimización algorítmica)**, "
        "**Distribuido (igualitario)** y **Adaptativo (híbrido)** a "
        "horizonte 2024–2034. El flujo de lectura recomendado está en el "
        "panel lateral (*Guía metodológica*)."
    )
    st.caption(
        "Los valores del IVC y de sus componentes viven en [0, 1]. "
        "La interpretación de umbrales numéricos (p. ej. 0,50 frente a 0,46) "
        "está en las etiquetas de ayuda (icono «i») de cada métrica, no "
        "sobre los gráficos principales.",
        help=(
            "Escala [0, 1]: 0 denota menor privación relativa y 1, máxima "
            "vulnerabilidad en el panel. Un valor de 0,50 es intermedio; "
            "0,46 es una mejora relativa de cuatro centésimas, no un umbral "
            "de política nacional ni un corte POT. Alta / Media / Baja son "
            "terciles del último año, no categorías administrativas."
        ),
    )

# Barra de confianza institucional
_n_terr_chip = (
    df_ranking[UT_COL].nunique()
    if df_ranking is not None and UT_COL in df_ranking.columns
    else (len(df_latest) if not df_latest.empty else "—")
)
_r2_chip = (
    f"R² CV {metrics['mean_R2']:.3f}"
    if metrics and "mean_R2" in metrics
    else "modelo cargado"
)
st.html(
    f'<div class="trust-bar">'
    f'<span class="chip">{CIUDAD}</span>'
    f'<span class="chip-muted">Datos hasta {latest_year or "—"}</span>'
    f'<span class="chip-muted">{_n_terr_chip} {UT_PLU.lower()}</span>'
    f'<span class="chip-muted">{_r2_chip}</span>'
    f'<span class="chip-muted">No sustituye el juicio humano</span>'
    f'</div>'
)

# Banner si el usuario eligió un objetivo en el onboarding
_goal_key = st.session_state.get("policy_goal")
if _goal_key in POLICY_GOALS:
    _g = POLICY_GOALS[_goal_key]
    st.html(
        f'<div class="goal-banner">'
        f'<b>Tu objetivo:</b> {_g["label"]}'
        f'<p>Ve a la pestaña <b>«{_g["tab"]}»</b> — {_g["hint"]}</p>'
        f'</div>'
    )

# Ficha para comité (PDF)
try:
    _ficha_pdf = build_committee_ficha_pdf(
        ciudad=CIUDAD,
        ut=UT,
        ut_col=UT_COL,
        ut_plu=UT_PLU,
        year=latest_year,
        df_ranking=df_ranking,
        df_latest=df_latest,
        metrics=metrics,
        abm_path=ABM_PATH,
        scenario_labels=SCENARIO_LABELS,
        scenario_descriptions=SCENARIO_DESCRIPTIONS,
        abm_base=ABM_BASE_YEAR,
        abm_final=ABM_FINAL_YEAR,
        n_top=min(5, n_top) if isinstance(n_top, int) else 5,
    )
    _ficha_err = None
except Exception as _ficha_exc:
    _ficha_pdf = None
    _ficha_err = _ficha_exc

_dl1, _dl2 = st.columns([1, 3])
with _dl1:
    if _ficha_pdf is not None:
        st.download_button(
            label="Descargar ficha PDF",
            data=_ficha_pdf,
            file_name=f"ficha_priorizacion_{CIUDAD}_{latest_year or 'na'}.pdf",
            mime="application/pdf",
            icon=":material/picture_as_pdf:",
            width="stretch",
            help=(
                "PDF listo para consejo / POT: KPIs, top territorios, "
                "política recomendada a 10 años y nota metodológica."
            ),
            key="dl_ficha_comite",
        )
    else:
        st.warning(f"No se pudo generar el PDF: {_ficha_err}")
with _dl2:
    st.caption(
        "La ficha PDF incluye ranking, prioridades del último año, política "
        "recomendada a 10 años y nota metodológica — lista para expediente."
    )

if errors:
    with st.expander("Advertencias de carga de archivos", expanded=False):
        for e in errors:
            st.warning(e)


# ══════════════════════════════════════════════════════════════════════════════
# PESTAÑAS
# ══════════════════════════════════════════════════════════════════════════════
_TAB_LABELS = [
    ":material/bar_chart: Resumen ejecutivo",
    ":material/map: Mapa de priorización",
    ":material/query_stats: Detalle técnico",
    ":material/hub: Comparar políticas (10 años)",
]
_SERVICE_TO_TAB = {
    "priorizacion": _TAB_LABELS[0],
    "mapa": _TAB_LABELS[1],
    "validacion": _TAB_LABELS[2],
    "pipeline": _TAB_LABELS[2],
    "politicas": _TAB_LABELS[3],
    "simulacion": _TAB_LABELS[3],
}
if "main_tabs" not in st.session_state:
    st.session_state["main_tabs"] = _SERVICE_TO_TAB.get(
        st.session_state.get("service_selected"),
        _TAB_LABELS[0],
    )
tab1, tab2, tab3, tab4 = st.tabs(
    _TAB_LABELS,
    key="main_tabs",
    on_change="rerun",
)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — VISIÓN GENERAL
# ──────────────────────────────────────────────────────────────────────────────
if tab1.open:
    with tab1:
        st.html(
            f"<div class='intro-card'>"
            f"<h4>{icon('info', size='1.2rem', color='#047857')} "
            f"Lectura del resumen ejecutivo</h4>"
            f"<p>Esta pestaña condensa el diagnóstico territorial de <b>{CIUDAD}</b>: "
            f"indicadores clave, ranking histórico de {UT_PLU.lower()} y "
            f"distribución de prioridades del último año. El objetivo del marco "
            f"y el flujo metodológico están en el recuadro superior y en la "
            f"<b>Guía metodológica</b> del panel lateral. Los valores numéricos "
            f"del IVC se interpretan en las etiquetas de ayuda (icono «i»), no "
            f"como umbrales normativos sobre el gráfico.</p>"
            f"</div>"
        )

        st.html(f'<p class="section-title">{icon("bar_chart")} Indicadores clave — {CIUDAD}</p>')

        kpis = city_kpi_specs(df_ranking, df_latest, UT_COL, UT_PLU)
        if kpis:
            render_kpi_row(kpis)

            st.html("<br>")
            col_left, col_right = st.columns([3, 2])

            with col_left:
                render_section_title(
                    f"Ranking de Vulnerabilidad por {UT_COL}",
                    icon_name="leaderboard",
                    help_title="¿Cómo leer este gráfico?",
                    help_body=(
                        "Cada barra es el IVC medio histórico de una "
                        f"{UT.lower()}. El color indica el tercil de la "
                        "distribución: púrpura = tercio superior (más vulnerable), "
                        "ocre = tercio medio, lima = tercio inferior. "
                        "La escala y la lectura de valores puntuales (p. ej. "
                        "0,50 frente a 0,46) están en el icono de ayuda de los "
                        "indicadores clave, no como umbral dibujado en el eje."
                    ),
                )
                top_df = df_ranking.head(n_top).copy()
                ut_vals = top_df[UT_COL] if UT_COL in top_df.columns else top_df.iloc[:, 0]
                top_df["color"] = top_df["mean_ivc"].apply(
                    lambda v: PALETTE["danger"] if v >= q66
                    else (PALETTE["warning"] if v >= q33 else PALETTE["success"])
                )
                fig_rank = go.Figure(go.Bar(
                    y=ut_vals[::-1],
                    x=top_df["mean_ivc"][::-1],
                    orientation="h",
                    marker_color=top_df["color"][::-1].tolist(),
                    text=[f"{v:.3f}" for v in top_df["mean_ivc"][::-1]],
                    textposition="outside",
                    hovertemplate=f"<b>%{{y}}</b><br>Índice medio: %{{x:.4f}}<extra></extra>",
                ))
                fig_rank.update_layout(
                    xaxis_title="Índice medio histórico",
                    yaxis_title="",
                    height=420,
                    margin=dict(l=10, r=60, t=10, b=40),
                )
                apply_chart_theme(fig_rank)
                st.plotly_chart(fig_rank, width="stretch")

            with col_right:
                render_section_title(
                    "Distribución de Prioridades",
                    icon_name="pie_chart",
                    help_title="¿Qué significa cada prioridad?",
                    help_body=(
                        "El modelo divide las unidades territoriales en terciles "
                        "según el índice estimado para el último año disponible: "
                        "Alta (tercio superior, requiere intervención urgente), "
                        "Media (tercio intermedio) y Baja (tercio inferior, "
                        "condiciones relativamente estables)."
                    ),
                )
                if "prioridad" in df_latest.columns:
                    prio_counts = df_latest["prioridad"].value_counts().reset_index()
                    prio_counts.columns = ["Prioridad", UT_PLU]
                    fig_pie = px.pie(
                        prio_counts,
                        names="Prioridad",
                        values=UT_PLU,
                        color="Prioridad",
                        color_discrete_map=PRIORITY_COLORS,
                        hole=0.45,
                        title=f"Prioridad por {UT_COL}",
                    )
                    fig_pie.update_layout(
                        height=220, margin=dict(l=0, r=0, t=30, b=0),
                        showlegend=True, legend=dict(orientation="h", y=-0.15),
                    )
                    apply_chart_theme(fig_pie)
                    st.plotly_chart(fig_pie, width="stretch")

                render_section_title(
                    "Evolución temporal del índice",
                    icon_name="show_chart",
                    help_title="Interpretación de la tendencia",
                    help_body=(
                        "Muestra la trayectoria histórica anual del índice de "
                        "vulnerabilidad para las 5 unidades territoriales más "
                        "vulnerables. Una pendiente descendente indica mejora "
                        "sostenida; una pendiente ascendente indica deterioro "
                        "de las condiciones."
                    ),
                )
                if df_indices is not None and UT_COL in df_indices.columns:
                    top5 = (df_ranking.head(5)[UT_COL].tolist()
                            if UT_COL in df_ranking.columns else [])
                    df_evo = df_indices[df_indices[UT_COL].isin(top5)]
                    fig_evo = px.line(
                        df_evo, x="Año", y="ivc", color=UT_COL, markers=True,
                        color_discrete_sequence=INSTITUTIONAL_COLORWAY,
                        labels={"ivc": "Índice", "Año": "Año", UT_COL: UT_COL},
                    )
                    fig_evo.update_layout(
                        height=200, margin=dict(l=0, r=0, t=0, b=0),
                        legend=dict(font=dict(size=9), y=1.0),
                    )
                    apply_chart_theme(fig_evo)
                    st.plotly_chart(fig_evo, width="stretch")

            render_section_title(
                f"Detalle por Índice Social — {UT_COL} (último año disponible)",
                icon_name="table_chart",
                help_title="Escala de los índices sociales",
                help_body=(
                    "Todos los índices dimensionales y el IVC estimado están "
                    "normalizados en [0, 1]; un valor más alto indica mayor "
                    "privación en esa dimensión. Un 0,50 es intermedio en el "
                    "panel; un 0,46 es una mejora relativa de cuatro centésimas, "
                    "no un corte normativo de política. El índice estimado "
                    "combina las dimensiones; no debe leerse como probabilidad "
                    "ni como porcentaje de población pobre."
                ),
            )
            if not df_latest.empty and UT_COL in df_latest.columns:
                cols_show = (
                    [UT_COL]
                    + [c for c in FEATURE_COLS[:-1] if c in df_latest.columns]
                    + ["ivc_pred", "prioridad"]
                )
                df_show = (
                    df_latest[cols_show]
                    .rename(columns={
                        **FEATURE_LABELS,
                        "ivc_pred": "Índice estimado",
                        "prioridad": "Prioridad",
                    })
                    .sort_values("Índice estimado", ascending=False)
                    .reset_index(drop=True)
                )

                def color_priority(val):
                    return {
                        "Alta":  "background-color:#ede9fe",
                        "Media": "background-color:#fef9c3",
                        "Baja":  "background-color:#d9f99d",
                    }.get(val, "")

                st.dataframe(
                    df_show.style
                           .map(color_priority, subset=["Prioridad"])
                           .format({c: "{:.3f}" for c in df_show.select_dtypes("float").columns}),
                    width="stretch",
                    height=300,
                )
        else:
            st.warning(
                f"No se cargaron datos de vulnerabilidad para {CIUDAD}. "
                f"Se buscaron `indices_comunas.csv` en `{DATA_PROC}` "
                f"y `vulnerability_ranking.csv` en `{MODELS_DIR}`."
            )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — MAPA PREDICTIVO
# ──────────────────────────────────────────────────────────────────────────────
if tab2.open:
    with tab2:
        render_section_title(
            f"Mapa de priorización por {UT_COL} · {CIUDAD}",
            icon_name="map",
            help_title="Cómo interpretar el mapa",
            help_body=(
                "El color de cada polígono representa el índice de "
                "vulnerabilidad estimado para el último año disponible "
                "(amarillo = baja; púrpura profundo = máxima). Pasa el cursor sobre "
                "cualquier zona para ver su nombre, el valor exacto y la "
                "prioridad de inversión; haz clic para fijar el detalle. "
                "Zonas en gris no tienen predicción disponible."
            ),
        )
        st.caption(
            f"Estimaciones para {latest_year} · {UT_COL}: unidad territorial activa",
            help=(
                "El color del polígono representa el IVC estimado del último "
                "año (amarillo = menor; púrpura profundo = mayor). Mayor valor "
                "= peor. Un 0,50 es intermedio en el panel; 0,46 es una mejora "
                "relativa, no un umbral POT. Zonas en gris no tienen predicción."
            ),
        )
        map_kpis = city_kpi_specs(df_ranking, df_latest, UT_COL, UT_PLU)
        if map_kpis:
            render_kpi_row(map_kpis)
            st.html("<div style='height:.6rem'></div>")

        gdf, geo_source = load_geodata(_geo_paths, _geo_bust)
        if gdf is not None:
            st.caption(
                f"Cartografía cargada: `{geo_source}` ({len(gdf)} polígonos). "
                "Si aún ves huecos, pulsa Ctrl+F5 para forzar recarga."
            )

        if gdf is not None and not df_latest.empty:
            try:
                import folium
                import unicodedata

                # Alias de nombre corto vs. nombre administrativo completo. El
                # GeoJSON oficial (ArcGIS) suele listar solo el nombre propio del
                # corregimiento (ej. "Palmitas"), mientras que los CSV de índices
                # (derivados de ECV/IPM/IML) usan el nombre administrativo
                # completo (ej. "Corregimiento de San Sebastián de Palmitas").
                # Se resuelve después de quitar el prefijo "corregimiento de".
                _NAME_ALIASES = {
                    "san sebastian de palmitas": "palmitas",
                    # Bogotá IDECA: LOCNOMBRE "CANDELARIA" ↔ CSV "La Candelaria"
                    "candelaria": "la candelaria",
                    # Barranquilla ArcGIS (MAYÚSCULAS) ↔ CSV Title Case
                    "norte-centro historico": "norte-centro historico",
                    "norte centro historico": "norte-centro historico",
                }

                def _norm_name(s: str) -> str:
                    """Normaliza nombre: sin tildes, minúsculas, sin prefijos
                    administrativos genéricos ("corregimiento de ...") y sin
                    espacios extremos — permite unir el GeoJSON (nombre corto)
                    con los CSV de índices (nombre administrativo completo)."""
                    key = (
                        unicodedata.normalize("NFD", str(s))
                        .encode("ascii", "ignore")
                        .decode()
                        .lower()
                        .strip()
                    )
                    for prefix in (
                        "corregimiento de ",
                        "corregimiento ",
                        "localidad de ",
                        "localidad ",
                        "comuna de ",
                        "comuna ",
                        "la ",
                    ):
                        if key.startswith(prefix):
                            key = key[len(prefix):]
                            break
                    return _NAME_ALIASES.get(key, key)

                # En demos con comunas+barrios, el coropleto usa solo la UT
                # administrativa (comuna/localidad) para no diluir el join.
                if "tipo" in gdf.columns:
                    _tipo = gdf["tipo"].astype(str).str.lower()
                    _keep = _tipo.isin(
                        {
                            str(UT).lower(),
                            "comuna",
                            "localidad",
                            "sector",
                            "nan",
                            "none",
                            "",
                        }
                    )
                    if _keep.any() and (~_keep).any():
                        gdf = gdf.loc[_keep].copy()

                # ── 1. Detectar columna de nombre en el GeoJSON ───────────────────
                name_col = next(
                    (c for c in [
                        "Unidad_Territorial", "Localidad", "nombre", "NOMBRE",
                        "LOCNOMBRE", "NOMBRE_COM", "NOMB_COM", "comnombre",
                        "NOMBRES_CO", "COMNOMBRE",
                        UT_COL, UT_COL.upper(), "Comuna", "LOCALIDAD",
                    ]
                     if c in gdf.columns),
                    None,
                )

                if name_col is None:
                    st.info(
                        f"GeoJSON sin columna de nombre reconocible. "
                        f"Columnas disponibles: {gdf.columns.tolist()}"
                    )
                    gdf = None
                else:
                    # ── 2. Merge robusto por clave normalizada ────────────────────
                    # Se toman SOLO las columnas necesarias del GeoJSON para evitar
                    # conflictos de nombres al unir con df_latest.
                    # CRÍTICO: si name_col == UT_COL (ej. ambos "Localidad" en
                    # Bogotá), pandas crea Localidad_x/_y y rompe el Choropleth.
                    geo_name_col = "_poly_name"
                    gdf_work = (
                        gdf[[name_col, "geometry"]]
                        .copy()
                        .rename(columns={name_col: geo_name_col})
                    )
                    gdf_work["_key"] = gdf_work[geo_name_col].astype(str).map(_norm_name)

                    df_merge = df_latest[[UT_COL, "ivc_pred", "prioridad"]].copy()
                    df_merge["_key"] = df_merge[UT_COL].astype(str).map(_norm_name)
                    # Cuando existen variantes del mismo nombre (ej. "Belén"/"Belen"),
                    # se conserva el promedio de IVC y la primera prioridad disponible.
                    df_merge = (
                        df_merge.groupby("_key", as_index=False)
                        .agg({"ivc_pred": "mean", "prioridad": "first", UT_COL: "first"})
                    )

                    gdf_m = (
                        gdf_work
                        .merge(df_merge, on="_key", how="left")
                        .drop(columns=["_key"])
                        .set_geometry("geometry")
                        .set_crs(epsg=4326, allow_override=True)
                        .reset_index(drop=True)
                    )

                    # Columna canónica de unión para Folium (= name_col original)
                    if UT_COL in gdf_m.columns:
                        gdf_m[name_col] = gdf_m[UT_COL].fillna(gdf_m[geo_name_col])
                    else:
                        gdf_m[name_col] = gdf_m[geo_name_col]

                    # ── 3. Limpiar tipos para serialización JSON segura ───────────
                    gdf_m["prioridad"] = (
                        gdf_m["prioridad"]
                        .astype(str)
                        .replace({"nan": "N/D", "<NA>": "N/D", "None": "N/D"})
                    )
                    gdf_m["ivc_pred"] = pd.to_numeric(gdf_m["ivc_pred"], errors="coerce")
                    # Campo de IVC formateado para tooltip (no altera la columna numérica)
                    gdf_m["ivc_fmt"] = gdf_m["ivc_pred"].apply(
                        lambda v: f"{v:.4f}" if pd.notna(v) else "N/D"
                    )
                    # Nombre a mostrar en tooltip/popup: se prioriza el nombre
                    # administrativo completo del CSV de índices (ej. "Corregimiento
                    # de San Sebastián de Palmitas") sobre el nombre corto del
                    # GeoJSON (ej. "Palmitas"), cayendo al nombre del GeoJSON solo
                    # si el polígono no tuvo coincidencia en los datos.
                    gdf_m["_display_name"] = gdf_m[name_col].fillna(gdf_m[geo_name_col])

                    matched = int(gdf_m["ivc_pred"].notna().sum())
                    st.caption(
                        f"Polígonos activos con IVC: **{matched}/{len(gdf_m)}** · "
                        f"Columna de unión: `{name_col}` · Año: {latest_year}"
                    )

                    # ── 4. Serializar GeoJSON como string ─────────────────────────
                    # to_json() es más estable que __geo_interface__ para Choropleth.
                    geojson_str = gdf_m.to_json()

                    # ── 5. DataFrame de color: MISMA fuente que geo_data ──────────
                    # PUNTO CRÍTICO: key_on = "feature.properties.<name_col>"
                    # → columns[0] debe ser la columna <name_col> del mismo GDF.
                    # Usar df_latest separado rompe el join cuando los nombres difieren.
                    chor_data = (
                        gdf_m[[name_col, "ivc_pred"]]
                        .dropna(subset=["ivc_pred"])
                        .reset_index(drop=True)
                    )

                    # ── 6. Mapa base ──────────────────────────────────────────────
                    # Centro geográfico oficial de Medellín (WGS84).
                    # Se usa el centroide de config si es válido, o las coords
                    # oficiales del Distrito como respaldo.
                    _cfg_lat = cfg.ciudad.centro.lat if config_ok else 6.2518
                    _cfg_lon = cfg.ciudad.centro.lon if config_ok else -75.5636
                    center = [_cfg_lat, _cfg_lon]

                    m = folium.Map(
                        location=center,
                        zoom_start=13,
                        tiles=BASEMAP_TILES,
                        attr=BASEMAP_ATTR,
                        control_scale=True,
                    )

                    # ── 7. Coropleta ──────────────────────────────────────────────
                    chor = folium.Choropleth(
                        geo_data=geojson_str,
                        data=chor_data,
                        columns=[name_col, "ivc_pred"],
                        key_on=f"feature.properties.{name_col}",
                        fill_color=FOLIUM_IVC_FILL,
                        fill_opacity=0.7,
                        line_color=PALETTE["charcoal"],
                        line_opacity=0.4,
                        legend_name=(
                            f"Índice de vulnerabilidad por {UT_COL}  "
                            f"(0 = baja · 1 = máxima)"
                        ),
                        nan_fill_color="#c8c8c8",
                        nan_fill_opacity=0.30,
                        highlight=True,
                    ).add_to(m)

                    # ── 8. Tooltip al pasar el cursor ─────────────────────────────
                    chor.geojson.add_child(
                        folium.features.GeoJsonTooltip(
                            fields=["_display_name", "ivc_fmt", "prioridad"],
                            aliases=[
                                f"<b>{UT_COL}</b>",
                                "<b>Índice de vulnerabilidad</b>",
                                "<b>Prioridad de inversión</b>",
                            ],
                            localize=True,
                            sticky=True,
                            style=(
                                "background-color: white; color: #1a1a2e; "
                                "font-family: 'Segoe UI', Arial, sans-serif; "
                                "font-size: 13px; padding: 8px 14px; "
                                "border-radius: 8px; border: 1px solid #c5c5c5; "
                                "box-shadow: 3px 3px 8px rgba(0,0,0,0.18);"
                            ),
                        )
                    )

                    # ── 9. Popup al hacer clic (sobre la misma capa, sin duplicar GeoJSON)
                    chor.geojson.add_child(
                        folium.features.GeoJsonPopup(
                            fields=["_display_name", "ivc_fmt", "prioridad"],
                            aliases=[
                                f"{UT_COL}:",
                                "Índice de vulnerabilidad:",
                                "Prioridad de inversión:",
                            ],
                            localize=True,
                            max_width=300,
                            style=(
                                "font-family: 'Segoe UI', Arial, sans-serif; "
                                "font-size: 13px; border-radius: 8px; padding: 8px;"
                            ),
                        )
                    )

                    st.components.v1.html(m._repr_html_(), height=580, scrolling=False)

            except Exception as e:
                st.error(f"Error al renderizar el mapa Folium: {e}")
                import traceback
                with st.expander("Detalle del error (para depuración)"):
                    st.code(traceback.format_exc(), language="python")
                gdf = None

        if gdf is None and not df_latest.empty and "ivc_pred" in df_latest.columns:
            st.info(
                f"No se encontro archivo geoespacial de {UT_PLU.lower()} en `{GEO_DIR}`. "
                f"Coloca el archivo como `{UT}s.geojson` para activar el mapa Folium. "
                f"Se muestra un mapa de burbujas con centroides del config."
            )

            # Centroides desde config.yaml (dinámicos)
            centroids: dict[str, tuple[float, float]] = {
                t.nombre: (t.lat, t.lon) for t in cfg.territorios
            } if config_ok else {}

            df_map = df_latest.copy()
            df_map["lat"] = df_map[UT_COL].map(lambda c: centroids.get(c, (np.nan, np.nan))[0])
            df_map["lon"] = df_map[UT_COL].map(lambda c: centroids.get(c, (np.nan, np.nan))[1])
            df_map = df_map.dropna(subset=["lat", "lon"])

            if df_map.empty:
                st.warning(
                    f"No se encontraron coordenadas para las {UT_PLU.lower()} del dataset. "
                    f"Verifica que los nombres en `indices_comunas.csv` coinciden con "
                    f"los nombres en `config.yaml > territorios`."
                )
            else:
                center = [cfg.ciudad.centro.lat, cfg.ciudad.centro.lon] if config_ok else [6.2442, -75.5812]
                fig_map = px.scatter_mapbox(
                    df_map,
                    lat="lat", lon="lon",
                    color="ivc_pred",
                    size="ivc_pred",
                    hover_name=UT_COL,
                    hover_data={"ivc_pred": ":.4f", "prioridad": True, "lat": False, "lon": False},
                    color_continuous_scale=IVC_COLORSCALE,
                    size_max=35,
                    zoom=11,
                    center={"lat": center[0], "lon": center[1]},
                    mapbox_style=PLOTLY_BASEMAP,
                    labels={"ivc_pred": "Índice estimado", UT_COL: UT_COL},
                    title=f"Priorizacion por {UT_COL} — {CIUDAD} (centroides del config)",
                )
                fig_map.update_layout(height=560, margin=dict(l=0, r=0, t=40, b=0))
                apply_chart_theme(fig_map)
                st.plotly_chart(fig_map, width="stretch")

                st.dataframe(
                    df_map[[UT_COL, "ivc_pred", "prioridad"]]
                    .sort_values("ivc_pred", ascending=False)
                    .rename(columns={"ivc_pred": "Índice estimado", "prioridad": "Prioridad"})
                    .reset_index(drop=True),
                    width="stretch",
                    height=260,
                )

        elif not run_pipeline and df_latest.empty:
            st.warning("No hay datos de prediccion disponibles. Ejecuta el Pipeline Automatico.")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — INTERPRETABILIDAD
# ──────────────────────────────────────────────────────────────────────────────
if tab3.open:
    with tab3:
        render_section_title(
            "Detalle técnico — qué factores explican la priorización",
            icon_name="query_stats",
            help_title="¿Qué es la interpretabilidad de un modelo?",
            help_body=(
                "El modelo predictivo estima el índice de vulnerabilidad "
                "combinando varias variables, pero no explica por sí mismo "
                "cuáles pesan más. Las dos métricas de esta sección "
                "(Gain Importance y SHAP) responden esa pregunta desde "
                "ángulos distintos, para auditar y justificar las "
                "estimaciones ante decisores de política pública."
            ),
        )
        st.caption(
            f"Variables sociales y climáticas que más influyen en el índice "
            f"estimado para las {UT_PLU.lower()} de {CIUDAD}.",
            help=(
                "Gain Importance cuantifica la utilidad de cada variable durante "
                "el entrenamiento de los árboles. SHAP cuantifica su contribución "
                "aditiva a cada predicción. Ninguna de las dos redefine los pesos "
                "del IVC: son diagnósticos de interpretabilidad. El ejemplo "
                "«Pobreza Gain: 0.3819» se lee como 38,19 % de la capacidad "
                "predictiva del modelo, no como un 38 % de pobreza territorial."
            ),
        )

        col_fi, col_shap = st.columns(2)

        with col_fi:
            col_fi_t, col_fi_h = st.columns([0.9, 0.1], vertical_alignment="center")
            with col_fi_t:
                st.markdown("**XGBoost Gain Importance**")
            with col_fi_h:
                with st.popover("", icon=":material/info:", help="Más información"):
                    st.markdown("**¿Qué es el Gain Importance?**")
                    st.caption(
                        "Mide cuánto reduce cada variable el error del modelo "
                        "cada vez que se usa para dividir un árbol de decisión "
                        "dentro del ensamble XGBoost. Se expresa como fracción "
                        "del total (todas las variables suman 1.0)."
                    )
                    st.markdown("**Ejemplo de interpretación:**")
                    st.caption(
                        "Un valor de *Pobreza Gain: 0.3819* significa que el "
                        "índice de pobreza multidimensional explica el 38.19 % "
                        "de toda la capacidad predictiva del modelo — es, con "
                        "amplio margen, el factor dominante en la predicción "
                        "del IVC."
                    )
            st.caption(
                "Contribución relativa de cada variable al criterio de división de los árboles.",
                help=(
                    "Gain Importance: fracción de la reducción de error atribuible "
                    "a cada variable en el ensamble XGBoost (las fracciones suman 1). "
                    "Ejemplo: «Pobreza Gain: 0.3819» significa que el índice de "
                    "pobreza explica el 38,19 % de la capacidad predictiva del "
                    "modelo —el factor dominante si es el máximo—. No es un peso "
                    "del IVC, ni un coeficiente de regresión, ni un porcentaje de "
                    "población pobre."
                ),
            )
            if model is not None:
                try:
                    fi = model.feature_importances_
                    fi_df = pd.DataFrame({"feature": FEATURE_COLS, "importance": fi})
                    fi_df = fi_df.sort_values("importance", ascending=True)
                    fi_df["feature_label"] = fi_df["feature"].map(FEATURE_LABELS)
                    max_fi = fi_df["importance"].max()
                    fi_df["color"] = fi_df["importance"].apply(
                        lambda v: PALETTE["indigo_mid"] if v == max_fi else PALETTE["navy"]
                    )
                    fig_fi = go.Figure(go.Bar(
                        y=fi_df["feature_label"],
                        x=fi_df["importance"],
                        orientation="h",
                        marker_color=fi_df["color"].tolist(),
                        text=[f"{v:.4f}" for v in fi_df["importance"]],
                        textposition="outside",
                        hovertemplate=f"<b>%{{y}}</b><br>Importancia: %{{x:.5f}}<extra></extra>",
                    ))
                    fig_fi.add_vline(
                        x=fi_df["importance"].mean(),
                        line_dash="dash", line_color=PALETTE["slate"],
                        annotation_text="Media", annotation_position="top right",
                    )
                    fig_fi.update_layout(
                        xaxis_title="Importancia (Gain)",
                        yaxis_title="",
                        title=f"Gain Importance — {CIUDAD}",
                        height=380,
                        margin=dict(l=10, r=70, t=40, b=40),
                    )
                    apply_chart_theme(fig_fi)
                    st.plotly_chart(fig_fi, width="stretch")
                    best_feat = fi_df.iloc[-1]["feature_label"]
                    st.success(f"**Variable más influyente:** {best_feat} (Gain: {max_fi:.4f})")
                except Exception as e:
                    st.error(f"No se pudo calcular Feature Importance: {e}")
            else:
                st.warning("Modelo no disponible. Ejecuta el Pipeline Automatico.")

        with col_shap:
            col_sh_t, col_sh_h = st.columns([0.9, 0.1], vertical_alignment="center")
            with col_sh_t:
                st.markdown("**SHAP — Importancia estable (CV temporal OOS)**")
            with col_sh_h:
                with st.popover("", icon=":material/info:", help="Más información"):
                    st.markdown("**¿Qué significa SHAP?**")
                    st.caption(
                        "SHAP (*SHapley Additive exPlanations*) es un método "
                        "basado en teoría de juegos que reparte de forma "
                        "matemáticamente justa la contribución de cada variable "
                        "a cada predicción individual."
                    )
                    st.markdown("**Fuente oficial del ranking (este panel):**")
                    st.caption(
                        "Se usa `shap_importance_cv_stability.csv`: media y "
                        "desviación de |SHAP| calculadas en el conjunto de "
                        "validación de cada pliegue de CV temporal "
                        "(`ranking_by_mean`). Eso valida la jerarquía "
                        "*out-of-sample*. El archivo `shap_values.csv` es "
                        "atribución *in-sample* del modelo al 100 % y solo "
                        "se reserva para detalle exploratorio por "
                        "territorio-año — nunca como ranking headline."
                    )
                    st.markdown("**Diferencia con Gain Importance:**")
                    st.caption(
                        "Gain mide la utilidad de la variable *durante el "
                        "entrenamiento* de los árboles; SHAP-CV mide su impacto "
                        "real sobre predicciones de validación temporal."
                    )
            st.caption(
                "Jerarquía de determinantes validada por CV temporal (fuera de muestra).",
                help=(
                    "SHAP (SHapley Additive exPlanations) reparte de forma aditiva "
                    "y axiomáticamente justa la contribución de cada variable a "
                    "cada predicción del IVC. El ranking de este panel usa la "
                    "media |SHAP| en los conjuntos de validación de cada pliegue "
                    "temporal (shap_importance_cv_stability.csv → ranking_by_mean), "
                    "no la atribución in-sample. Un valor alto indica mayor impacto "
                    "medio sobre el IVC estimado; no es un peso normativo de política "
                    "ni un porcentaje de pobreza. Fuente alternativa shap_values.csv: "
                    "solo detalle exploratorio por territorio-año."
                ),
            )
            if df_shap_stability is not None and "ranking_by_mean" in df_shap_stability.columns:
                try:
                    # Ranking headline: estabilidad OOS (NO shap_values.csv in-sample).
                    # Motivo: la atribución Shapley del modelo reentrenado al 100 %
                    # no verifica estabilidad entre ventanas temporales; ranking_by_mean
                    # promedia |SHAP| sobre los sets de validación de cada fold.
                    stab = df_shap_stability.copy()
                    # ranking_by_mean: 1 = más importante → ordenar ascendente para
                    # barras horizontales (arriba = menos importante en plotly barh
                    # con sort invertido visual: usamos ascending=False en y vía sort)
                    stab = stab.sort_values("ranking_by_mean", ascending=False)
                    stab["feature_label"] = stab["feature"].map(
                        lambda f: FEATURE_LABELS.get(f, f)
                    )
                    mean_col = "shap_abs_mean_across_folds"
                    std_col = "shap_abs_std_across_folds"
                    if mean_col not in stab.columns:
                        raise KeyError(mean_col)
                    max_s = float(stab[mean_col].max())
                    top_row = stab.loc[stab["ranking_by_mean"].idxmin()]
                    stab["color"] = stab["ranking_by_mean"].apply(
                        lambda r: PALETTE["indigo_mid"] if int(r) == 1 else PALETTE["navy"]
                    )
                    err = stab[std_col] if std_col in stab.columns else None
                    fig_shap = go.Figure(go.Bar(
                        y=stab["feature_label"],
                        x=stab[mean_col],
                        orientation="h",
                        marker_color=stab["color"].tolist(),
                        error_x=dict(
                            type="data",
                            array=err.tolist() if err is not None else None,
                            visible=err is not None,
                        ) if err is not None else None,
                        text=[
                            f"#{int(r)}  {v:.4f}"
                            for r, v in zip(stab["ranking_by_mean"], stab[mean_col])
                        ],
                        textposition="outside",
                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            "media |SHAP| CV: %{x:.5f}"
                            + (" ± %{error_x.array:.5f}" if err is not None else "")
                            + "<extra></extra>"
                        ),
                    ))
                    fig_shap.update_layout(
                        xaxis_title="Media |SHAP| entre pliegues de CV (± 1 DE)",
                        yaxis_title="",
                        title=f"Importancia SHAP (estable en CV) — {CIUDAD}",
                        height=380,
                        margin=dict(l=10, r=90, t=40, b=40),
                    )
                    apply_chart_theme(fig_shap)
                    st.plotly_chart(fig_shap, width="stretch")
                    best_shap = FEATURE_LABELS.get(
                        top_row["feature"], top_row["feature"]
                    )
                    st.success(
                        f"**Determinante dominante (ranking_by_mean=1, CV OOS):** "
                        f"{best_shap} "
                        f"(media |SHAP|={float(top_row[mean_col]):.4f}"
                        + (
                            f", DE entre folds={float(top_row[std_col]):.4f}"
                            if std_col in top_row.index else ""
                        )
                        + ")"
                    )
                except Exception as e:
                    st.error(f"Error al graficar SHAP-CV: {e}")
            else:
                st.info(
                    "shap_importance_cv_stability.csv no encontrado. "
                    "Ejecuta ml/03_xgboost_model.py para generarlo. "
                    "No se usa shap_values.csv como ranking headline "
                    "(ese archivo es solo exploratorio / territorio-año)."
                )

            # Detalle exploratorio in-sample (opcional): no define el ranking
            if df_shap is not None:
                with st.expander(
                    "Detalle exploratorio in-sample (shap_values.csv) — no oficial",
                    expanded=False,
                ):
                    st.caption(
                        "Atribución del modelo reentrenado al 100 % de los datos. "
                        "Útil para inspeccionar contribuciones por territorio-año; "
                        "**no** sustituye el ranking CV-estable de arriba.",
                        help=(
                            "SHAP in-sample no verifica estabilidad entre ventanas "
                            "temporales. El ranking oficial es ranking_by_mean en "
                            "shap_importance_cv_stability.csv (media |SHAP| OOS)."
                        ),
                    )
                    try:
                        shap_feats = [c for c in FEATURE_COLS if c in df_shap.columns]
                        preview = (
                            df_shap[[c for c in df_shap.columns if c in
                                     ([UT_COL, "Comuna", "Año"] + shap_feats)]]
                            .head(20)
                        )
                        st.dataframe(preview, width="stretch", hide_index=True)
                    except Exception as e:
                        st.caption(f"No se pudo mostrar el detalle: {e}")

        st.markdown("---")
        render_section_title(
            "Métricas de Validación Cruzada Temporal (Walk-Forward)",
            icon_name="rule",
            help_title="R² y RMSE — ¿qué significan?",
            help_body=(
                "R² indica qué fracción de la variabilidad del IVC explica "
                "el modelo en validación walk-forward (1.0 = ajuste perfecto "
                "fuera de muestra). RMSE es el error típico en la misma "
                "escala [0, 1] del índice: un RMSE de 0,01 equivale a un "
                "error de un punto porcentual del rango del IVC, no a un "
                "«1 % de pobreza». El R² de datos completos es in-sample y "
                "no debe usarse como evidencia de generalización."
            ),
        )
        if metrics:
            cv_df = pd.DataFrame(metrics["cv_folds"])[
                ["fold", "train_years", "test_years", "RMSE", "MAE", "R2"]
            ]
            cv_df.columns = ["Fold", "Años Entrenamiento", "Años Prueba", "RMSE", "MAE", "R²"]

            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                render_kpi_card("R² CV Promedio", f"{metrics['mean_R2']:.4f}",
                                 "5 pliegues walk-forward", icon_name="target",
                                 help_text=(
                                     "R² medio en validación walk-forward: el modelo "
                                     "se entrena con años pasados y se evalúa en años "
                                     "futuros. Cercano a 1 indica mejor ajuste fuera "
                                     "de muestra."
                                 ))
            with col_m2:
                render_kpi_card("RMSE CV Promedio", f"{metrics['mean_RMSE']:.5f}",
                                 "Error medio de predicción", icon_name="show_chart",
                                 help_text=(
                                     "Raíz del error cuadrático medio del IVC en los "
                                     "pliegues de prueba. Escala [0, 1]: un RMSE de "
                                     "0,01 es un error típico de un punto porcentual "
                                     "del rango del índice, no un 1 % de pobreza."
                                 ))
            with col_m3:
                render_kpi_card("R² Datos Completos", f"{metrics['full_data_R2']:.4f}",
                                 "Ajuste in-sample", icon_name="verified",
                                 help_text=(
                                     "R² al reentrenar sobre todo el panel. Suele ser "
                                     "más alto que el R² de CV; se reporta como "
                                     "referencia de ajuste, no como evidencia de "
                                     "generalización."
                                 ))

            fig_cv = go.Figure()
            fig_cv.add_trace(go.Scatter(
                x=cv_df["Fold"], y=cv_df["R²"],
                mode="lines+markers+text",
                text=[f"{v:.3f}" for v in cv_df["R²"]],
                textposition="top center",
                name="R² por Fold",
                line=dict(color=PALETTE["indigo"], width=2),
                marker=dict(size=9),
            ))
            fig_cv.add_hline(
                y=metrics["mean_R2"], line_dash="dot", line_color=PALETTE["navy"],
                annotation_text=f"R² promedio = {metrics['mean_R2']:.4f}",
                annotation_position="bottom right",
            )
            fig_cv.update_layout(
                xaxis_title="Fold Temporal",
                yaxis_title="R²",
                title=f"Walk-Forward CV — {CIUDAD}",
                height=280,
                margin=dict(l=0, r=20, t=40, b=40),
                yaxis=dict(range=[0.85, 1.0]),
                xaxis=dict(tickmode="linear", dtick=1),
            )
            apply_chart_theme(fig_cv)
            st.plotly_chart(fig_cv, width="stretch")

            st.dataframe(
                cv_df.style.format({"RMSE": "{:.5f}", "MAE": "{:.5f}", "R²": "{:.4f}"}),
                width="stretch",
                hide_index=True,
            )
            with st.expander("Hiperparametros del modelo final"):
                best_params = metrics.get("best_params", {})
                if best_params:
                    rows = [
                        {
                            "Hiperparámetro": HYPERPARAM_LABELS.get(k, (k, ""))[0],
                            "Valor Óptimo": v,
                            "Descripción": HYPERPARAM_LABELS.get(k, (k, ""))[1],
                        }
                        for k, v in best_params.items()
                    ]
                    st.dataframe(
                        pd.DataFrame(rows),
                        width="stretch",
                        hide_index=True,
                    )
                    st.caption(
                        "Configuración final del `XGBRegressor`, seleccionada mediante "
                        "`GridSearchCV` sobre el pliegue con menor RMSE de la validación "
                        "cruzada temporal (walk-forward)."
                    )
                else:
                    st.info("No hay hiperparámetros registrados para este modelo.")
        else:
            st.warning(
                "No se cargaron metricas. Ejecuta el Pipeline Automatico para entrenar el modelo."
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SIMULACIÓN BASADA EN AGENTES (ABM)  2024 → 2034
# ══════════════════════════════════════════════════════════════════════════════
if tab4.open:
    with tab4:
        render_section_title(
            "Comparar políticas de inversión — proyección a 10 años",
            icon_name="hub",
            help_title="¿Cómo se proyectan las políticas?",
            help_body=(
                "Cada unidad territorial se simula como un agente que recibe "
                "inversión y actualiza su bienestar y vulnerabilidad año a "
                "año, según reglas calibradas con datos históricos. Permite "
                "comparar el efecto proyectado de distintas políticas de "
                "asignación presupuestaria antes de implementarlas."
            ),
        )

        st.info(
            "Las trayectorias muestran la proyección a **10 años** "
            "partiendo de la línea base observada en **2024**. "
            "Cada zona arranca con su índice de vulnerabilidad histórico "
            "(datos 2007–2024) y evoluciona hasta **2034** "
            "bajo tres estrategias de distribución presupuestaria con "
            "restricción fiscal constante. "
            f"La banda sombreada representa la dispersión territorial (±1 DE) entre {UT_PLU.lower()}.",
            icon=":material/calendar_month:",
        )

        # ── Glosario conceptual de los 3 escenarios de inversión ────────────────
        st.html(
            f'<p class="section-title" style="font-size:.95rem">'
            f'{icon("info", size="1rem")} Glosario de escenarios de inversión</p>'
        )
        badge_cols = st.columns(3)
        for badge_col, (sc, label) in zip(badge_cols, SCENARIO_LABELS.items()):
            with badge_col:
                st.html(
                    f'<div class="scenario-badge" style="--accent:{SCENARIO_COLORS[sc]}">'
                    f'<b>{icon(SCENARIO_ICONS[sc], color=SCENARIO_COLORS[sc])} {label}</b>'
                    f'<p>{SCENARIO_DESCRIPTIONS[sc]}</p>'
                    f'</div>'
                )
        with st.expander(
            "Definición formal de cada heurística (sin ambigüedad normativa)",
            expanded=False,
            icon=":material/menu_book:",
        ):
            st.markdown(
                "**Escenario Focalizado — Optimización algorítmica.** "
                "Asigna una fracción ρ del presupuesto anual a las cinco unidades "
                "territoriales de mayor prioridad. El peso intra-grupo sigue la "
                "señal SHAP cuando está disponible; si no, el IVC. El remanente "
                "(1−ρ) se reparte de forma uniforme entre el resto. El objetivo "
                "es maximizar el impacto en los territorios de mayor atribución "
                "de vulnerabilidad, no «invertir solo en el peor»."
            )
            st.markdown(
                "**Escenario Distribuido — Igualitario.** "
                "Cada unidad recibe la misma fracción del presupuesto (1/N), "
                "con independencia del IVC y del SHAP. Es el contrapunto de "
                "equidad nominal: igualdad de asignación, no necesariamente "
                "igualdad de resultado. Sirve de referencia para medir cuánto "
                "gana (o pierde) la focalización frente a un reparto uniforme."
            )
            st.markdown(
                "**Escenario Adaptativo — Híbrido.** "
                "Repondera el presupuesto en proporción al IVC corriente "
                "penalizado por la tasa reciente de mejora "
                "(φ = 1/(1+5g), con g ≥ 0,01). "
                "No es una combinación lineal de Focalizado y Distribuido, "
                "sino una heurística independiente que desplaza recursos hacia "
                "territorios vulnerables con menor recuperación observada a "
                "partir del segundo año de simulación."
            )
            st.caption(
                "Las tres reglas operan bajo el mismo techo fiscal. Las "
                "trayectorias no son pronósticos oficiales: son contrafactuales "
                "controlados para contrastar políticas antes de ejecutarlas.",
                help=(
                    "El IVC simulado permanece en [0, 1]. Un descenso de 0,50 a "
                    "0,46 es una mejora relativa de cuatro centésimas del índice "
                    "medio, no un cumplimiento de meta POT. El criterio de "
                    "recomendación en el tablero es: menor IVC medio; en empate, "
                    "menor CV territorial."
                ),
            )
        st.html("<div style='height:.6rem'></div>")

        if ABM_PATH.exists():
            df_abm = load_abm_results(str(ABM_PATH), base_year=ABM_BASE_YEAR)
            abm_ut_col = _resolve_abm_ut_col(df_abm)
            if abm_ut_col is None:
                st.warning(
                    "El CSV de simulación ABM no contiene columna de unidad "
                    "territorial (comuna, localidad, etc.)."
                )
                abm_ut_col = df_abm.columns[0]

            # ── Línea base 2024 ───────────────────────────────────────────────────
            ivc_baseline = float(df_abm["ivc_initial"].mean())
            ivc_base_std = float(df_abm["ivc_initial"].std())

            # ── Agregado por escenario y año calendario ───────────────────────────
            df_agg = (
                df_abm.groupby(["scenario", "calendar_year"], as_index=False)
                .agg(
                    ivc_mean=("ivc_current",  "mean"),
                    ivc_std=("ivc_current",   "std"),
                    wb_mean=("wellbeing",      "mean"),
                    wb_std=("wellbeing",       "std"),
                )
            )

            # Insertar fila de arranque (año 2024, estado basal)
            base_rows = pd.DataFrame([
                {
                    "scenario":     sc,
                    "calendar_year": ABM_BASE_YEAR,
                    "ivc_mean":     ivc_baseline,
                    "ivc_std":      ivc_base_std,
                    "wb_mean":      0.5,
                    "wb_std":       0.0,
                }
                for sc in df_abm["scenario"].unique()
            ])
            df_agg = (
                pd.concat([base_rows, df_agg], ignore_index=True)
                .sort_values(["scenario", "calendar_year"])
            )

            # ── Slider de horizonte temporal ──────────────────────────────────────
            sel_year = st.slider(
                "Horizonte de análisis",
                min_value=ABM_BASE_YEAR,
                max_value=ABM_FINAL_YEAR,
                value=ABM_FINAL_YEAR,
                step=1,
                format="Año %d",
                help=(
                    "Año calendario de la proyección. 2024 es la línea base "
                    "observada; 2025–2034 son estados simulados bajo techo fiscal "
                    "constante. El IVC permanece en [0, 1]: 0,50 es intermedio; "
                    "una caída a 0,46 es una mejora relativa, no un umbral "
                    "absoluto de política."
                ),
            )

            # ── KPIs al año seleccionado ──────────────────────────────────────────
            if sel_year == ABM_BASE_YEAR:
                st.markdown("#### Línea base — **Año 2024** (estado inicial observado)")
            else:
                st.markdown(
                    f"#### Indicadores proyectados al **Año {sel_year}** "
                    f"({sel_year - ABM_BASE_YEAR} año{'s' if sel_year - ABM_BASE_YEAR > 1 else ''} "
                    f"desde la línea base)"
                )

            df_sel  = df_agg[df_agg["calendar_year"] == sel_year]
            kpi_cols = st.columns(len(SCENARIO_LABELS))
            for i, (sc, label) in enumerate(SCENARIO_LABELS.items()):
                row = df_sel[df_sel["scenario"] == sc]
                with kpi_cols[i]:
                    if not row.empty:
                        ivc_val   = float(row["ivc_mean"].iloc[0])
                        reduction = (ivc_baseline - ivc_val) / ivc_baseline * 100
                        render_kpi_card(
                            label=label,
                            value=f"{ivc_val:.4f}",
                            sub="Índice medio de vulnerabilidad",
                            icon_name=SCENARIO_ICONS.get(sc),
                            delta=(f"Reducción de −{reduction:.1f} % vs. base 2024" if reduction >= 0
                                   else f"Aumento de +{abs(reduction):.1f} % vs. base 2024"),
                            delta_positive=reduction >= 0,
                            help_text=(
                                f"{SCENARIO_DESCRIPTIONS[sc]} El valor mostrado es el "
                                "IVC medio simulado en [0, 1]. Un 0,50 es intermedio; "
                                "0,46 es una mejora relativa respecto de 0,50, no un "
                                "corte normativo. La variación porcentual se calcula "
                                "frente a la media observada en 2024."
                            ),
                        )
                    else:
                        render_kpi_card(label=label, value="—", sub="Sin datos",
                                         icon_name=SCENARIO_ICONS.get(sc))

            # ── Política ganadora + selector de escenario a revisar ───────────────
            # Criterio de decisión (alineado a Cap. 4): menor índice medio; en
            # empate, menor CV territorial (más equidad). Solo aplica a años > base.
            win_sc = None
            if sel_year > ABM_BASE_YEAR and not df_sel.empty:
                score_rows = []
                for sc in SCENARIO_LABELS:
                    row = df_sel[df_sel["scenario"] == sc]
                    if row.empty:
                        continue
                    ivc_m = float(row["ivc_mean"].iloc[0])
                    ivc_s = float(row["ivc_std"].iloc[0]) if pd.notna(row["ivc_std"].iloc[0]) else 0.0
                    cv    = (ivc_s / ivc_m) if ivc_m > 0 else 0.0
                    red   = (ivc_baseline - ivc_m) / ivc_baseline * 100
                    score_rows.append({
                        "scenario": sc,
                        "ivc_mean": ivc_m,
                        "cv": cv,
                        "reduction": red,
                    })
                if score_rows:
                    scores = pd.DataFrame(score_rows).sort_values(
                        ["ivc_mean", "cv"], ascending=[True, True]
                    )
                    win = scores.iloc[0]
                    win_sc = win["scenario"]
                    win_label = SCENARIO_LABELS[win_sc]
                    st.html(
                        f'<div class="winner-card">'
                        f'<h4>{icon("verified", color="#047857")} '
                        f'Recomendación al año {sel_year}</h4>'
                        f'<div class="winner-label">{win_label}</div>'
                        f'<p>Es la política con <b>menor índice medio de '
                        f'vulnerabilidad</b> en el horizonte seleccionado'
                        f'{"; además presenta el menor CV territorial" if win["cv"] == scores["cv"].min() else ""}. '
                        f'{SCENARIO_DESCRIPTIONS[win_sc]}</p>'
                        f'<div class="winner-metrics">'
                        f'<span><strong>Índice medio:</strong> {win["ivc_mean"]:.4f}</span>'
                        f'<span><strong>Reducción vs 2024:</strong> {win["reduction"]:.1f} %</span>'
                        f'<span><strong>CV territorial:</strong> {win["cv"]:.3f}</span>'
                        f'</div></div>'
                    )

            sc_keys = list(SCENARIO_LABELS.keys())
            if "abm_focus_sc" not in st.session_state:
                st.session_state.abm_focus_sc = win_sc if win_sc in sc_keys else sc_keys[0]
            focus_sc = st.radio(
                "Política a revisar en detalle",
                options=sc_keys,
                format_func=lambda k: SCENARIO_LABELS[k],
                horizontal=True,
                key="abm_focus_sc",
                help=(
                    "Resalta la heurística en las trayectorias y fija el detalle "
                    "territorial. Focalizado = optimización algorítmica (fracción ρ "
                    "en el top-5). Distribuido = igualitario (1/N). Adaptativo = "
                    "híbrido independiente (reponderación por IVC y eficiencia "
                    "reciente), no una mezcla de las otras dos."
                ),
            )
            st.caption(SCENARIO_DESCRIPTIONS.get(focus_sc, ""))

            # ── Función constructora de gráfico de trayectoria ────────────────────
            def _tray_chart(df_src, y_col, y_std_col, y_label, title):
                fig = go.Figure()
                for sc, label in SCENARIO_LABELS.items():
                    d     = df_src[df_src["scenario"] == sc].sort_values("calendar_year")
                    color = SCENARIO_COLORS[sc]
                    band  = SCENARIO_BAND[sc]
                    xs    = list(d["calendar_year"])
                    ys    = list(d[y_col])
                    ys_s  = list(d[y_std_col])
                    is_focus = sc == focus_sc

                    # Banda dispersión territorial
                    fig.add_trace(go.Scatter(
                        x=xs + xs[::-1],
                        y=[y + s for y, s in zip(ys, ys_s)] +
                          [y - s for y, s in zip(ys, ys_s)][::-1],
                        fill="toself",
                        fillcolor=band,
                        line=dict(color="rgba(0,0,0,0)"),
                        showlegend=False,
                        hoverinfo="skip",
                    ))
                    # Línea principal (más gruesa si es la política en foco)
                    fig.add_trace(go.Scatter(
                        x=xs, y=ys,
                        name=label + ("  ← foco" if is_focus else ""),
                        mode="lines+markers",
                        line=dict(color=color, width=3.5 if is_focus else 2.0),
                        marker=dict(size=9 if is_focus else 6),
                        opacity=1.0 if is_focus else 0.55,
                        hovertemplate=(
                            f"<b>{label}</b><br>"
                            "Año: %{x}<br>"
                            f"{y_label}: %{{y:.4f}}"
                            "<extra></extra>"
                        ),
                    ))

                # Línea vertical al año del slider
                if sel_year > ABM_BASE_YEAR:
                    fig.add_vline(
                        x=sel_year,
                        line_dash="dash",
                        line_color=PALETTE["neutral"],
                        annotation_text=f"{sel_year}",
                        annotation_position="top right",
                        annotation_font_size=11,
                    )
                # Referencia base 2024 solo en gráfico IVC
                if y_col == "ivc_mean":
                    fig.add_hline(
                        y=ivc_baseline,
                        line_dash="dot",
                        line_color=PALETTE["slate"],
                        annotation_text=f"Base 2024  ({ivc_baseline:.3f})",
                        annotation_position="bottom right",
                        annotation_font_size=10,
                    )

                fig.update_layout(
                    title=dict(text=title, font=dict(size=13)),
                    xaxis=dict(
                        title="Año calendario",
                        tickmode="linear",
                        dtick=1,
                        tickformat="d",
                        tickangle=-35,
                        range=[ABM_BASE_YEAR - 0.3, ABM_FINAL_YEAR + 0.3],
                        showgrid=True,
                        gridcolor="#f0f0f0",
                    ),
                    yaxis=dict(
                        title=y_label,
                        showgrid=True,
                        gridcolor="#e8e8e8",
                    ),
                    legend=dict(
                        orientation="h", y=-0.32, x=0,
                        font=dict(size=9),
                    ),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    height=400,
                    margin=dict(l=10, r=10, t=48, b=80),
                )
                apply_chart_theme(fig)
                return fig

            # ── Gráficas de trayectorias ──────────────────────────────────────────
            df_plot = df_agg[df_agg["calendar_year"] <= sel_year]
            col_ivc, col_wb = st.columns(2)

            with col_ivc:
                render_section_title(
                    "Trayectoria del índice de vulnerabilidad", icon_name="trending_down",
                    help_title="Cómo leer este gráfico",
                    help_body=(
                        "Cada línea es el IVC medio proyectado para una heurística. "
                        "La banda (±1 DE) resume la dispersión entre "
                        f"{UT_PLU.lower()}. Una banda más angosta indica un "
                        "resultado más homogéneo. La línea punteada es la media "
                        "observada en 2024. Los valores viven en [0, 1]; 0,50 vs "
                        "0,46 se interpreta en la ayuda de los indicadores, no "
                        "como meta dibujada en el gráfico."
                    ),
                )
                st.plotly_chart(
                    _tray_chart(
                        df_plot, "ivc_mean", "ivc_std",
                        "Índice medio",
                        "Proyección 2024–2034 — índice de vulnerabilidad",
                    ),
                    width="stretch",
                )

            with col_wb:
                render_section_title(
                    "Trayectoria del bienestar", icon_name="trending_up",
                    help_title="¿Qué es el bienestar?",
                    help_body=(
                        "El índice de bienestar es la contraparte positiva del "
                        "índice de vulnerabilidad en la simulación: aumenta "
                        "cuando una zona recibe inversión sostenida y mejora "
                        "su calidad de vida simulada. Escala [0, 1], donde 1 "
                        "representa el máximo bienestar posible."
                    ),
                )
                st.plotly_chart(
                    _tray_chart(
                        df_plot, "wb_mean", "wb_std",
                        "Bienestar medio",
                        "Proyección 2024–2034 — bienestar",
                    ),
                    width="stretch",
                )

            # ── Tabla estadística al año seleccionado ─────────────────────────────
            if sel_year > ABM_BASE_YEAR:
                render_section_title(
                    f"Estadísticos descriptivos — Año {sel_year}",
                    icon_name="table_chart",
                    help_title="Interpretación de las columnas",
                    help_body=(
                        "IVC̄ y WB̄ son las medias del escenario en el año "
                        "seleccionado. DE es la desviación estándar entre "
                        f"{UT_PLU.lower()}. CV (coeficiente de variación = "
                        "DE / IVC̄) mide la desigualdad territorial relativa: "
                        "cuanto más bajo, más equitativo es el resultado del "
                        "escenario, independientemente de su nivel absoluto de "
                        "IVC."
                    ),
                )

                df_det = df_abm[df_abm["calendar_year"] == sel_year].copy()
                df_det["Escenario"] = df_det["scenario"].map(SCENARIO_LABELS)

                def _cv(x):
                    m = x.mean()
                    return x.std() / m if m != 0 else 0.0

                tbl = (
                    df_det.groupby("Escenario")
                    .agg(
                        IVC_medio=("ivc_current",  "mean"),
                        IVC_min=("ivc_current",    "min"),
                        IVC_max=("ivc_current",    "max"),
                        DE=("ivc_current",         "std"),
                        CV=("ivc_current",         _cv),
                        WB_medio=("wellbeing",     "mean"),
                        Reduccion=("ivc_current",
                                   lambda x: (ivc_baseline - x.mean()) / ivc_baseline * 100),
                    )
                    .round(4)
                    .reset_index()
                )
                tbl.columns = [
                    "Escenario", "IVC̄", "IVC mín", "IVC máx",
                    "DE", "CV", "WB̄", "Reducción (%)",
                ]
                st.dataframe(tbl, width="stretch", hide_index=True)

                # Detalle por unidad territorial (usa la política en foco del selector)
                with st.expander(
                    f"Ver detalle por {UT_COL} — {SCENARIO_LABELS[focus_sc]} · Año {sel_year}",
                    expanded=False,
                ):
                    df_det_sc = (
                        df_det[df_det["scenario"] == focus_sc]
                        [[abm_ut_col, "ivc_current", "wellbeing",
                          "investment_allocated", "infrastructure_capacity"]]
                        .rename(columns={
                            abm_ut_col:                 UT_COL,
                            "ivc_current":              "Índice",
                            "wellbeing":                "Bienestar",
                            "investment_allocated":     "Inversión asignada",
                            "infrastructure_capacity":  "Cap. infraestructura",
                        })
                        .sort_values("Índice", ascending=False)
                        .reset_index(drop=True)
                    )
                    st.dataframe(
                        df_det_sc.style.format({
                            "Índice":              "{:.4f}",
                            "Bienestar":           "{:.4f}",
                            "Cap. infraestructura": "{:.4f}",
                            "Inversión asignada":  "{:,.0f}",
                        }),
                        width="stretch",
                        height=320,
                    )

            st.caption(
                f"Fuente: simulación territorial de agentes (Python) · "
                f"Horizonte 2024–{ABM_FINAL_YEAR} · {df_abm[abm_ut_col].nunique()} "
                f"{UT_PLU.lower()} · presupuesto agregado constante",
                help=(
                    "Focalizado = optimización algorítmica; Distribuido = "
                    "igualitario; Adaptativo = híbrido independiente. El IVC "
                    "simulado está en [0, 1]: 0,50 es intermedio; 0,46 es una "
                    "mejora relativa, no un umbral POT. No sustituye un "
                    "pronóstico oficial de la administración municipal."
                ),
            )

        else:
            st.warning(
                f"No se encontró el archivo de simulación ABM en "
                f"`{ABM_PATH.as_posix()}`. "
                "Ejecuta el Pipeline Automático (Medellín) o "
                "`generar_datos_bogota.py` (Bogotá beta) para generarlo."
            )
