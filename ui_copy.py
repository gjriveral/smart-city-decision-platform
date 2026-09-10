# © 2026 Gabriel Jaime Rivera León. Todos los derechos reservados.
"""
ui_copy.py
Textos de interfaz por rol. Editar aquí; no mezclar con lógica de datos.
"""

from __future__ import annotations

# Claves alineadas con ui_chrome.ROLE_OPTIONS / session_state["sidebar_mode"].
ROLE_CONTEXT: dict[str, str] = {
    "Admin": (
        "Como administrador, puede ejecutar el pipeline de datos, actualizar "
        "las fuentes territoriales y supervisar las métricas de validación "
        "del motor predictivo y de las simulaciones de asignación presupuestal."
    ),
    "Decisión": (
        "Como responsable de decisión, puede explorar los escenarios de "
        "inversión territorial, comparar su impacto en el índice de "
        "vulnerabilidad y descargar la ficha de soporte para la asignación "
        "de presupuesto."
    ),
}

# Catálogo único de servicios de UI (no es lógica de dominio).
# icon: Material Symbol de Streamlit (:material/nombre:).
SERVICES: dict[str, dict[str, str]] = {
    "priorizacion": {
        "title": "Priorización territorial",
        "description": "Indicadores, ranking e IVC para decidir dónde invertir primero.",
        "icon": ":material/bar_chart:",
    },
    "mapa": {
        "title": "Mapa de priorización",
        "description": "Cartografía del IVC por comuna, localidad u otra unidad territorial.",
        "icon": ":material/map:",
    },
    "politicas": {
        "title": "Comparar políticas (10 años)",
        "description": "Contraste Focalizado, Distribuido y Adaptativo bajo techo fiscal.",
        "icon": ":material/hub:",
    },
    "pipeline": {
        "title": "Pipeline y fuentes",
        "description": "ETL, carga de datos reales y reentrenamiento del motor predictivo.",
        "icon": ":material/play_arrow:",
    },
    "validacion": {
        "title": "Validación del modelo",
        "description": "Métricas de CV temporal, Gain Importance y atribución SHAP.",
        "icon": ":material/query_stats:",
    },
    "simulacion": {
        "title": "Simulación presupuestal",
        "description": "Supervisión de las trayectorias ABM de asignación a diez años.",
        "icon": ":material/account_tree:",
    },
}

SERVICES_BY_ROLE: dict[str, tuple[str, ...]] = {
    "Decisión": ("priorizacion", "mapa", "politicas"),
    "Admin": ("pipeline", "validacion", "simulacion"),
}


def services_for_role(role: str) -> list[dict[str, str]]:
    ids = SERVICES_BY_ROLE.get(role, SERVICES_BY_ROLE["Decisión"])
    return [{"id": sid, **SERVICES[sid]} for sid in ids if sid in SERVICES]
