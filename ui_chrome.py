# © 2026 Gabriel Jaime Rivera León. Todos los derechos reservados.
# Barra superior de la plataforma — solo capa de presentación Streamlit.
"""
ui_chrome.py
Barra sticky: logo y selector de rol (Decisión | Admin). El rol se
guarda en st.session_state bajo la clave ya usada por el dashboard
(`sidebar_mode`).
"""

from __future__ import annotations

import html as html_lib
from pathlib import Path

import streamlit as st

from ui_copy import ROLE_CONTEXT, services_for_role

ROLE_KEY = "sidebar_mode"
ROLE_DECISION = "Decisión"
ROLE_ADMIN = "Admin"
ROLE_OPTIONS = (ROLE_DECISION, ROLE_ADMIN)
_PREV_ROLE_KEY = "_prev_app_role"

# Estado de vista por rol (no es auth). Al cambiar de rol se limpia para
# volver al mensaje contextual + bloques de servicio de ese modo.
_ROLE_VIEW_KEYS = (
    "service_selected",
    "main_tabs",
)


def current_role() -> str:
    value = st.session_state.get(ROLE_KEY, ROLE_DECISION)
    return value if value in ROLE_OPTIONS else ROLE_DECISION


def _reset_role_view() -> None:
    for key in _ROLE_VIEW_KEYS:
        st.session_state.pop(key, None)
    st.session_state["role_view_epoch"] = (
        int(st.session_state.get("role_view_epoch", 0)) + 1
    )


def _on_role_change(new_role: str) -> None:
    prev = st.session_state.get(_PREV_ROLE_KEY)
    if prev is None:
        st.session_state[_PREV_ROLE_KEY] = new_role
        return
    if prev != new_role:
        st.session_state[_PREV_ROLE_KEY] = new_role
        _reset_role_view()


def _render_role_selector() -> str:
    if st.session_state.get(ROLE_KEY) not in ROLE_OPTIONS:
        st.session_state[ROLE_KEY] = ROLE_DECISION
    help_txt = (
        "Decisión: priorización territorial y políticas. "
        "Admin: pipeline y métricas de validación."
    )
    kwargs = dict(
        label="Selector de rol: Decisión o Admin",
        options=list(ROLE_OPTIONS),
        key=ROLE_KEY,
        help=help_txt,
        label_visibility="collapsed",
    )
    if hasattr(st, "segmented_control"):
        role = st.segmented_control(required=True, **kwargs)
    else:
        role = st.radio(horizontal=True, **kwargs)
    if role not in ROLE_OPTIONS:
        return current_role()
    return role


def _return_to_service_hub() -> None:
    """Sale del dashboard y vuelve a las tarjetas de servicio."""
    st.session_state.pop("service_selected", None)
    st.session_state.pop("main_tabs", None)
    st.rerun()


def render_app_topbar(*, logo_path: Path, product_name: str) -> str:
    """Barra superior única. Devuelve el rol activo (Decisión | Admin)."""
    with st.container():
        st.html('<span class="smic-topbar-marker" aria-hidden="true"></span>')
        col_logo, _mid, col_role = st.columns(
            [1.55, 4.35, 2.2],
            vertical_alignment="center",
            gap="small",
        )
        with col_logo:
            if logo_path.exists():
                st.image(str(logo_path), width=120)
            else:
                st.markdown(f"**{product_name}**")
        with col_role:
            role = _render_role_selector()
    _on_role_change(role)
    return role


def render_dashboard_back_footer() -> None:
    """Pastilla 'Volver atrás' fija en la esquina inferior derecha."""
    if not st.session_state.get("service_selected"):
        return
    with st.container():
        st.html('<span class="smic-back-footer-marker" aria-hidden="true"></span>')
        if st.button(
            "Volver atrás",
            key="smic_back_to_hub",
            icon=":material/arrow_back:",
            type="tertiary",
            help="Volver a la vista donde elige el servicio",
            width="content",
        ):
            _return_to_service_hub()


def render_role_context() -> None:
    """Hero de aterrizaje. Solo presentación; no altera rol ni navegación."""
    if st.session_state.get("service_selected"):
        return
    role = current_role()
    text = ROLE_CONTEXT.get(role, ROLE_CONTEXT[ROLE_DECISION])
    eyebrow = (
        "Modo decisión" if role == ROLE_DECISION else "Modo administración"
    )
    st.html(
        '<section class="smic-hero">'
        f'<p class="smic-hero-eyebrow">{html_lib.escape(eyebrow)}</p>'
        '<span class="smic-hero-rule" aria-hidden="true"></span>'
        '<h2 class="smic-hero-title">'
        "Planificación estratégica<br>de inversiones urbanas."
        "</h2>"
        f'<p class="smic-hero-lede">{html_lib.escape(text)}</p>'
        "</section>"
    )


CARD_EMOJIS = {
    "priorizacion": "📊",
    "mapa": "🗺️",
    "politicas": "⚛️",
    "pipeline": "⚙️",
    "validacion": "📈",
    "simulacion": "🏛️",
}

CUSTOM_CARD_CSS = """
<style>
.svc-hit-area{display:none;}
div[data-testid="stHorizontalBlock"]:has(.svc-hit-area){
  align-items:stretch !important;
}
div[data-testid="stColumn"]:has(.svc-hit-area){
  display:flex;
  align-items:stretch;
}
div[data-testid="stColumn"]:has(.svc-hit-area) > div{
  width:100%;
  height:100%;
  min-height:15.25rem;
  background:#fff;
  border-radius:14px;
  padding:1.45rem 1.25rem 1.35rem;
  border:1px solid rgba(0,123,94,.08);
  box-shadow:0 4px 14px rgba(31,41,55,.05);
  box-sizing:border-box;
  display:flex;
  flex-direction:column;
  align-items:center;
  text-align:center;
  transition:transform .2s ease, box-shadow .2s ease, border-color .2s ease;
}
div[data-testid="stColumn"]:has(.svc-hit-area):hover > div{
  transform:translateY(-4px);
  border-color:rgba(0,123,94,.28);
  box-shadow:0 10px 24px rgba(0,123,94,.12);
}
div[data-testid="stColumn"]:has(.svc-hit-area) [data-testid="stHtml"]{
  display:none;
}
div[data-testid="stColumn"]:has(.svc-hit-area) [data-testid="stMarkdown"]{
  width:100%;
  flex:1;
}
div[data-testid="stColumn"]:has(.svc-hit-area) [data-testid="stVerticalBlock"]{
  gap:.2rem !important;
}
div[data-testid="stColumn"]:has(.svc-hit-area) [data-testid="stButton"]{
  display:flex;
  justify-content:center;
  width:100%;
  margin:0 0 .7rem 0;
  flex-shrink:0;
}
div[data-testid="stColumn"]:has(.svc-hit-area) [data-testid="stButton"] button{
  width:3.15rem !important;
  height:3.15rem !important;
  min-width:3.15rem !important;
  max-width:3.15rem !important;
  min-height:3.15rem !important;
  padding:0 !important;
  margin:0 auto !important;
  border-radius:50% !important;
  background:#ecfdf5 !important;
  border:1px solid rgba(0,123,94,.18) !important;
  box-shadow:none !important;
  font-size:1.4rem !important;
  line-height:1 !important;
  position:relative !important;
  display:flex !important;
  align-items:center !important;
  justify-content:center !important;
  overflow:visible !important;
  cursor:pointer !important;
  transition:transform .18s ease, background-color .18s ease, border-color .18s ease,
    box-shadow .18s ease;
}
div[data-testid="stColumn"]:has(.svc-hit-area) [data-testid="stButton"] button p{
  position:absolute !important;
  width:1px !important;
  height:1px !important;
  overflow:hidden !important;
  clip:rect(0,0,0,0) !important;
  margin:0 !important;
  padding:0 !important;
  border:0 !important;
}
div[data-testid="stColumn"]:has(.svc-hit-area) [data-testid="stButton"] button [data-testid="stIconMaterial"],
div[data-testid="stColumn"]:has(.svc-hit-area) [data-testid="stButton"] button [data-testid="stIconEmoji"]{
  font-size:1.45rem !important;
  color:#047857 !important;
}
div[data-testid="stColumn"]:has(.svc-hit-area) [data-testid="stButton"] button:hover{
  transform:scale(1.08);
  background:#d1fae5 !important;
  border-color:#007B5E !important;
  box-shadow:0 4px 12px rgba(0,123,94,.18) !important;
}
div[data-testid="stColumn"]:has(.svc-hit-area) [data-testid="stButton"] button:focus-visible{
  outline:2px solid #007B5E !important;
  outline-offset:3px !important;
}
.custom-card{
  background:transparent;
  border:none;
  box-shadow:none;
  padding:0;
  margin:0;
  width:100%;
  flex:1;
  display:flex;
  flex-direction:column;
  align-items:center;
  text-align:center;
}
.custom-card .card-title{
  font-size:1.05rem;
  font-weight:700;
  color:#1F2937;
  margin:0 0 .65rem 0;
  min-height:2.7em;
  line-height:1.35;
  text-align:center;
  width:100%;
}
.custom-card .card-desc{
  font-size:.9rem;
  color:#6b7280;
  margin:0;
  min-height:3.2em;
  line-height:1.45;
  text-align:center;
  width:100%;
  flex:1;
}
</style>
"""


def custom_card_html(*, title: str, description: str) -> str:
    """Cuerpo de la tarjeta (título + texto). Divs para que Streamlit no cree anclas."""
    return (
        '<div class="custom-card">'
        f'<div class="card-title">{html_lib.escape(title)}</div>'
        f'<div class="card-desc">{html_lib.escape(description)}</div>'
        "</div>"
    )


def render_service_card(service: dict[str, str], *, key_suffix: str) -> bool:
    """Icono circular = botón nativo que abre el dashboard; alturas iguales."""
    icon = service.get("icon") or CARD_EMOJIS.get(service["id"], "▸")
    st.html('<span class="svc-hit-area" aria-hidden="true"></span>')
    opened = st.button(
        service["title"],
        key=f"svc_{service['id']}_{key_suffix}",
        icon=icon,
        help=f"Abrir: {service['title']}",
        type="tertiary",
        width="content",
    )
    st.markdown(
        custom_card_html(
            title=service["title"],
            description=service["description"],
        ),
        unsafe_allow_html=True,
    )
    return opened


def maybe_render_service_hub() -> None:
    """Tres tarjetas HTML en columnas nativas; si no hay selección, no carga el dashboard."""
    selected = st.session_state.get("service_selected")
    if selected:
        return

    services = services_for_role(current_role())
    epoch = st.session_state.get("role_view_epoch", 0)
    while len(services) < 3:
        services.append(None)

    st.html(CUSTOM_CARD_CSS)
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        if services[0] is not None and render_service_card(
            services[0], key_suffix=str(epoch)
        ):
            st.session_state["service_selected"] = services[0]["id"]
            st.rerun()

    with col2:
        if services[1] is not None and render_service_card(
            services[1], key_suffix=str(epoch)
        ):
            st.session_state["service_selected"] = services[1]["id"]
            st.rerun()

    with col3:
        if services[2] is not None and render_service_card(
            services[2], key_suffix=str(epoch)
        ):
            st.session_state["service_selected"] = services[2]["id"]
            st.rerun()

    st.stop()
