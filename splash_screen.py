# © 2026 Gabriel Jaime Rivera León. Todos los derechos reservados.
# Vista de entrada (splash) — solo capa de presentación Streamlit.
"""
splash_screen.py
Pantalla de entrada de SMIC Analytics: logo, autoría, progreso 0–100 %
en al menos 5 s, cue de audio institucional y fondo de partículas.

No carga modelos, índices ni pipelines. El porcentaje se calibra al
tiempo mínimo de espera (5 s) porque la inicialización de datos ocurre
después, en el dashboard.

Cue de audio:
  Coloque el archivo final en assets/audio/intro-cue.mp3
  Mientras tanto se usa assets/splash/intro-cue.wav (placeholder).
"""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
SPLASH_DIR = ROOT / "assets" / "splash"
INTRO_CUE_MP3 = ROOT / "assets" / "audio" / "intro-cue.mp3"

_splash_component = None


def _get_splash_component():
    """Declaración perezosa: evita tocar Streamlit antes de set_page_config."""
    global _splash_component
    if _splash_component is None:
        _splash_component = components.declare_component(
            "smic_splash",
            path=str(SPLASH_DIR),
        )
    return _splash_component


def _mp3_data_uri() -> str | None:
    if not INTRO_CUE_MP3.exists():
        return None
    b64 = base64.b64encode(INTRO_CUE_MP3.read_bytes()).decode("ascii")
    return f"data:audio/mpeg;base64,{b64}"


def maybe_render_splash() -> None:
    """Muestra el splash una vez por sesión y detiene el resto de la app."""
    if st.session_state.get("splash_complete"):
        return

    st.html(
        """
<style>
section[data-testid="stSidebar"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
footer { display:none !important; }
[data-testid="stAppViewContainer"],
.stApp { background: #F5F5F4 !important; } /* --smic-canvas */
.block-container {
  padding: 0 !important;
  max-width: 100% !important;
}
iframe {
  position: fixed !important;
  inset: 0 !important;
  width: 100vw !important;
  height: 100vh !important;
  max-width: none !important;
  border: none !important;
  z-index: 99999;
}
</style>
"""
    )

    result = _get_splash_component()(
        audio_uri=_mp3_data_uri(),
        default=None,
        key="smic_splash",
    )
    if isinstance(result, dict) and result.get("done"):
        st.session_state.splash_complete = True
        st.rerun()
    st.stop()
