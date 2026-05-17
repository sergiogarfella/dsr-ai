"""
DSR-AI — Interfaz de Análisis de Sentimiento en Reseñas (Estilo Gemini)
Proyecto I. Introducción a la IA — UPV
"""

import streamlit as st
import sys
import os
from datetime import datetime

# Añadir el paquete del proyecto al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'proyect-package'))
from dsr import Dsr

# ── Configuración de página ──
st.set_page_config(
    page_title="DSR-AI | Análisis de Sentimiento",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── CSS: Estilo Gemini ──
st.markdown("""
<style>
    /* Fondo principal oscuro tipo Gemini */
    .stApp {
        background-color: #131314;
        color: #e3e3e3;
    }

    /* Ocultar elementos nativos de Streamlit */
    #MainMenu, footer, header { visibility: hidden; }
    .stApp > header { background-color: rgba(19, 19, 20, 0.8); backdrop-filter: blur(10px); }
    
    /* Tipografía general */
    html, body, [class*="css"] { font-family: 'Google Sans', 'Inter', sans-serif; }
    
    /* Contenedor principal centrado */
    .main .block-container {
        max-width: 750px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* EL RECUADRO DE TEXTO */
    div[data-testid="stTextArea"] {
        background-color: #1e1f20 !important;
        border: 1px solid #3c4043 !important;
        border-radius: 24px !important;
        padding: 15px 20px !important;
        transition: all 0.2s ease;
    }
    div[data-testid="stTextArea"]:focus-within {
        border-color: #8ab4f8 !important;
        box-shadow: 0 0 0 1px #8ab4f8;
    }
    div[data-testid="stTextArea"] textarea {
        background-color: transparent !important;
        color: #e3e3e3 !important;
        font-size: 16px;
    }
    div[data-testid="stTextArea"] textarea::placeholder {
        color: #9aa0a6 !important;
    }

    /* SELECTOR DE MODELO */
    div[data-testid="stSelectbox"] {
        background-color: #303134;
        border-radius: 20px;
        padding: 4px;
    }
    div[data-testid="stSelectbox"] label {
        display: none;
    }
    /* Forzar que el texto sea blanco y no se corte */
    .stSelectbox div[data-baseweb="select"] {
        background-color: transparent !important;
        color: #e3e3e3 !important;
    }
    .stSelectbox span[data-baseweb="tag"] {
        background-color: transparent !important;
        color: #e3e3e3 !important;
        font-weight: normal;
        white-space: normal;
    }
    /* Icono de la flecha */
    svg[width="12"] { fill: #9aa0a6; }

    /* BOTÓN DE ANALIZAR */
    .stButton > button[kind="secondary"] {
        background-color: #8ab4f8;
        color: #202124;
        border: none;
        border-radius: 20px;
        font-weight: 600;
        padding: 8px 25px;
        transition: all 0.2s ease;
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: #aecbfa;
        color: #202124;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }

    /* RESULTADOS */
    div.stSuccess, div.stError {
        background-color: #1e1f20;
        border: 1px solid #3c4043;
        border-radius: 16px;
        padding: 1.2rem;
    }
    div.stSuccess { border-left: 4px solid #34a853; }
    div.stError { border-left: 4px solid #ea4335; }
    
    /* BARRA DE PROGRESO */
    .stProgress > div > div > div {
        background-color: #8ab4f8;
        border-radius: 10px;
    }

    /* HISTORIAL Y EXPANDERS */
    .stExpander {
        background-color: #1e1f20;
        border: 1px solid #3c4043;
        border-radius: 16px;
        margin-bottom: 10px;
    }
    button[data-baseweb="accordion-trigger"] {
        color: #e3e3e3 !important;
        font-size: 15px;
    }
    .stSegmentedControl > div { background-color: #303134; border-radius: 12px; }
    .stSegmentedControl button { color: #9aa0a6; border-radius: 10px; }
    .stSegmentedControl button[aria-pressed="true"] { background-color: #8ab4f8; color: #202124; }
    
    div[data-testid="stMetricValue"] { color: #e3e3e3; }
    div[data-testid="stMetricLabel"] { color: #9aa0a6; }
    hr { border-color: #3c4043; margin: 2rem 0; }
    
    /* PIE DE PÁGINA */
    .footer {
        text-align: center;
        color: #9aa0a6;
        font-size: 14px;
        margin-top: 4rem;
        padding-top: 1.5rem;
        border-top: 1px solid #3c4043;
    }
</style>
""", unsafe_allow_html=True)

# ── Estado de sesión ──
if "historial" not in st.session_state:
    st.session_state.historial = []
if "dsr" not in st.session_state:
    with st.spinner("Cargando modelos de IA..."):
        st.session_state.dsr = Dsr()

# ════
