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
        padding-top: 3rem;
        padding-bottom: 5rem;
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

    /* SELECTOR DE MODELO (Corregido para que se lea bien) */
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
</style>
""", unsafe_allow_html=True)

# ── Estado de sesión ──
if "historial" not in st.session_state:
    st.session_state.historial = []
if "dsr" not in st.session_state:
    with st.spinner("Cargando modelos de IA..."):
        st.session_state.dsr = Dsr()

# ═══════════════════════════════════════
# CABECERA
# ═══════════════════════════════════════
if not st.session_state.historial:
    st.markdown("<h1 style='text-align:center; font-weight:400; margin-bottom:0.5rem;'>Bienvenido</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#9aa0a6; font-size:18px; margin-top:0;'>¿Qué reseña vamos a analizar hoy?</p>", unsafe_allow_html=True)
else:
    st.markdown("<h2 style='text-align:center; font-weight:400; margin-bottom:2rem;'>DSR-AI</h2>", unsafe_allow_html=True)

# ═══════════════════════════════════════
# RECUADRO PRINCIPAL
# ═══════════════════════════════════════

resena = st.text_area(
    label="resena_oculta", 
    placeholder="Escribe o pega tu reseña aquí. Puedes escribir en cualquier idioma...",
    height=120,
    label_visibility="collapsed",
)

# ── MODELOS ACTUALIZADOS ──
# Modificado según el aviso: el modelo de regresión se llama lr_model.pkl
MODELOS = {
    "KNN (K-Nearest Neighbors)": "knn",
    "Regresión Logística": "lr_model", 
}

col_modelo, col_boton = st.columns([3, 1.5], gap="large")
with col_modelo:
    modelo_nombre = st.selectbox("modelo_oculto", list(MODELOS.keys()), label_visibility="collapsed")
with col_boton:
    st.markdown("<div style='height: 15px'></div>", unsafe_allow_html=True) 
    analizar = st.button("Analizar", use_container_width=True, type="secondary")

# ═══════════════════════════════════════
# LÓGICA DE PREDICCIÓN
# ═══════════════════════════════════════
if analizar:
    modelo_id = MODELOS[modelo_nombre]

    if not resena or not resena.strip():
        st.warning("⚠️ Introduce una reseña para analizar.")
    else:
        with st.spinner("Analizando sentimiento..."):
            prediccion, probabilidades = st.session_state.dsr.predict(resena, modelo=modelo_id)

            es_positivo = prediccion[0] == 1
            prob_pos = probabilidades[0][1] * 100
            prob_neg = probabilidades[0][0] * 100
            confianza = prob_pos if es_positivo else prob_neg
            sentimiento = "POSITIVO" if es_positivo else "NEGATIVO"
            emoji = "😊" if es_positivo else "😞"

            if es_positivo:
                st.success(f"**{emoji} Sentimiento {sentimiento}** — Confianza: {confianza:.1f}%")
            else:
                st.error(f"**{emoji} Sentimiento {sentimiento}** — Confianza: {confianza:.1f}%")

            st.progress(float(confianza / 100))
            st.caption(f"Positivo: {prob_pos:.1f}%  |  Negativo: {prob_neg:.1f}%")

            st.session_state.historial.insert(0, {
                "texto": resena.strip(),
                "sentimiento": sentimiento,
                "confianza": confianza,
                "prob_positivo": prob_pos,
                "prob_negativo": prob_neg,
                "modelo": modelo_nombre.split(" (")[0],
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            })

# ═══════════════════════════════════════
# HISTORIAL
# ═══════════════════════════════════════
if st.session_state.historial:
    st.divider()
    
    col_filtro, col_limpiar = st.columns([4, 1])
    with col_filtro:
        filtro = st.segmented_control(
            "Filtro", ["Todas", "Positivas", "Negativas"],
            default="Todas", label_visibility="collapsed",
            selection_mode="single"
        )
    with col_limpiar:
        st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
        if st.button("Limpiar", use_container_width=True):
            st.session_state.historial = []
            st.rerun()

    historial = st.session_state.historial
    if filtro == "Positivas":
        historial = [r for r in historial if r["sentimiento"] == "POSITIVO"]
    elif filtro == "Negativas":
        historial = [r for r in historial if r["sentimiento"] == "NEGATIVO"]

    total = len(st.session_state.historial)
    positivas = sum(1 for r in st.session_state.historial if r["sentimiento"] == "POSITIVO")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total", total)
    c2.metric("Positivas", positivas)
    c3.metric("Negativas", total - positivas)

    for r in historial:
        emoji_hist = "😊" if r["sentimiento"] == "POSITIVO" else "😞"
        with st.expander(f'{emoji_hist} {r["sentimiento"]} — {r["confianza"]:.1f}% — {r["timestamp"]}'):
            st.write(r["texto"])
            st.caption(f'Modelo: **{r["modelo"]}** |  +{r["prob_positivo"]:.1f}%  /  -{r["prob_negativo"]:.1f}%')

st.divider()
with st.expander("Acerca de DSR-AI"):
    readme_path = os.path.join(os.path.dirname(__file__), '..', 'README.md')
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            st.markdown(f.read())
    except FileNotFoundError:
        st.markdown("""
        **DSR-AI** — Sistema de Detección de Sentimiento en Reseñas de Usuarios.  
        Proyecto desarrollado para la asignatura de Introducción a la IA en la UPV.  
        Modelos utilizados: Doc2Vec + K-Nearest Neighbors + Regresión Logística.
        """)
