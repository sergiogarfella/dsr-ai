"""
DSR-AI — Interfaz de Análisis de Sentimiento en Reseñas (Estilo Light/Minimalista)
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

# ── CSS: Estilo Light Mode Minimalista ──
st.markdown("""
<style>
    /* Fondo principal claro */
    .stApp {
        background-color: #f8f9fa;
        color: #212529;
    }

    /* Ocultar elementos nativos */
    #MainMenu, footer, header { visibility: hidden; }
    .stApp > header { background-color: rgba(248, 249, 250, 0.9); backdrop-filter: blur(10px); }
    
    /* Tipografía */
    html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    
    /* Contenedor centrado */
    .main .block-container {
        max-width: 750px;
        padding-top: 4rem;
        padding-bottom: 5rem;
    }

    /* EL RECUADRO DE TEXTO */
    div[data-testid="stTextArea"] {
        background-color: #ffffff !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 20px !important;
        padding: 20px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.03); /* Sombra muy suave */
        transition: all 0.2s ease;
    }
    div[data-testid="stTextArea"]:focus-within {
        border-color: #4dabf7 !important;
        box-shadow: 0 0 0 3px rgba(77, 171, 247, 0.15);
    }
    div[data-testid="stTextArea"] textarea {
        background-color: transparent !important;
        color: #212529 !important;
        font-size: 16px;
    }
    div[data-testid="stTextArea"] textarea::placeholder {
        color: #adb5bd !important;
    }

    /* SELECTOR DE MODELO */
    div[data-testid="stSelectbox"] label { display: none; }
    div[data-testid="stSelectbox"] {
        background-color: #ffffff !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 12px !important;
        padding: 4px 12px !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        background-color: transparent !important;
        color: #495057 !important;
    }
    .stSelectbox span[data-baseweb="tag"] {
        background-color: transparent !important;
        color: #495057 !important;
        font-weight: normal;
        white-space: normal;
    }
    svg[width="12"] { fill: #868e96; }

    /* BOTÓN DE ANALIZAR (Azul llamativo) */
    .stButton > button[kind="secondary"] {
        background-color: #0d6efd;
        color: #ffffff !important;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        padding: 10px 28px;
        box-shadow: 0 2px 4px rgba(13, 110, 253, 0.2);
        transition: all 0.2s ease;
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: #0b5ed7;
        color: #ffffff !important;
        box-shadow: 0 4px 8px rgba(13, 110, 253, 0.3);
        transform: translateY(-1px);
    }

    /* RESULTADOS (Tarjetas blancas con borde lateral) */
    div.stSuccess {
        background-color: #ffffff;
        border: 1px solid #d3f9d8;
        border-left: 5px solid #2b8a3e;
        border-radius: 12px;
        color: #2b8a3e;
        padding: 1.2rem;
    }
    div.stError {
        background-color: #ffffff;
        border: 1px solid #ffe3e3;
        border-left: 5px solid #e03131;
        border-radius: 12px;
        color: #e03131;
        padding: 1.2rem;
    }
    
    /* BARRA DE PROGRESO */
    .stProgress > div > div > div {
        background-color: #0d6efd;
        border-radius: 10px;
    }

    /* HISTORIAL Y EXPANDERS */
    .stExpander {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        margin-bottom: 8px;
    }
    button[data-baseweb="accordion-trigger"] {
        color: #343a40 !important;
        font-size: 15px;
    }
    
    /* Filtros segmentados */
    .stSegmentedControl > div { background-color: #e9ecef; border-radius: 10px; }
    .stSegmentedControl button { color: #495057; border-radius: 8px; }
    .stSegmentedControl button[aria-pressed="true"] { 
        background-color: #ffffff; 
        color: #0d6efd; 
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Métricas */
    div[data-testid="stMetricValue"] { color: #212529; }
    div[data-testid="stMetricLabel"] { color: #868e96; }
    
    hr { border-color: #dee2e6; margin: 2.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Estado de sesión ──
if "historial" not in st.session_state:
    st.session_state.historial = []
if "dsr" not in st.session_state:
    with st.spinner("Cargando modelos de IA..."):
        st.session_state.dsr = Dsr()

# ═══════════════════════════════════════
# CABECERA (Texto exacto de la imagen)
# ═══════════════════════════════════════
if not st.session_state.historial:
    st.markdown("<h1 style='text-align:center; font-weight:600; color:#212529; margin-bottom:0.2rem;'>Bienvenido a DSR-AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#868e96; font-size:18px; margin-top:0;'>¿Qué reseña vamos a analizar hoy?</p>", unsafe_allow_html=True)
else:
    st.markdown("<h2 style='text-align:center; font-weight:600; color:#212529; margin-bottom:2rem;'>DSR-AI</h2>", unsafe_allow_html=True)

# ═══════════════════════════════════════
# RECUADRO PRINCIPAL
# ═══════════════════════════════════════

resena = st.text_area(
    label="resena_oculta", 
    placeholder="Escribe o pega tu reseña aquí. Puedes escribir en cualquier idioma...",
    height=120,
    label_visibility="collapsed",
)

# ── MODELOS ──
MODELOS = {
    "KNN (K-Nearest Neighbors)": "knn",
    "Regresión Logística": "logistic", 
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

            st.progress(confianza / 100)
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
            st.caption(f'Modelo: **{r["modelo"]}**  |  +{r["prob_positivo"]:.1f}%  /  -{r["prob_negativo"]:.1f}%')

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
