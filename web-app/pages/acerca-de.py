import streamlit as st
import os

# ── Configuración de página ──
st.set_page_config(
    page_title="Acerca de | DSR-AI",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Cargar CSS externo ──
from utils import load_css
base_dir = os.path.join(os.path.dirname(__file__), '..')
load_css(base_dir)

# Botón para volver usando page_link
st.page_link("app.py", label="Volver al inicio")
st.divider()

# Cargar README
readme_path = os.path.join(os.path.dirname(__file__), '..', '..', 'README.md')
try:
    with open(readme_path, 'r', encoding='utf-8') as f:
        st.markdown(f.read())
except FileNotFoundError:
    st.markdown("""
    # Sistema de Detección de Sentimiento en Reseñas de Usuarios (DSR-AI)
    
    Proyecto desarrollado para la asignatura de la UPV.  
    Modelos utilizados: Doc2Vec + K-Nearest Neighbors + Regresión Logística.
    """)
