import streamlit as st
import os
import re

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

# ── Cargar y renderizar README ──
readme_path = os.path.join(os.path.dirname(__file__), '..', '..', 'README.md')
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Dividir el contenido por las líneas de imagen ![alt](ruta)
    # para renderizarlas con st.image() (Streamlit no sirve archivos locales en markdown)
    image_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
    parts = image_pattern.split(content)

    # parts alterna: [texto, alt, src, texto, alt, src, ...]
    i = 0
    while i < len(parts):
        if i % 3 == 0:
            # Bloque de markdown puro
            if parts[i].strip():
                st.markdown(parts[i], unsafe_allow_html=True)
        else:
            # alt = parts[i], src = parts[i+1]
            alt = parts[i]
            src = parts[i + 1]
            img_path = os.path.join(repo_root, src)
            if os.path.exists(img_path):
                st.image(img_path, caption=alt if alt else None)
            else:
                st.markdown(f"*Imagen no encontrada: `{src}`*")
            i += 1  # saltar el src, el bucle suma otro +1
        i += 1

except FileNotFoundError:
    st.markdown("""
    # Sistema de Detección de Sentimiento en Reseñas de Usuarios (DSR-AI)

    Proyecto desarrollado para la asignatura de la UPV.  
    Modelos utilizados: Doc2Vec + K-Nearest Neighbors + Regresión Logística.
    """)
