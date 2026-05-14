"""
DSR-AI — Interfaz de Análisis de Sentimiento en Reseñas
Proyecto I. Introducción a la IA — UPV
"""

import streamlit as st
import sys
import os
from datetime import datetime

# Añadir el paquete del proyecto al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'proyect-package'))
from dsr.dsr import Dsr

# ── Configuración de página ──
st.set_page_config(
    page_title="DSR-AI | Análisis de Sentimiento",
    layout="centered",
)

# ── CSS: solo fuente Inter y estilos mínimos ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter'; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Estado de sesión
if "historial" not in st.session_state:
    st.session_state.historial = []
if "dsr" not in st.session_state:
    with st.spinner("Cargando modelos..."):
        st.session_state.dsr = Dsr()

# Pestañas
tab_analisis, tab_acerca = st.tabs(["Análisis", "Acerca de"])

# ═══════════════════════════════════════
# Pestaña: Análisis
# ═══════════════════════════════════════
with tab_analisis:

    st.markdown(
        "<h1 style='text-align:center; margin-top:2rem; margin-bottom:1.5rem;'>DSR-AI</h1>",
        unsafe_allow_html=True,
    )

    # Campo de texto
    resena = st.text_area(
        "Escribe una reseña",
        placeholder="Escribe o pega tu reseña aquí. Puedes escribir en cualquier idioma...",
        height=150,
        label_visibility="collapsed",
    )

    # Selector de modelo + botón (misma fila)
    MODELOS = {
        "KNN (K-Nearest Neighbors)": "knn",
        "Regresión Logística (Próximamente)": None,
    }

    col_modelo, col_boton = st.columns([2, 1])
    with col_modelo:
        modelo_nombre = st.selectbox("Modelo", list(MODELOS.keys()), label_visibility="collapsed")
    with col_boton:
        analizar = st.button("Analizar sentimiento", use_container_width=True)

    # ── Predicción ──
    if analizar:
        modelo_id = MODELOS[modelo_nombre]

        if not resena or not resena.strip():
            st.warning("Introduce una reseña para analizar.")
        elif modelo_id is None:
            st.info("Este modelo aún no está disponible. Selecciona otro.")
        else:
            with st.spinner("Analizando..."):
                prediccion, probabilidades = st.session_state.dsr.predict(resena.strip(), modelo=modelo_id)

                es_positivo = prediccion[0] == 1
                prob_pos = probabilidades[0][1] * 100
                prob_neg = probabilidades[0][0] * 100
                confianza = prob_pos if es_positivo else prob_neg
                sentimiento = "POSITIVO" if es_positivo else "NEGATIVO"

                # Resultado
                if es_positivo:
                    st.success(f"**{sentimiento}** — Confianza: {confianza:.1f}%")
                else:
                    st.error(f"**{sentimiento}** — Confianza: {confianza:.1f}%")

                st.progress(confianza / 100)
                st.caption(f"Positivo: {prob_pos:.1f}%  |  Negativo: {prob_neg:.1f}%")

                # Guardar en historial
                st.session_state.historial.insert(0, {
                    "texto": resena.strip(),
                    "sentimiento": sentimiento,
                    "confianza": confianza,
                    "prob_positivo": prob_pos,
                    "prob_negativo": prob_neg,
                    "modelo": modelo_nombre.split(" (")[0],
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                })

    # ── Historial ──
    if st.session_state.historial:
        st.divider()
        st.subheader("Historial")

        col_filtro, col_limpiar = st.columns([3, 1])
        with col_filtro:
            filtro = st.segmented_control(
                "Filtro", ["Todas", "Positivas", "Negativas"],
                default="Todas", label_visibility="collapsed",
            )
        with col_limpiar:
            if st.button("Limpiar historial", use_container_width=True):
                st.session_state.historial = []
                st.rerun()

        # Filtrar
        historial = st.session_state.historial
        if filtro == "Positivas":
            historial = [r for r in historial if r["sentimiento"] == "POSITIVO"]
        elif filtro == "Negativas":
            historial = [r for r in historial if r["sentimiento"] == "NEGATIVO"]

        # Estadísticas
        total = len(st.session_state.historial)
        positivas = sum(1 for r in st.session_state.historial if r["sentimiento"] == "POSITIVO")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", total)
        c2.metric("Positivas", positivas)
        c3.metric("Negativas", total - positivas)

        # Lista
        for r in historial:
            with st.expander(f'{r["sentimiento"]} — {r["confianza"]:.1f}% — {r["timestamp"]}'):
                st.write(r["texto"])
                st.caption(f'{r["modelo"]}  |  Positivo: {r["prob_positivo"]:.1f}%  |  Negativo: {r["prob_negativo"]:.1f}%')

# ═══════════════════════════════════════
# Pestaña: Acerca de
# ═══════════════════════════════════════
with tab_acerca:
    readme_path = os.path.join(os.path.dirname(__file__), '..', 'README.md')
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            st.markdown(f.read())
    except FileNotFoundError:
        st.warning("No se encontró el archivo README.md")