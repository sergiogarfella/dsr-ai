"""
DSR-AI — Interfaz de Análisis de Sentimiento en Reseñas
Proyecto de la UPV
"""

import streamlit as st
import sys
import os
from datetime import datetime

# Añadir el paquete del proyecto al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dsr import Dsr
from utils import load_css

MODELOS = {
    "KNN (K-Nearest Neighbors)": "knn",
    "Regresión Logística": "lr_model", 
}

def configurar_pagina():
    """Configura el entorno de Streamlit, carga el CSS y el estado de sesión."""
    st.set_page_config(
        page_title="DSR-AI | Análisis de Sentimiento",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    load_css(os.path.dirname(__file__))

    if "historial" not in st.session_state:
        st.session_state.historial = []
    if "dsr" not in st.session_state:
        with st.spinner("Cargando modelos..."):
            st.session_state.dsr = Dsr()

def mostrar_cabecera():
    """Renderiza el menú superior y el título dinámico."""
    st.markdown("""
    <div style='text-align: right; margin-bottom: -20px; z-index: 100; position: relative;'>
        <a href='acerca-de' target='_self' style='color: #9aa0a6; text-decoration: none; font-weight: 500;'>Acerca de</a>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.historial:
        st.markdown("<div style='text-align:center; font-size:2.2em; font-weight:700; margin-bottom:0.5rem; margin-top:-1rem;'>Bienvenido</div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#9aa0a6; font-size:18px; margin-top:0;'>¿Qué reseña vamos a analizar hoy?</p>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align:center; font-size:1.8em; font-weight:700; margin-bottom:2rem; margin-top:-1rem;'>DSR-AI</div>", unsafe_allow_html=True)

def mostrar_formulario():
    """Muestra la caja de texto, el selector de modelos y el botón. Retorna los inputs."""
    resena = st.text_area(
        label="resena_oculta", 
        placeholder="Escribe o pega tu reseña aquí. Puedes escribir en cualquier idioma...",
        height=120,
        label_visibility="collapsed",
    )
    
    col_model, col_btn = st.columns([0.65, 0.35], vertical_alignment="center")
    with col_model:
        modelo_nombre = st.selectbox("modelo_oculto", list(MODELOS.keys()), label_visibility="collapsed")
    with col_btn:
        analizar = st.button("Analizar", use_container_width=True, type="secondary")
        
    return resena, modelo_nombre, analizar

def ejecutar_analisis(resena, modelo_nombre):
    """Ejecuta la predicción del modelo y actualiza la UI y el historial."""
    if not resena or not resena.strip():
        st.warning("⚠️ Introduce una reseña para analizar.")
        return

    modelo_id = MODELOS[modelo_nombre]
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

def mostrar_historial():
    """Dibuja el historial filtrable de análisis anteriores."""
    if not st.session_state.historial:
        return

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

def mostrar_pie():
    """Renderiza el footer estático al final de la página."""
    st.markdown("""
    <div class="footer">
        Proyecto realizado por Adrián A. Acosta Villegas, Sergio Garfella Pérez, Ainara Sanfélix Ruiz, Jairo E. Urdaneta Colmenares.
    </div>
    """, unsafe_allow_html=True)

def main():
    configurar_pagina()
    mostrar_cabecera()
    resena, modelo_nombre, analizar = mostrar_formulario()
    if analizar:
        ejecutar_analisis(resena, modelo_nombre)
    mostrar_historial()
    mostrar_pie()

if __name__ == "__main__":
    main()
