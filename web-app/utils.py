import streamlit as st
import os
import base64

@st.cache_data
def get_cached_css(base_dir):
    """
    Lee los archivos estáticos y codifica la imagen de fondo en base64.
    Esta función está totalmente cacheada para evitar accesos repetidos a disco
    y operaciones redundantes de codificación de imagen.
    """
    css_path = os.path.join(base_dir, 'style.css')
    try:
        with open(css_path, 'r', encoding='utf-8') as f:
            css = f.read()
    except FileNotFoundError:
        css = ""

    bg_path = os.path.join(base_dir, 'bg.png')
    if os.path.exists(bg_path):
        with open(bg_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        css += f"""
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-attachment: fixed;
        }}
        """
    return css

def load_css(base_dir):
    """
    Recupera el CSS optimizado de la caché e inyecta los estilos globalmente.
    """
    css = get_cached_css(base_dir)
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
