import streamlit as st
import os
import base64

def load_css(base_dir):
    """
    Carga el archivo CSS externo y codifica el fondo en base64 para inyectarlo 
    globalmente en la aplicación Streamlit.
    """
    css_path = os.path.join(base_dir, 'style.css')
    try:
        with open(css_path, 'r') as f:
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
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
