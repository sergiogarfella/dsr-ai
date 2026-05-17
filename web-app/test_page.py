import streamlit as st
st.markdown("""
<style>
/* Reset */
div[data-testid="stSelectbox"] { margin-bottom: 0px !important; }

/* Proper selectbox styling */
.stSelectbox div[data-baseweb="select"] {
    background-color: rgba(30, 31, 32, 0.45) !important;
    backdrop-filter: blur(12px) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    overflow: hidden !important;
}
.stSelectbox div[data-baseweb="select"] > div {
    background-color: transparent !important;
}
</style>
""", unsafe_allow_html=True)
col1, col2 = st.columns([0.65, 0.35], vertical_alignment="center")
with col1:
    st.selectbox("Test", ["A", "B"], label_visibility="collapsed")
with col2:
    st.button("Button", use_container_width=True)
