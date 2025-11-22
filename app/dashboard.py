
import streamlit as st
from config import SEED

st.set_page_config(page_title="PTB MRI Segmentation", layout="wide")
st.title("🧠 PTB Research: AI-Powered MRI Segmentation")
st.sidebar.info(f"System Seed: {SEED} (Reproducibility Active)")
st.write("Upload an MRI scan to begin analysis.")
