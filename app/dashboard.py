import os
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

from config import IMG_SIZE

# --- CONFIGURATION ---
st.set_page_config(page_title="PTB AI Analysis", page_icon="🧠", layout="wide")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/efficientnet_b4_kaggle.pth"

# --- STYLES ---
st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- MODEL LOADER (Single Model Mode) ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"⚠️ Model file missing: {MODEL_PATH}")
        return None
    
    try:
        # SOTA Architecture: U-Net++ with EfficientNet-B4
        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b4",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None
        )
        
        # Load Weights
        state_dict = torch.load(MODEL_PATH, map_location=torch.device(DEVICE))
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess(image_pil):
    # EfficientNet needs RGB
    image = np.array(image_pil.convert('RGB'))
    
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        ToTensorV2()
    ])
    
    # Normalize to 0-1
    image = image.astype(np.float32) / 255.0
    
    aug = transform(image=image)
    return aug['image'].unsqueeze(0)

def create_overlay(image_pil, mask):
    img = np.array(image_pil.convert('RGB'))
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    overlay = img.copy()
    overlay[mask == 1] = [255, 0, 50] # Vivid Red
    
    return cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

# --- UI LAYOUT ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/PTB_Logo.svg/1200px-PTB_Logo.svg.png", width=150)
st.sidebar.title("⚙️ Settings")
st.sidebar.info(f"Model: **EfficientNet-B4**\nDevice: **{DEVICE.upper()}**")
threshold = st.sidebar.slider("Sensitivity", 0.0, 1.0, 0.5)

st.title("🧠 PTB Research: Brain Tumor Segmentation")
st.markdown("Automated analysis using **SOTA U-Net++ Architecture**.")

# Load Model
model = load_model()

if model:
    uploaded = st.file_uploader("Upload MRI Scan", type=['png', 'jpg', 'tif'])
    
    if uploaded:
        col1, col2 = st.columns(2)
        img_pil = Image.open(uploaded)
        
        with col1:
            st.subheader("1. Raw Scan")
            st.image(img_pil, use_container_width=True)
            
        with st.spinner("Running AI Inference..."):
            inp = preprocess(img_pil).to(DEVICE)
            with torch.no_grad():
                out = model(inp)
                pred = (torch.sigmoid(out) > threshold).float().cpu().numpy()[0,0]
        
        with col2:
            st.subheader("2. AI Detection")
            overlay = create_overlay(img_pil, pred)
            st.image(overlay, caption="Red Area = Tumor Prediction", use_container_width=True)
            
        # Stats
        st.divider()
        if pred.max() > 0:
            area = np.sum(pred)
            st.error(f"🚨 **Tumor Detected** (Area: {int(area)} pixels)")
        else:
            st.success("✅ **Clean Scan** - No anomalies detected.")