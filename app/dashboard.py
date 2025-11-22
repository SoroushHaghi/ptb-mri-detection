import sys
import os
# --- PATH FIX (Add root directory to path) ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from app.model import UNet
from config import IMG_SIZE, SEED

# --- CONFIGURATION ---
st.set_page_config(
    page_title="PTB AI Research Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "model_final.pth"

# --- STYLES ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNCTIONS ---
@st.cache_resource
def load_model():
    # Check if model exists in root
    if not os.path.exists(MODEL_PATH):
        return None
    
    model = UNet(n_channels=1, n_classes=1)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
    except:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        
    model.to(DEVICE)
    model.eval()
    return model

def preprocess_image(image_pil):
    # Convert PIL to Numpy (Grayscale)
    image = np.array(image_pil.convert('L'))
    
    # Albumentations Pipeline (Same as validation)
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        ToTensorV2()
    ])
    
    # Normalize (Min-Max)
    if image.max() > 0:
        image = (image - image.min()) / (image.max() - image.min())
    else:
        image = image / 1.0
        
    augmented = transform(image=image.astype(np.float32))
    return augmented['image'].unsqueeze(0) # Add batch dim

def postprocess_mask(pred_tensor, threshold):
    pred = torch.sigmoid(pred_tensor)
    pred = (pred > threshold).float()
    return pred.squeeze().cpu().detach().numpy()

def create_overlay(original_pil, mask_np):
    # Resize mask back to original image size for visualization
    original_np = np.array(original_pil.convert('RGB'))
    mask_resized = cv2.resize(mask_np, (original_np.shape[1], original_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Create red overlay
    overlay = original_np.copy()
    overlay[mask_resized == 1] = [255, 0, 0] # Red color
    
    # Blend
    alpha = 0.4
    blended = cv2.addWeighted(overlay, alpha, original_np, 1 - alpha, 0)
    return blended

# --- UI LAYOUT ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/PTB_Logo.svg/1200px-PTB_Logo.svg.png", width=150)
st.sidebar.title("⚙️ Control Panel")
st.sidebar.info(f"Running on: **{DEVICE.upper()}**")

threshold = st.sidebar.slider("Sensitivity Threshold", 0.0, 1.0, 0.5, 0.05, help="Lower value = More sensitive (finds smaller tumors)")

st.title("🧠 PTB Medical Imaging: Brain Tumor Segmentation")
st.markdown("### Department 8.4: Mathematical Modeling & Data Analysis")
st.divider()

# --- MAIN LOGIC ---
model = load_model()

if model is None:
    st.warning("⚠️ Model not found yet. Please wait for training to finish.")
else:
    uploaded_file = st.file_uploader("Upload MRI Scan (TIF, JPG, PNG)", type=['tif', 'jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        col1, col2, col3 = st.columns([1, 1, 1])
        
        # 1. Show Original
        image_pil = Image.open(uploaded_file)
        with col1:
            st.subheader("1. Input Scan")
            st.image(image_pil, use_container_width=True)

        # 2. Run AI
        with st.spinner("Analysing patterns..."):
            input_tensor = preprocess_image(image_pil).to(DEVICE)
            with torch.no_grad():
                output = model(input_tensor)
            
            mask = postprocess_mask(output, threshold)

        # 3. Show Result
        with col2:
            st.subheader("2. AI Mask")
            st.image(mask, clamp=True, channels='GRAY', use_container_width=True)
            
        # 4. Show Overlay
        with col3:
            st.subheader("3. Physician View")
            overlay_img = create_overlay(image_pil, mask)
            st.image(overlay_img, use_container_width=True)
            
        # --- ANALYSIS & FEEDBACK ---
        st.divider()
        if mask.max() > 0:
            st.error(f"🚨 Tumor Detected! Coverage: {mask.mean()*100:.2f}% of slice area.")
        else:
            st.success("✅ No Tumor Detected.")
            
        with st.expander("👨‍⚕️ Physician Feedback Loop (Human-in-the-loop)"):
            feedback = st.text_area("Correct the diagnosis if needed:", placeholder="e.g., False positive on the right lobe...")
            if st.button("Submit Feedback"):
                st.toast("Feedback sent to Telegram Archive! (Simulated)", icon="🚀")