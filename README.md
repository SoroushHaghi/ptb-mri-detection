# PTB Research: MRI Brain Tumor Segmentation

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ptb-mri-detection.streamlit.app)

![PTB Status](https://img.shields.io/badge/Status-Research%20Prototype-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Enabled-red)

This repository contains a **Scientific Deep Learning Application** for the automated segmentation of brain tumors from MRI scans. It provides an interactive Streamlit dashboard for visualizing the results.

## 🧠 Project Architecture

The system is a **Streamlit Web Application** that uses a pre-trained **U-Net++ with an EfficientNet-B4 Encoder** for semantic segmentation.

*   **Model:** `UnetPlusPlus` with an `efficientnet-b4` backbone.
*   **Framework:** `segmentation_models_pytorch`
*   **Interface:** Streamlit

## 🚀 Key Features

*   **SOTA Model Architecture:** Utilizes `segmentation_models_pytorch` for easy access to state-of-the-art models like `U-Net++`.
*   **Interactive Dashboard:** An intuitive Streamlit interface for uploading MRI scans and visualizing the segmentation results.
*   **CUDA Acceleration:** Optimized for NVIDIA GPU inference.

## 🛠️ Installation

### Prerequisites

*   Python 3.9+
*   All other dependencies are listed in `requirements.txt`.

### Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/SoroushHaghi/ptb-mri-detection.git
    cd ptb-mri-detection
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 🚦 Usage

### Running the Dashboard

Launch the physician interface:

```bash
streamlit run app/dashboard.py
```

You can also access the deployed application here: [https://ptb-mri-detection.streamlit.app](https://ptb-mri-detection.streamlit.app)

---

*Developed for PTB Department 8.4 Application.*