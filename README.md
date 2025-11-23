
# PTB Research: MRI Brain Tumor Segmentation Pipeline

![PTB Status](https://img.shields.io/badge/Status-Research%20Prototype-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Enabled-red)

This repository contains a **Scientific Deep Learning Pipeline** designed for the automated segmentation of brain tumors from MRI scans (LGG Segmentation Dataset). It demonstrates an end-to-end MLOps workflow suitable for medical imaging research.

## 🧠 Project Architecture

The system is built upon a **Stateless Microservices** architecture:

1.  **Ingestion Layer (Telegram Bot):** Secure & rapid image acquisition interface for physicians.
2.  **Core Engine (U-Net++ with EfficientNet-B4 Encoder):** A state-of-the-art deep convolutional neural network for semantic segmentation, leveraging the `segmentation_models_pytorch` library.
    * **Model:** `UnetPlusPlus` with an `efficientnet-b4` backbone.
    * **Loss Function:** Combined `Dice Loss` + `BCEWithLogitsLoss` for training stability.
    * **Metrics:** `Dice Coefficient` (Overlap) and `Hausdorff Distance` (Safety/Geometric accuracy).
3.  **Visualization (Streamlit):** An interactive dashboard providing overlay visualization and human-in-the-loop feedback mechanisms.

## 🚀 Key Features

* **SOTA Model Architecture:** Utilizes `segmentation_models_pytorch` for easy access to state-of-the-art models like `U-Net++`.
* **Scientific Preprocessing:** * Min-Max Normalization (handling 8-bit & 16-bit depth).
    * Padding-based resizing (preserving aspect ratio/geometry).
* **Robust Training:**
    * Data Augmentation via `Albumentations` (Noise, Rotation, Shift) to simulate real-world clinical data variance.
    * **CUDA Acceleration:** Optimized for NVIDIA GPU training.
* **Reproducibility:** Global Random Seed (`42`) enforced across all modules.

## 🛠️ Installation

### Prerequisites
* Python 3.9+
* NVIDIA GPU (Recommended for training) with CUDA 11.8+
* All other dependencies (including `segmentation_models_pytorch`) are listed in `requirements.txt`.

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

3.  Setup Kaggle API:
    Place your `kaggle.json` inside `~/.kaggle/` (or user home directory on Windows).

## 🚦 Usage

### 1. Data Preparation
Download and split the dataset (Patient-wise split to prevent data leakage):
```bash
python utils/data_manager.py
```

### 2. Training the Model
Train the U-Net model (Auto-detects GPU):
```bash
python app/train.py
```
*Artifacts are saved as `model_final.pth`.*

### 3. Running the Dashboard
Launch the physician interface:
```bash
streamlit run app/dashboard.py
```

### 4. Running the Bot (Optional)
To enable the Telegram interface:
```bash
python app/bot.py
```

## 🔬 Scientific Validation
The model is evaluated using **Hausdorff Distance (HD95)** to ensure clinical safety by measuring the maximum outlier distance between the predicted mask and ground truth.

---
*Developed for PTB Department 8.4 Application.*
