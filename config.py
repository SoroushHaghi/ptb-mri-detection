
import os
import random
import numpy as np
import torch

# --- REPRODUCIBILITY SETUP ---
SEED = 42

def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- IMAGING CONSTANTS ---
IMG_SIZE = 256          # Input size for U-Net
CHANNELS = 1            # Force Grayscale
NUM_CLASSES = 1         # Binary Segmentation
THRESHOLD = 0.5         # Sensitivity
