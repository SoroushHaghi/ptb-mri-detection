
import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import glob
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.model import UNet
from utils.metrics import CombinedLoss, dice_coeff, compute_hausdorff_distance
from config import seed_everything, SEED, IMG_SIZE, CHANNELS

# --- 1. REPRODUCIBILITY ---
seed_everything(SEED)

# --- 2. DATASET CLASS (Preprocessing Logic) ---
class MRIDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.images = sorted(glob.glob(f"{root_dir}/{split}/images/*.tif"))
        self.masks = sorted(glob.glob(f"{root_dir}/{split}/masks/*.tif"))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        # A. READ IMAGE (GrayScale Force)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # B. READ MASK
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # C. MIN-MAX SCALING (Scientific Normalization)
        # Handling 16-bit or 8-bit dynamic range
        if image.max() > 0:
            image = (image - image.min()) / (image.max() - image.min())
        else:
            image = image / 1.0 # Handle empty black images
            
        image = image.astype(np.float32)
        mask = (mask / 255.0).astype(np.float32) # Convert mask to 0-1
        mask = np.where(mask > 0.5, 1.0, 0.0)    # Ensure Binary

        # D. AUGMENTATION & PADDING
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask.unsqueeze(0) # Return with channel dim for mask

# --- 3. AUGMENTATIONS (Data-centric AI) ---
def get_transforms(split, img_size):
    base_transform = [
        A.Resize(img_size, img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=0),
    ]
    
    if split == 'train':
        # Heavy Augmentation for Robustness
        train_transform = base_transform + [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15, p=0.5),
            # A.ElasticTransform(p=0.3, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.RandomBrightnessContrast(p=0.4),
        ]
        return A.Compose(train_transform + [ToTensorV2()])
    else:
        # Validation/Test: Only resize and normalize
        return A.Compose(base_transform + [ToTensorV2()])

# --- 4. TRAINING FUNCTION (Core Logic) ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_val_dice = 0.0
    if not torch.cuda.is_available(): 
        raise RuntimeError("❌ STOP! No NVIDIA GPU found. I refuse to use CPU.")
    device = torch.device("cuda")
    model.to(device)

    for epoch in range(num_epochs):
        # --- TRAINING ---
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            probs = torch.sigmoid(logits)
            train_dice += dice_coeff(probs > 0.5, masks)
            
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_hausdorff = 0.0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images, masks = images.to(device), masks.to(device)
                
                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5)
                val_dice += dice_coeff(preds, masks)
                val_hausdorff += compute_hausdorff_distance(preds, masks)

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_hausdorff /= len(val_loader)
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val Hausdorff: {val_hausdorff:.2f}")

        # --- SAVE BEST MODEL (Checkpointing) ---
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✅ New Best Model Saved! Dice: {best_val_dice:.4f}")

# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    DATA_DIR = "data/processed"
    
    # DataLoaders
    train_ds = MRIDataset(DATA_DIR, split='train', transform=get_transforms('train', IMG_SIZE))
    val_ds = MRIDataset(DATA_DIR, split='val', transform=get_transforms('val', IMG_SIZE))
    
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)

    # Model, Loss, Optimizer
    model = UNet(n_channels=CHANNELS, n_classes=1)
    criterion = CombinedLoss(weight_bce=0.5, weight_dice=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Start Training
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=15)
    print("🏁 Training Finished!")