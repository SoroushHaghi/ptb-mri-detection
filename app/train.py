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

# Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.model import UNet
from utils.metrics import CombinedLoss, dice_coeff, compute_hausdorff_distance
from config import seed_everything, SEED, IMG_SIZE, CHANNELS

# --- 1. REPRODUCIBILITY ---
seed_everything(SEED)

# --- 2. DATASET CLASS ---
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
        
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Min-Max Scaling
        if image.max() > 0:
            image = (image - image.min()) / (image.max() - image.min())
        else:
            image = image / 1.0
            
        image = image.astype(np.float32)
        mask = (mask / 255.0).astype(np.float32)
        mask = np.where(mask > 0.5, 1.0, 0.0)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        return image, mask.unsqueeze(0)

# --- 3. AUGMENTATIONS (Full Power) ---
def get_transforms(split, img_size):
    base_transform = [
        A.Resize(img_size, img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=0),
    ]
    
    if split == 'train':
        train_transform = base_transform + [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # Re-enabled heavy augmentations for better generalization
            A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.ElasticTransform(p=0.3, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.RandomBrightnessContrast(p=0.4),
            A.GaussNoise(var_limit=(0.001, 0.01), p=0.3),
        ]
        return A.Compose(train_transform + [ToTensorV2()])
    else:
        return A.Compose(base_transform + [ToTensorV2()])

# --- 4. TRAINING FUNCTION (Turbo Mode) ---
def train():
    # BOOSTED HYPERPARAMETERS
    BATCH_SIZE = 4  # Slightly increased from 2 (Try 4, if VRAM error, go back to 2)
    LR = 1e-4
    EPOCHS = 50     # Increased from 15 to 50 for max accuracy
    
    # Strict GPU Check
    if not torch.cuda.is_available():
        raise RuntimeError("❌ STOP! No NVIDIA GPU found. Training requires GPU.")
    DEVICE = "cuda"
    
    print(f"🚀 Starting High-Accuracy Training on {DEVICE.upper()}...")
    
    DATA_DIR = "data/processed"
    train_ds = MRIDataset(DATA_DIR, split='train', transform=get_transforms('train', IMG_SIZE))
    val_ds = MRIDataset(DATA_DIR, split='val', transform=get_transforms('val', IMG_SIZE))
    
    # Optimized DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    model = UNet(n_channels=CHANNELS, n_classes=1).to(DEVICE)
    criterion = CombinedLoss(weight_bce=0.4, weight_dice=0.6) # Focus more on Dice
    
    # Better Optimizer (AdamW)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    
    # Scheduler (Smart Learning Rate)
    # Reduce LR if validation loss doesn't improve for 3 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_dice = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for img, mask in loop:
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, mask)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            probs = torch.sigmoid(outputs)
            train_dice += dice_coeff(probs > 0.5, mask)
            
            loop.set_postfix(loss=loss.item())
            
        # Validation
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_hd95 = 0.0
        
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(DEVICE), mask.to(DEVICE)
                outputs = model(img)
                loss = criterion(outputs, mask)
                val_loss += loss.item()
                
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_dice += dice_coeff(preds, mask)
                
                # Hausdorff (Batch avg)
                batch_hd = 0
                for i in range(img.shape[0]):
                    batch_hd += compute_hausdorff_distance(preds[i,0], mask[i,0])
                val_hd95 += batch_hd / img.shape[0]

        # Averages
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_hd95 /= len(val_loader)
        
        # Step Scheduler
        scheduler.step(val_loss)
        
        print(f"   📉 Val Loss: {val_loss:.4f} | 🎲 Val Dice: {val_dice:.4f} | 🛡️ HD95: {val_hd95:.2f}")
        
        # Save Best
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), "model_final.pth")
            print(f"   ✅ Best Model Saved! ({best_val_dice:.4f})")

    print("🏁 High-Accuracy Training Finished!")

if __name__ == "__main__":
    train()