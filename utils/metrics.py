
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import directed_hausdorff

# --- 1. METRICS (Scientific Evaluation) ---

def dice_coeff(pred, target, smooth=1e-5):
    """
    Calculates the Dice Coefficient (Overlap Metric).
    Formula: (2 * Intersection) / (Area_Pred + Area_Target)
    """
    # Flatten the tensors to 1D arrays
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    pred_sum = pred.sum()
    target_sum = target.sum()

    # Handle cases where both pred and target are all zeros
    if pred_sum == 0 and target_sum == 0:
        return 1.0 # Perfect match if both are empty

    dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
    
    return dice.item()

def compute_hausdorff_distance(pred_mask, target_mask):
    """
    Calculates the Hausdorff Distance (Safety Metric).
    Measures the maximum distance between the prediction and ground truth contours.
    """
    # Move to CPU and convert to numpy for Scipy
    if torch.is_tensor(pred_mask):
        pred_mask = pred_mask.cpu().detach().numpy()
    if torch.is_tensor(target_mask):
        target_mask = target_mask.cpu().detach().numpy()

    # Ensure binary (0 or 1)
    pred_mask = (pred_mask > 0.5).astype(bool)
    target_mask = (target_mask > 0.5).astype(bool)

    hausdorff_distances = []
    
    # Handle batch processing for Hausdorff
    if pred_mask.ndim == 4: # Assume BCHW format
        for i in range(pred_mask.shape[0]):
            single_pred = pred_mask[i, 0, :, :] # Squeeze batch and channel
            single_target = target_mask[i, 0, :, :] # Squeeze batch and channel

            if not np.any(single_pred) or not np.any(single_target):
                if not np.any(single_pred) and not np.any(single_target):
                    hausdorff_distances.append(0.0) # Both empty = Perfect match
                else:
                    hausdorff_distances.append(100.0) # One empty = Worst case penalty
            else:
                d1 = directed_hausdorff(single_pred, single_target)[0]
                d2 = directed_hausdorff(single_target, single_pred)[0]
                hausdorff_distances.append(max(d1, d2))
        return np.mean(hausdorff_distances)
    
    elif pred_mask.ndim == 2: # Single image, already 2D
        if not np.any(pred_mask) or not np.any(target_mask):
            if not np.any(pred_mask) and not np.any(target_mask):
                return 0.0 # Both empty = Perfect match
            return 100.0 # One empty = Worst case penalty (e.g., 100 pixels error)
        
        d1 = directed_hausdorff(pred_mask, target_mask)[0]
        d2 = directed_hausdorff(target_mask, pred_mask)[0]
        return max(d1, d2)
    else:
        raise ValueError(f"Unsupported number of dimensions for input masks: {pred_mask.ndim}")

# --- 2. LOSS FUNCTIONS (Training Logic) ---

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice # We want to minimize error, so 1 - Dice

class CombinedLoss(nn.Module):
    """
    The 'Combo Loss' strategy for stability.
    Loss = BCE (Binary Cross Entropy) + Dice Loss
    """
    def __init__(self, weight_dice=0.5, weight_bce=0.5):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss() # Stable BCE
        self.dice = DiceLoss()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce

    def forward(self, pred_logits, target):
        # BCE takes logits (before sigmoid), Dice takes probabilities (after sigmoid)
        pred_probs = torch.sigmoid(pred_logits)
        
        loss_bce = self.bce(pred_logits, target)
        loss_dice = self.dice(pred_probs, target)
        
        return (self.weight_bce * loss_bce) + (self.weight_dice * loss_dice)
