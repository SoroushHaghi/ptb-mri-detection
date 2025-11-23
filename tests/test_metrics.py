import torch
import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.metrics import dice_coeff, DiceLoss

def test_dice_perfect_match():
    """
    Math Check: Dice of identical masks must be 1.0
    """
    mask1 = torch.ones(1, 256, 256)
    mask2 = torch.ones(1, 256, 256)
    
    score = dice_coeff(mask1, mask2)
    assert score > 0.99, f"❌ Math Error: Perfect match should be ~1.0, got {score}"

def test_dice_no_overlap():
    """
    Math Check: Dice of opposite masks must be ~0.0
    """
    mask1 = torch.ones(1, 256, 256)
    mask2 = torch.zeros(1, 256, 256)
    
    score = dice_coeff(mask1, mask2)
    assert score < 0.01, f"❌ Math Error: No overlap should be ~0.0, got {score}"
    print("\n✅ Dice Metric mathematics verified.")

def test_dice_loss_perfect_match():
    """
    Math Check: DiceLoss of identical masks must be ~0.0
    """
    loss_fn = DiceLoss()
    mask1 = torch.ones(1, 256, 256)
    mask2 = torch.ones(1, 256, 256)
    
    loss = loss_fn(mask1, mask2)
    assert loss < 0.01, f"❌ Math Error: Perfect match should yield ~0.0 loss, got {loss}"

def test_dice_loss_no_overlap():
    """
    Math Check: DiceLoss of opposite masks must be ~1.0
    """
    loss_fn = DiceLoss()
    mask1 = torch.ones(1, 256, 256)
    mask2 = torch.zeros(1, 256, 256)
    
    loss = loss_fn(mask1, mask2)
    assert loss > 0.99, f"❌ Math Error: No overlap should yield ~1.0 loss, got {loss}"

def test_dice_loss_all_zeros():
    """
    Math Check: DiceLoss when both prediction and target are all zeros.
    Should be ~0.0 (perfect match for emptiness)
    """
    loss_fn = DiceLoss()
    pred = torch.zeros(1, 256, 256)
    target = torch.zeros(1, 256, 256)
    
    loss = loss_fn(pred, target)
    assert loss < 0.01, f"❌ Math Error: All zeros should yield ~0.0 loss, got {loss}"

def test_dice_coeff_all_zeros():
    """
    Math Check: dice_coeff when both prediction and target are all zeros.
    Should be 1.0 (perfect match for emptiness)
    """
    pred = torch.zeros(1, 256, 256)
    target = torch.zeros(1, 256, 256)
    
    score = dice_coeff(pred, target)
    assert score > 0.99, f"❌ Math Error: All zeros should yield ~1.0 dice coeff, got {score}"
    print("✅ DiceLoss edge cases and all zeros scenario verified.")

