import torch
import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.metrics import dice_coeff

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
