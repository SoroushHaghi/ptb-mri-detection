import torch
import pytest
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.model import UNet

def test_unet_output_shape():
    """
    Sanity Check: Does the model preserve spatial dimensions?
    Input: (Batch, 1, 256, 256) -> Output: (Batch, 1, 256, 256)
    """
    model = UNet(n_channels=1, n_classes=1)
    # Create a fake MRI image (Random tensor)
    dummy_input = torch.randn(1, 1, 256, 256)
    
    output = model(dummy_input)
    
    assert output.shape == (1, 1, 256, 256), f"❌ Shape Mismatch! Expected (1,1,256,256) but got {output.shape}"
    print("\n✅ U-Net preserves spatial dimensions correctly.")
