"""
Test SRAE model without external dependencies.
Creates synthetic tensors and runs a forward pass.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from stage1 import SRAE


def test_srae_forward():
    """Test SRAE forward pass with dummy data."""
    
    print("=" * 60)
    print("Testing SRAE Model")
    print("=" * 60)
    
    # Create model
    print("\n1. Creating SRAE model...")
    model = SRAE(
        dinov2_model_name="facebook/dinov2-base",
        img_size=224,
        patch_size=14,
        mask_ratio=0.75,
        bottleneck_dim=768,
        use_l2_norm=True,
        projector_hidden_dim=4096,
        decoder_num_layers=4,
        decoder_num_heads=8,
        decoder_dim=256,
        loss_rec_weight=1.0,
        loss_align_weight=0.1,
        loss_reg_weight=0.01,
        teacher_output_type="cls",
    )
    print("   Model created successfully!")
    
    # Create dummy input
    print("\n2. Creating dummy input (B=2, C=3, H=224, W=224)...")
    dummy_images = torch.randn(2, 3, 224, 224).clamp(0, 1)
    print(f"   Input shape: {dummy_images.shape}")
    print(f"   Input range: [{dummy_images.min():.3f}, {dummy_images.max():.3f}]")
    
    # Forward pass
    print("\n3. Running forward pass...")
    model.eval()
    with torch.no_grad():
        try:
            recon = model(dummy_images)
            print(f"   Output shape: {recon.shape}")
            print(f"   Output range: [{recon.min():.3f}, {recon.max():.3f}]")
            
            # Get losses
            losses = model.get_last_losses()
            print("\n4. Losses computed:")
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    print(f"   {key}: {value.item():.6f}")
                else:
                    print(f"   {key}: {value}")
            
            print("\n" + "=" * 60)
            print("✓ SRAE forward pass successful!")
            print("=" * 60)
            return True
            
        except Exception as e:
            print(f"\n✗ Error during forward pass:")
            import traceback
            traceback.print_exc()
            return False


def test_srae_encode_decode():
    """Test encode/decode methods."""
    
    print("\n" + "=" * 60)
    print("Testing SRAE Encode/Decode")
    print("=" * 60)
    
    model = SRAE(
        dinov2_model_name="facebook/dinov2-base",
        img_size=224,
        patch_size=14,
        mask_ratio=0.0,  # No masking for encode
        bottleneck_dim=768,
        use_l2_norm=True,
        projector_hidden_dim=4096,
        decoder_num_layers=4,
        decoder_num_heads=8,
        decoder_dim=256,
    )
    model.eval()
    
    print("\n1. Testing encode...")
    dummy_images = torch.randn(1, 3, 224, 224).clamp(0, 1)
    with torch.no_grad():
        try:
            z = model.encode(dummy_images)
            print(f"   Latent shape: {z.shape}")
            
            print("\n2. Testing decode...")
            recon = model.decode(z)
            print(f"   Reconstructed shape: {recon.shape}")
            print(f"   Reconstructed range: [{recon.min():.3f}, {recon.max():.3f}]")
            
            print("\n" + "=" * 60)
            print("✓ Encode/Decode successful!")
            print("=" * 60)
            return True
            
        except Exception as e:
            print(f"\n✗ Error:")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# SRAE Model Test Suite")
    print("#" * 60 + "\n")
    
    # Note: This test requires downloading DINOv2 weights
    print("Note: This will download DINOv2-base weights (~300MB) on first run.\n")
    
    success1 = test_srae_forward()
    success2 = test_srae_encode_decode()
    
    print("\n" + "#" * 60)
    if success1 and success2:
        print("# All tests passed!")
    else:
        print("# Some tests failed.")
    print("#" * 60 + "\n")
