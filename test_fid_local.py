#!/usr/bin/env python3
"""Test FID computation locally using pure torch."""

import torch
import torch.nn as nn
from torchvision import models
import time


def torch_cov(matrix, rowvar=False):
    """Estimate covariance matrix using torch."""
    if rowvar:
        matrix = matrix.T
    
    mean = matrix.mean(dim=0, keepdim=True)
    centered = matrix - mean
    n = matrix.size(0)
    return (centered.T @ centered) / (n - 1)


def torch_sqrtm(matrix, num_iters=10):
    """Compute matrix square root using Newton's method."""
    # Initial guess
    n = matrix.size(0)
    I = torch.eye(n, device=matrix.device, dtype=matrix.dtype)
    Y = matrix
    Z = I
    
    for _ in range(num_iters):
        Y_next = 0.5 * (Y + Z.inverse())
        Z_next = 0.5 * (Z + Y.inverse())
        Y = Y_next
        Z = Z_next
    
    return Y


class SimpleFID:
    """Simple FID using torchvision InceptionV3 and pure torch."""
    
    def __init__(self, device='cuda'):
        self.device = device
        # Load InceptionV3
        inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        inception.fc = nn.Identity()
        inception.eval().to(device)
        self.inception = inception
        
        self.real_features = []
        self.fake_features = []
    
    def update(self, images, real=True):
        """Update with batch of images [B, 3, H, W] in [0, 1]."""
        images = images.to(self.device)
        
        # Resize to 299x299
        if images.shape[2] != 299 or images.shape[3] != 299:
            images = torch.nn.functional.interpolate(
                images, size=(299, 299), mode='bilinear', align_corners=False
            )
        
        # Normalize to [-1, 1]
        images = images * 2 - 1
        
        with torch.no_grad():
            feats = self.inception(images)
        
        if real:
            self.real_features.append(feats)
        else:
            self.fake_features.append(feats)
    
    def compute(self):
        """Compute FID."""
        real_feats = torch.cat(self.real_features, dim=0)
        fake_feats = torch.cat(self.fake_features, dim=0)
        
        mu_real = real_feats.mean(dim=0)
        mu_fake = fake_feats.mean(dim=0)
        
        # Use numpy for covariance and sqrtm (should work now with basic operations)
        import numpy as np
        real_np = real_feats.cpu().numpy()
        fake_np = fake_feats.cpu().numpy()
        
        # Compute covariance manually
        def cov_np(x):
            mean = x.mean(axis=0, keepdims=True)
            return ((x - mean).T @ (x - mean)) / (x.shape[0] - 1)
        
        sigma_real = cov_np(real_np)
        sigma_fake = cov_np(fake_np)
        
        # Use scipy for sqrtm
        import scipy.linalg
        covmean = scipy.linalg.sqrtm(sigma_real @ sigma_fake)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        diff = mu_real.cpu().numpy() - mu_fake.cpu().numpy()
        fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2.0 * covmean)
        return torch.tensor(max(fid, 0.0))


def test_fid():
    print("="*60)
    print("Testing FID with torchvision InceptionV3")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Initialize
    print("\nInitializing FID...")
    start = time.time()
    fid = SimpleFID(device=device)
    print(f"Initialized in {time.time() - start:.2f}s")
    
    # Process batches
    batch_size = 8
    num_batches = 10
    print(f"\nProcessing {num_batches} batches of {batch_size} images...")
    
    for i in range(num_batches):
        real = torch.rand(batch_size, 3, 256, 256)
        fake = torch.rand(batch_size, 3, 256, 256)
        fid.update(real, real=True)
        fid.update(fake, real=False)
        print(f"  Batch {i+1}/{num_batches} done")
    
    # Compute FID
    print("\nComputing FID...")
    start = time.time()
    result = fid.compute()
    print(f"FID computed in {time.time() - start:.2f}s")
    print(f"Result: {result.item():.4f}")
    
    print("\n✓ Test passed!")


if __name__ == "__main__":
    test_fid()
