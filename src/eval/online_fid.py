"""
Online FID metric for training-time evaluation.
Accumulates features in memory without saving NPZ files.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Dict, List
from tqdm import tqdm
import sys
import numpy as np

# Try to import scipy for FID calculation
try:
    import scipy.linalg
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    HAS_TORCH_FIDELITY = True
except ImportError:
    HAS_TORCH_FIDELITY = False


def _calculate_fid_from_features(real_features: torch.Tensor, fake_features: torch.Tensor) -> float:
    """
    Calculate FID from accumulated features.
    
    Args:
        real_features: [N, 2048] tensor of real image features
        fake_features: [M, 2048] tensor of fake image features
        
    Returns:
        FID score
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required for FID calculation")
    
    # Check minimum samples
    if real_features.size(0) < 2 or fake_features.size(0) < 2:
        print(f"[Warning] Insufficient samples for FID calculation: real={real_features.size(0)}, fake={fake_features.size(0)}")
        return -1.0
    
    # Convert to numpy
    real_features = real_features.cpu().numpy()
    fake_features = fake_features.cpu().numpy()
    
    # Calculate mean and covariance
    mu_real = real_features.mean(axis=0)
    mu_fake = fake_features.mean(axis=0)
    
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)
    
    # Calculate FID
    mu_real = np.asarray(mu_real, dtype=np.float64)
    mu_fake = np.asarray(mu_fake, dtype=np.float64)
    sigma_real = np.asarray(sigma_real, dtype=np.float64)
    sigma_fake = np.asarray(sigma_fake, dtype=np.float64)
    
    diff = mu_real - mu_fake
    covmean = scipy.linalg.sqrtm(sigma_real @ sigma_fake)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2.0 * covmean)
    return float(max(fid, 0.0))


class OnlineFIDMetric:
    """
    Online FID metric that accumulates features during evaluation.
    Supports distributed training.
    """
    
    def __init__(
        self,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
        feature_dim: int = 2048,
    ):
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.feature_dim = feature_dim
        
        if not HAS_TORCH_FIDELITY:
            raise ImportError("torch_fidelity is required for online FID computation")
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractorInceptionV3(
            name="inception-v3-compat", 
            features_list=['2048']
        ).to(device).eval()
        
        # Storage for features
        self.real_features: List[torch.Tensor] = []
        self.fake_features: List[torch.Tensor] = []
    
    def reset(self):
        """Reset metric state."""
        self.real_features = []
        self.fake_features = []
    
    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract InceptionV3 features from images.
        
        Args:
            images: [B, 3, H, W] in range [0, 1] or [0, 255]
            
        Returns:
            features: [B, 2048]
        """
        # Make a copy to avoid modifying original
        images = images.clone()
        
        # Convert to uint8 [0, 255] as expected by torch_fidelity
        if images.max() <= 1.5:
            images = (images * 255.0).clamp(0, 255)
        
        images = images.to(self.device)
        
        # Ensure uint8
        if images.dtype != torch.uint8:
            images = images.to(torch.uint8)
        
        # Extract features without autocast (InceptionV3 expects float32)
        with torch.cuda.amp.autocast(enabled=False):
            features = self.feature_extractor(images)[0]  # [B, 2048]
        return features.detach()
    
    @torch.no_grad()
    def update(self, real_images: torch.Tensor, fake_images: torch.Tensor):
        """
        Update metric with a batch of images.
        
        Args:
            real_images: [B, 3, H, W] in range [0, 1] or [0, 255]
            fake_images: [B, 3, H, W] in range [0, 1] or [0, 255]
        """
        real_feat = self.extract_features(real_images)
        fake_feat = self.extract_features(fake_images)
        
        self.real_features.append(real_feat.cpu())
        self.fake_features.append(fake_feat.cpu())
    
    def compute(self) -> float:
        """
        Compute FID score.
        
        Returns:
            FID score (float), synchronized across all ranks in distributed setting
        """
        # Concatenate all features on this rank
        real_feats = torch.cat(self.real_features, dim=0) if self.real_features else torch.empty(0, self.feature_dim, device=self.device)
        fake_feats = torch.cat(self.fake_features, dim=0) if self.fake_features else torch.empty(0, self.feature_dim, device=self.device)
        
        # Ensure tensors are on the correct device
        real_feats = real_feats.to(self.device)
        fake_feats = fake_feats.to(self.device)
        
        # In distributed mode, gather features from all ranks
        if self.world_size > 1:
            # First, gather the sizes from all ranks
            local_size = torch.tensor([real_feats.size(0)], device=self.device)
            sizes = [torch.empty(1, dtype=torch.long, device=self.device) for _ in range(self.world_size)]
            dist.all_gather(sizes, local_size)
            sizes = [int(s.item()) for s in sizes]
            max_size = max(sizes)
            
            # Pad tensors to max size
            real_feats_padded = torch.zeros(max_size, self.feature_dim, device=self.device)
            fake_feats_padded = torch.zeros(max_size, self.feature_dim, device=self.device)
            real_feats_padded[:real_feats.size(0)] = real_feats
            fake_feats_padded[:fake_feats.size(0)] = fake_feats
            
            # Gather padded features
            gathered_real = [torch.empty(max_size, self.feature_dim, device=self.device) for _ in range(self.world_size)]
            gathered_fake = [torch.empty(max_size, self.feature_dim, device=self.device) for _ in range(self.world_size)]
            
            dist.all_gather(gathered_real, real_feats_padded)
            dist.all_gather(gathered_fake, fake_feats_padded)
            
            # Unpad and concatenate
            all_real = []
            all_fake = []
            for i, size in enumerate(sizes):
                all_real.append(gathered_real[i][:size])
                all_fake.append(gathered_fake[i][:size])
            
            real_feats = torch.cat(all_real, dim=0).cpu()
            fake_feats = torch.cat(all_fake, dim=0).cpu()
        else:
            real_feats = real_feats.cpu()
            fake_feats = fake_feats.cpu()
        
        # Calculate FID on all ranks (to avoid blocking)
        fid_score = -1.0
        if self.rank == 0:
            if len(real_feats) == 0 or len(fake_feats) == 0:
                fid_score = -1.0
            elif len(real_feats) < 2 or len(fake_feats) < 2:
                print(f"[Warning] Not enough samples for FID: real={len(real_feats)}, fake={len(fake_feats)}")
                fid_score = -1.0
            else:
                fid_score = _calculate_fid_from_features(real_feats, fake_feats)
        
        # Synchronize to ensure all ranks wait for rank 0
        if self.world_size > 1:
            dist.barrier()
        
        return fid_score


# Flag to indicate availability
HAS_ONLINE_FID = HAS_TORCH_FIDELITY


@torch.no_grad()
def evaluate_reconstruction_online_fid(
    model,
    val_dataset,
    num_samples: int,
    batch_size: int,
    rank: int,
    world_size: int,
    device: torch.device,
    global_step: int,
    autocast_kwargs: dict,
    metrics_to_compute: Optional[List[str]] = ("psnr", "ssim", "rfid"),
) -> Optional[Dict[str, float]]:
    """
    Evaluate reconstruction metrics using online FID computation.
    No NPZ files are saved.
    
    Args:
        model: Model to evaluate
        val_dataset: Validation dataset
        num_samples: Number of samples to evaluate
        batch_size: Batch size per GPU
        rank: Current GPU rank
        world_size: Total number of GPUs
        device: Device to use
        global_step: Current training step (for logging)
        autocast_kwargs: Autocast configuration
        metrics_to_compute: List of metrics to compute
        
    Returns:
        Dictionary of metrics (only on rank 0)
    """
    from torch.utils.data import DataLoader, Subset
    
    if rank == 0:
        print(f"\n[Eval] Starting online evaluation at step {global_step}")
    
    # Initialize online FID metric
    fid_metric = None
    compute_fid = "rfid" in metrics_to_compute or "fid" in metrics_to_compute
    if compute_fid:
        fid_metric = OnlineFIDMetric(device, rank, world_size)
        fid_metric.reset()
    
    # Split dataset across ranks
    N = min(len(val_dataset), num_samples)
    chunk = N // world_size
    
    if rank < world_size - 1:
        start = rank * chunk
        end = (rank + 1) * chunk
    else:
        start = rank * chunk
        end = N
    
    rank_indices = list(range(start, end))
    subset = Subset(val_dataset, rank_indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    
    # Accumulate PSNR/SSIM stats
    all_psnr = []
    all_ssim = []
    
    iterator = tqdm(loader, desc=f"[Rank {rank}] Evaluating", file=sys.stdout) if rank == 0 else loader
    
    model.eval()
    with torch.inference_mode():
        for images, _ in iterator:
            images = images.to(device, non_blocking=True)
            
            # Forward pass with autocast if enabled
            if autocast_kwargs.get('enabled', True):
                with torch.autocast(device_type=device.type, **autocast_kwargs):
                    recon = model(images)
            else:
                recon = model(images)
            
            # Clamp to [0, 1]
            recon = recon.clamp(0, 1)
            
            # Update FID metric
            if fid_metric is not None:
                fid_metric.update(images, recon)
            
            # Calculate PSNR and SSIM on the fly
            if "psnr" in metrics_to_compute or "ssim" in metrics_to_compute:
                # Simple PSNR calculation in PyTorch
                mse = ((images - recon) ** 2).mean(dim=[1, 2, 3])
                psnr_val = (20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse + 1e-10)).mean().item()
                all_psnr.append(psnr_val)
                
                # Simple SSIM approximation (or skip for speed)
                if "ssim" in metrics_to_compute:
                    # Use a simple structural similarity or skip
                    # For now, skip detailed SSIM to save computation
                    pass
    
    model.train()
    
    # Synchronize across ranks
    if world_size > 1:
        dist.barrier()
    
    # Compile results
    metrics = {}
    
    if rank == 0:
        # Compute FID
        if fid_metric is not None:
            try:
                fid_score = fid_metric.compute()
                metrics["rfid"] = fid_score
            except Exception as e:
                print(f"[Warning] FID computation failed: {e}")
                import traceback
                traceback.print_exc()
                metrics["rfid"] = -1.0
        
        # Average PSNR
        if all_psnr:
            metrics["psnr"] = sum(all_psnr) / len(all_psnr)
        
        # TODO: Add proper SSIM computation if needed
        if "ssim" in metrics_to_compute:
            metrics["ssim"] = -1.0  # Placeholder
        
        print(f"[Eval] Step {global_step} Metrics:")
        for key, value in metrics.items():
            if value >= 0:
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: N/A")
    
    return metrics if rank == 0 else None
