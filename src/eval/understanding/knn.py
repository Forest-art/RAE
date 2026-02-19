"""
K-Nearest Neighbors evaluation for encoder representations.
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Tuple, Optional


class KNNEvaluator:
    """
    K-Nearest Neighbors evaluation.
    Uses cosine similarity or L2 distance to find nearest neighbors.
    """
    
    def __init__(
        self,
        k: int = 20,
        distance: str = "cosine",  # "cosine" or "l2"
        device: str = "cuda",
        batch_size: int = 256,
        num_workers: int = 4,
    ):
        self.k = k
        self.distance = distance
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Storage for training features
        self.train_features = None
        self.train_labels = None

    @staticmethod
    def _get_dist_rank_world() -> Tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    @staticmethod
    def _gather_local_tensors_to_rank0(
        local_features: torch.Tensor, local_labels: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        rank, world_size = KNNEvaluator._get_dist_rank_world()
        if world_size == 1:
            return local_features, local_labels

        gathered_features = [None for _ in range(world_size)] if rank == 0 else None
        gathered_labels = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(local_features, gathered_features, dst=0)
        dist.gather_object(local_labels, gathered_labels, dst=0)

        if rank != 0:
            return None, None

        feat_chunks = []
        label_chunks = []
        for feat_shard, label_shard in zip(gathered_features, gathered_labels):
            if not isinstance(feat_shard, torch.Tensor) or not isinstance(label_shard, torch.Tensor):
                continue
            if feat_shard.numel() == 0 or label_shard.numel() == 0:
                continue
            feat_chunks.append(feat_shard)
            label_chunks.append(label_shard)

        if not feat_chunks:
            feat_dim = local_features.shape[1]
            return (
                torch.empty((0, feat_dim), dtype=local_features.dtype),
                torch.empty((0,), dtype=torch.long),
            )
        return torch.cat(feat_chunks, dim=0), torch.cat(label_chunks, dim=0)
        
    @torch.no_grad()
    def extract_features(
        self,
        model,
        dataloader: DataLoader,
        device: str = "cuda",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features from encoder.
        
        Args:
            model: Encoder model
            dataloader: Data loader
            device: Device
            
        Returns:
            features: [N, encoder_dim]
            labels: [N]
        """
        model.eval()
        rank, _ = self._get_dist_rank_world()
        local_features = []
        local_labels = []
        
        iterator = tqdm(dataloader, desc="Extracting features", leave=False, disable=(rank != 0))
        for images, labels in iterator:
            images = images.to(device)
            
            # Extract latent representation
            if hasattr(model, 'forward_student'):
                # SRAE style: use student encoder without masking
                z, _, _, _ = model.forward_student(images, mask_ratio=0.0)
                # Pool features: mean over patches
                z = z.mean(dim=1)  # [B, bottleneck_dim]
            elif hasattr(model, 'encode'):
                # RAE style: encode returns latent
                z = model.encode(images)
            else:
                # Generic: try to get encoder output
                z = model(images)
                if isinstance(z, dict):
                    z = z.get('latent', z.get('z', z))
            
            # Flatten if needed
            if z.dim() > 2:
                z = z.flatten(1)
            
            # Normalize for cosine similarity
            if self.distance == "cosine":
                z = F.normalize(z, p=2, dim=1)

            local_features.append(z.detach().cpu())
            local_labels.append(labels.detach().to(dtype=torch.long).cpu())

        if local_features:
            local_features_tensor = torch.cat(local_features, dim=0)
            local_labels_tensor = torch.cat(local_labels, dim=0)
            feature_dim = local_features_tensor.shape[1]
        else:
            feature_dim = self.train_features.shape[1] if self.train_features is not None else 1
            local_features_tensor = torch.empty((0, feature_dim), dtype=torch.float32)
            local_labels_tensor = torch.empty((0,), dtype=torch.long)

        features, labels = self._gather_local_tensors_to_rank0(local_features_tensor, local_labels_tensor)

        if rank != 0:
            return (
                torch.empty((0, feature_dim), dtype=torch.float32),
                torch.empty((0,), dtype=torch.long),
            )

        if features is None or labels is None or features.numel() == 0:
            return (
                torch.empty((0, feature_dim), dtype=torch.float32),
                torch.empty((0,), dtype=torch.long),
            )
        return features, labels
    
    def fit(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
    ):
        """
        Store training features and labels.
        
        Args:
            train_features: [N_train, encoder_dim]
            train_labels: [N_train]
        """
        self.train_features = train_features.to(self.device)
        self.train_labels = train_labels.to(self.device)
        print(f"Stored {len(self.train_features)} training features for KNN")
    
    @torch.no_grad()
    def evaluate(
        self,
        test_features: torch.Tensor,
        test_labels: torch.Tensor,
        k: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate using KNN.
        
        Args:
            test_features: [N_test, encoder_dim]
            test_labels: [N_test]
            k: Number of neighbors (default: self.k)
            
        Returns:
            metrics: dict with top1 and top5 accuracy
        """
        if self.train_features is None:
            raise ValueError("Must call fit() before evaluate()")
        
        k = k or self.k
        test_features = test_features.to(self.device)
        test_labels = test_labels.to(self.device)
        
        # Normalize test features if using cosine distance
        if self.distance == "cosine":
            test_features = F.normalize(test_features, p=2, dim=1)
        
        num_test = len(test_features)
        top1_correct = 0
        top5_correct = 0
        
        # Process in batches to avoid OOM
        batch_size = 256
        for i in tqdm(range(0, num_test, batch_size), desc="KNN evaluation"):
            batch_features = test_features[i:i+batch_size]
            batch_labels = test_labels[i:i+batch_size]
            
            # Compute distances
            if self.distance == "cosine":
                # Cosine similarity (already normalized)
                similarities = batch_features @ self.train_features.T  # [B, N_train]
                distances = -similarities  # Negative similarity as distance
            else:
                # L2 distance
                distances = torch.cdist(batch_features, self.train_features)  # [B, N_train]
            
            # Find k nearest neighbors
            knn_indices = distances.topk(k, largest=False, dim=1).indices  # [B, k]
            knn_labels = self.train_labels[knn_indices]  # [B, k]
            
            # Top-1: majority vote
            batch_top1 = knn_labels[:, 0]
            top1_correct += (batch_top1 == batch_labels).sum().item()
            
            # Top-5: check if true label is in top 5
            if k >= 5:
                top5_matches = (knn_labels[:, :5] == batch_labels.unsqueeze(1)).any(dim=1)
                top5_correct += top5_matches.sum().item()
        
        top1_acc = top1_correct / num_test * 100
        top5_acc = top5_correct / num_test * 100 if k >= 5 else 0.0
        
        return {
            'top1': top1_acc,
            'top5': top5_acc,
            'k': k,
        }
    
    def run_full_evaluation(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        k_list: list = [1, 5, 10, 20, 100, 200],
    ) -> Dict[int, Dict[str, float]]:
        """
        Run complete KNN evaluation with multiple k values.
        
        Args:
            model: Encoder model
            train_loader: Training data loader
            val_loader: Validation data loader
            k_list: List of k values to try
            
        Returns:
            results: dict mapping k to metrics
        """
        rank, _ = self._get_dist_rank_world()
        if rank == 0:
            print("Extracting training features...")
        train_features, train_labels = self.extract_features(model, train_loader)

        if rank == 0:
            self.fit(train_features, train_labels)
            print("Extracting validation features...")
        val_features, val_labels = self.extract_features(model, val_loader)

        if rank != 0:
            return {}
        
        results = {}
        for k in k_list:
            if k > len(train_features):
                print(f"Skipping k={k} (larger than training set)")
                continue
            
            print(f"\nEvaluating with k={k}...")
            metrics = self.evaluate(val_features, val_labels, k=k)
            results[k] = metrics
            print(f"  Top-1: {metrics['top1']:.2f}%, Top-5: {metrics['top5']:.2f}%")
        
        return results
