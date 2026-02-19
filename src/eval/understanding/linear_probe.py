"""
Linear Probing evaluation for encoder representations.
Trains a linear classifier on frozen encoder features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from typing import Dict, Optional, Tuple


class LinearProbeEvaluator:
    """
    Linear probing evaluation.
    Fits a linear classifier on top of frozen encoder representations.
    """
    
    def __init__(
        self,
        encoder_dim: int,
        num_classes: int,
        device: str = "cuda",
        epochs: int = 100,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        batch_size: int = 256,
        num_workers: int = 4,
        feature_pool: str = "avg",
    ):
        self.encoder_dim = encoder_dim
        self.num_classes = num_classes
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.feature_pool = feature_pool
        
        # Linear classifier
        self.classifier = nn.Linear(encoder_dim, num_classes).to(device)

    def _pool_features(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 4:
            if self.feature_pool == "flatten":
                return z.flatten(1)
            return z.mean(dim=(-2, -1))
        if z.dim() == 3:
            if self.feature_pool == "flatten":
                return z.flatten(1)
            if self.feature_pool == "cls":
                return z[:, 0]
            return z.mean(dim=1)
        if z.dim() > 2:
            return z.flatten(1)
        return z

    @staticmethod
    def _get_dist_rank_world() -> Tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    @staticmethod
    def _gather_local_tensors_to_rank0(
        local_features: torch.Tensor, local_labels: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        rank, world_size = LinearProbeEvaluator._get_dist_rank_world()
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
        
    def reset(self):
        """Reset classifier weights."""
        self.classifier = nn.Linear(self.encoder_dim, self.num_classes).to(self.device)
        
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
            model: Encoder model with encode() method
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
            
            z = self._pool_features(z)

            local_features.append(z.detach().cpu())
            local_labels.append(labels.detach().to(dtype=torch.long).cpu())

        if local_features:
            local_features_tensor = torch.cat(local_features, dim=0)
            local_labels_tensor = torch.cat(local_labels, dim=0)
        else:
            local_features_tensor = torch.empty((0, self.encoder_dim), dtype=torch.float32)
            local_labels_tensor = torch.empty((0,), dtype=torch.long)

        features, labels = self._gather_local_tensors_to_rank0(local_features_tensor, local_labels_tensor)

        if rank != 0:
            return (
                torch.empty((0, self.encoder_dim), dtype=torch.float32),
                torch.empty((0,), dtype=torch.long),
            )

        if features is None or labels is None or features.numel() == 0:
            return (
                torch.empty((0, self.encoder_dim), dtype=torch.float32),
                torch.empty((0,), dtype=torch.long),
            )
        return features, labels
    
    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        val_features: Optional[torch.Tensor] = None,
        val_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, list]:
        """
        Train linear classifier on features.
        
        Args:
            features: [N, encoder_dim] training features
            labels: [N] training labels
            val_features: [N_val, encoder_dim] validation features
            val_labels: [N_val] validation labels
            
        Returns:
            history: dict with train_acc and val_acc lists
        """
        self.reset()
        self.classifier.train()
        
        optimizer = torch.optim.SGD(
            self.classifier.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            momentum=0.9,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        history = {
            'train_acc': [],
            'val_acc': [],
        }

        train_dataset = TensorDataset(features, labels)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

        val_loader = None
        if val_features is not None and val_labels is not None:
            val_dataset = TensorDataset(val_features, val_labels)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                drop_last=False,
            )
        
        for epoch in range(self.epochs):
            # Train
            self.classifier.train()
            train_correct = 0
            train_total = 0
            for feat_batch, label_batch in train_loader:
                feat_batch = feat_batch.to(self.device, non_blocking=True)
                label_batch = label_batch.to(self.device, non_blocking=True)

                logits = self.classifier(feat_batch)
                loss = F.cross_entropy(logits, label_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                preds = logits.argmax(dim=1)
                train_correct += (preds == label_batch).sum().item()
                train_total += label_batch.numel()
            scheduler.step()
            
            # Evaluate
            with torch.no_grad():
                self.classifier.eval()
                train_acc = float(train_correct) / max(1, train_total)
                history['train_acc'].append(train_acc)
                
                if val_loader is not None:
                    val_correct = 0
                    val_total = 0
                    for feat_batch, label_batch in val_loader:
                        feat_batch = feat_batch.to(self.device, non_blocking=True)
                        label_batch = label_batch.to(self.device, non_blocking=True)
                        val_logits = self.classifier(feat_batch)
                        val_preds = val_logits.argmax(dim=1)
                        val_correct += (val_preds == label_batch).sum().item()
                        val_total += label_batch.numel()
                    val_acc = float(val_correct) / max(1, val_total)
                    history['val_acc'].append(val_acc)
                else:
                    val_acc = None
            
            if (epoch + 1) % 20 == 0 or epoch == 0:
                if val_acc is not None:
                    print(f"Epoch {epoch+1}/{self.epochs}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{self.epochs}: Train Acc={train_acc:.4f}")
        
        return history
    
    @torch.no_grad()
    def evaluate(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Evaluate classifier on features.
        
        Args:
            features: [N, encoder_dim]
            labels: [N]
            
        Returns:
            metrics: dict with top1 and top5 accuracy
        """
        self.classifier.eval()
        dataset = TensorDataset(features, labels)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

        top1_correct = 0
        top5_correct = 0
        total = 0
        for feat_batch, label_batch in loader:
            feat_batch = feat_batch.to(self.device, non_blocking=True)
            label_batch = label_batch.to(self.device, non_blocking=True)
            logits = self.classifier(feat_batch)

            preds = logits.argmax(dim=1)
            top1_correct += (preds == label_batch).sum().item()

            top5_preds = logits.topk(min(5, logits.shape[1]), dim=1).indices
            top5_correct += (top5_preds == label_batch.unsqueeze(1)).any(dim=1).sum().item()
            total += label_batch.numel()

        top1_acc = float(top1_correct) / max(1, total)
        top5_acc = float(top5_correct) / max(1, total)
        
        return {
            'top1': top1_acc * 100,
            'top5': top5_acc * 100,
        }
    
    def run_full_evaluation(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Run complete linear probing evaluation.
        
        Args:
            model: Encoder model
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            results: dict with final metrics
        """
        rank, _ = self._get_dist_rank_world()
        if rank == 0:
            print("Extracting training features...")
        train_features, train_labels = self.extract_features(model, train_loader)
        
        if rank == 0:
            print("Extracting validation features...")
        val_features, val_labels = self.extract_features(model, val_loader)

        if rank != 0:
            return {}

        print(f"Training linear classifier for {self.epochs} epochs...")
        history = self.fit(train_features, train_labels, val_features, val_labels)
        
        print("Evaluating on validation set...")
        results = self.evaluate(val_features, val_labels)
        
        results['train_acc'] = history['train_acc'][-1] * 100
        results['best_val_acc'] = max(history['val_acc']) * 100 if history['val_acc'] else results['top1']
        
        return results
