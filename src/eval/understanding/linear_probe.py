"""
Linear Probing evaluation for encoder representations.
Trains a linear classifier on frozen encoder features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional, Tuple
import numpy as np


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
    ):
        self.encoder_dim = encoder_dim
        self.num_classes = num_classes
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Linear classifier
        self.classifier = nn.Linear(encoder_dim, num_classes).to(device)
        
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
        all_features = []
        all_labels = []
        
        for images, labels in tqdm(dataloader, desc="Extracting features", leave=False):
            images = images.to(device)
            
            # Extract latent representation
            if hasattr(model, 'encode'):
                # RAE style: encode returns latent
                z = model.encode(images)
            elif hasattr(model, 'forward_student'):
                # SRAE style: use student encoder without masking
                z, _, _, _ = model.forward_student(images, mask_ratio=0.0)
                # Pool features: mean over patches
                z = z.mean(dim=1)  # [B, bottleneck_dim]
            else:
                # Generic: try to get encoder output
                z = model(images)
                if isinstance(z, dict):
                    z = z.get('latent', z.get('z', z))
            
            # Flatten if needed
            if z.dim() > 2:
                z = z.flatten(1)
            
            all_features.append(z.cpu())
            all_labels.append(labels)
        
        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
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
        
        features = features.to(self.device)
        labels = labels.to(self.device)
        
        history = {
            'train_acc': [],
            'val_acc': [],
        }
        
        for epoch in range(self.epochs):
            # Train
            self.classifier.train()
            logits = self.classifier(features)
            loss = F.cross_entropy(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Evaluate
            with torch.no_grad():
                self.classifier.eval()
                train_preds = self.classifier(features).argmax(dim=1)
                train_acc = (train_preds == labels).float().mean().item()
                history['train_acc'].append(train_acc)
                
                if val_features is not None and val_labels is not None:
                    val_logits = self.classifier(val_features.to(self.device))
                    val_preds = val_logits.argmax(dim=1)
                    val_acc = (val_preds == val_labels.to(self.device)).float().mean().item()
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
        features = features.to(self.device)
        labels = labels.to(self.device)
        
        logits = self.classifier(features)
        
        # Top-1 accuracy
        preds = logits.argmax(dim=1)
        top1_acc = (preds == labels).float().mean().item()
        
        # Top-5 accuracy
        top5_preds = logits.topk(5, dim=1).indices
        top5_acc = (top5_preds == labels.unsqueeze(1)).any(dim=1).float().mean().item()
        
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
        print("Extracting training features...")
        train_features, train_labels = self.extract_features(model, train_loader)
        
        print("Extracting validation features...")
        val_features, val_labels = self.extract_features(model, val_loader)
        
        print(f"Training linear classifier for {self.epochs} epochs...")
        history = self.fit(train_features, train_labels, val_features, val_labels)
        
        print("Evaluating on validation set...")
        results = self.evaluate(val_features, val_labels)
        
        results['train_acc'] = history['train_acc'][-1] * 100
        results['best_val_acc'] = max(history['val_acc']) * 100 if history['val_acc'] else results['top1']
        
        return results
