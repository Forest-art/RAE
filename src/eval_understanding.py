"""
Understanding evaluation script for encoder representations.
Evaluates linear probing and KNN classification performance.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from omegaconf import OmegaConf
import json

from dataset import HFImageNetDataset, load_dataset_from_hf
from stage1 import RAE, SRAE
from eval.understanding import LinearProbeEvaluator, KNNEvaluator
from utils.model_utils import instantiate_from_config
from utils.dist_utils import setup_distributed, cleanup_distributed
from utils.train_utils import center_crop_arr


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate encoder understanding")
    parser.add_argument("--config", type=str, required=True, help="Model config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--data-path", type=str, required=True, help="Validation data path")
    parser.add_argument("--train-data-path", type=str, default=None, help="Training data path (for KNN/probing)")
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of classes")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="Num workers")
    parser.add_argument("--image-size", type=int, default=224, help="Image size")
    parser.add_argument("--use-hf", action="store_true", help="Use HuggingFace dataset")
    parser.add_argument("--hf-split", type=str, default="validation", help="HF split")
    parser.add_argument("--output", type=str, default="understanding_results.json", help="Output file")
    
    # Evaluation specific
    parser.add_argument("--eval-knn", action="store_true", help="Run KNN evaluation")
    parser.add_argument("--eval-linear", action="store_true", help="Run linear probing")
    parser.add_argument("--knn-k", type=int, nargs="+", default=[1, 5, 10, 20, 100, 200], help="K values for KNN")
    parser.add_argument("--linear-epochs", type=int, default=100, help="Linear probe epochs")
    parser.add_argument("--linear-lr", type=float, default=0.001, help="Linear probe learning rate")
    
    args = parser.parse_args()
    return args


@torch.no_grad()
def get_encoder_dim(model, device):
    """Infer encoder output dimension."""
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    if hasattr(model, 'encode'):
        z = model.encode(dummy_input)
    elif hasattr(model, 'forward_student'):
        z, _, _, _ = model.forward_student(dummy_input, mask_ratio=0.0)
        z = z.mean(dim=1)
    else:
        z = model(dummy_input)
        if isinstance(z, dict):
            z = z.get('latent', z.get('z', z))
    
    if z.dim() > 2:
        z = z.flatten(1)
    
    return z.shape[1]


def main():
    args = parse_args()
    
    # Setup
    rank, world_size, device = setup_distributed()
    
    if rank == 0:
        print(f"Loading config from {args.config}")
    
    # Load config
    cfg = OmegaConf.load(args.config)
    model_config = cfg.get("stage_1")
    
    # Load model
    if rank == 0:
        print(f"Loading model from {args.checkpoint}")
    
    model = instantiate_from_config(model_config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'ema' in checkpoint:
        state_dict = checkpoint['ema']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    if rank == 0:
        print("Model loaded successfully")
    
    # Get encoder dimension
    encoder_dim = get_encoder_dim(model, device)
    if rank == 0:
        print(f"Encoder dimension: {encoder_dim}")
    
    # Data transforms
    eval_transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
    ])
    
    # Load validation data
    if args.use_hf:
        val_dataset = load_dataset_from_hf(
            load_from_disk_path=args.data_path,
            split=args.hf_split
        )
        val_dataset = HFImageNetDataset(val_dataset, transform=eval_transform)
    else:
        val_dataset = ImageFolder(args.data_path, transform=eval_transform)
    
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    # Load training data (for KNN and linear probing)
    train_loader = None
    if args.train_data_path is not None:
        if args.use_hf:
            train_dataset = load_dataset_from_hf(
                load_from_disk_path=args.train_data_path,
                split="train"
            )
            train_dataset = HFImageNetDataset(train_dataset, transform=eval_transform)
        else:
            train_dataset = ImageFolder(args.train_data_path, transform=eval_transform)
        
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    
    results = {}
    
    # KNN Evaluation
    if args.eval_knn:
        if train_loader is None:
            print("Warning: KNN requires training data, skipping")
        else:
            if rank == 0:
                print("\n" + "=" * 60)
                print("Running KNN Evaluation")
                print("=" * 60)
            
            knn_evaluator = KNNEvaluator(
                k=max(args.knn_k),
                distance="cosine",
                device=device,
            )
            
            knn_results = knn_evaluator.run_full_evaluation(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                k_list=args.knn_k,
            )
            
            results['knn'] = {f"k={k}": v for k, v in knn_results.items()}
    
    # Linear Probing Evaluation
    if args.eval_linear:
        if train_loader is None:
            print("Warning: Linear probing requires training data, skipping")
        else:
            if rank == 0:
                print("\n" + "=" * 60)
                print("Running Linear Probing Evaluation")
                print("=" * 60)
            
            linear_evaluator = LinearProbeEvaluator(
                encoder_dim=encoder_dim,
                num_classes=args.num_classes,
                device=device,
                epochs=args.linear_epochs,
                lr=args.linear_lr,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            
            linear_results = linear_evaluator.run_full_evaluation(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
            )
            
            results['linear_probe'] = linear_results
    
    # Save results
    if rank == 0:
        print("\n" + "=" * 60)
        print("Final Results")
        print("=" * 60)
        print(json.dumps(results, indent=2))
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
