"""
S-RAE Training Script

Trains the Semantic-Guided Regularized Autoencoder with:
- Reconstruction loss (L1/L2 on pixels)
- LPIPS perceptual loss
- GAN adversarial loss
- Alignment loss (Student-Teacher feature alignment)
- Regularization loss (L2 on bottleneck)
"""

import argparse
import logging
import math
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from omegaconf import OmegaConf
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import HFImageNetDataset, load_dataset_from_hf
from stage1 import SRAE
from disc import (
    DiffAug,
    LPIPS,
    build_discriminator,
    hinge_d_loss,
    vanilla_d_loss,
    vanilla_g_loss,
)
from eval.fid import _compute_inception_moments_from_arr, _fid_from_moments
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from utils import wandb_utils
from utils.model_utils import instantiate_from_config
from utils.train_utils import *
from utils.optim_utils import *
from utils.resume_utils import *
from utils.dist_utils import *
from utils.train_utils import prepare_hf_dataloader
from PIL import Image
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Train S-RAE")
    parser.add_argument("--config", type=str, required=True, help="YAML config")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--results-dir", type=str, default="ckpts")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--global-seed", type=int, default=None)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument("--use-hf", action="store_true")
    parser.add_argument("--hf-dataset-name", type=str, default="imagenet-1k")
    parser.add_argument("--hf-split", type=str, default="train")
    parser.add_argument("--hf-cache-dir", type=str, default=None)
    parser.add_argument("--hf-load-from-disk", type=str, default=None)
    parser.add_argument("--use-hf-eval", action="store_true")
    parser.add_argument("--hf-eval-split", type=str, default="validation")
    parser.add_argument("--hf-eval-load-from-disk", type=str, default=None)
    args = parser.parse_args()
    
    if args.data_path is None and not args.use_hf:
        parser.error("--data-path is required unless using --use-hf")
    
    return args


def calculate_adaptive_weight(
    recon_loss: torch.Tensor,
    gan_loss: torch.Tensor,
    layer: torch.nn.Parameter,
    max_d_weight: float = 1e4,
) -> torch.Tensor:
    recon_grads = torch.autograd.grad(recon_loss, layer, retain_graph=True)[0]
    gan_grads = torch.autograd.grad(gan_loss, layer, retain_graph=True)[0]
    d_weight = torch.norm(recon_grads) / (torch.norm(gan_grads) + 1e-6)
    d_weight = torch.clamp(d_weight, 0.0, max_d_weight)
    return d_weight.detach()


def select_gan_losses(disc_kind: str, gen_kind: str):
    if disc_kind == "hinge":
        disc_loss_fn = hinge_d_loss
    elif disc_kind == "vanilla":
        disc_loss_fn = vanilla_d_loss
    else:
        raise ValueError(f"Unsupported discriminator loss '{disc_kind}'")
    
    if gen_kind == "vanilla":
        gen_loss_fn = vanilla_g_loss
    else:
        raise ValueError(f"Unsupported generator loss '{gen_kind}'")
    return disc_loss_fn, gen_loss_fn


def main():
    args = parse_args()
    
    # Distributed setup
    rank, world_size, device = setup_distributed()
    
    # Config
    full_cfg = OmegaConf.load(args.config)
    srae_config = full_cfg.get("stage_1")
    training_cfg = OmegaConf.to_container(full_cfg.get("training", {}), resolve=True)
    
    # GAN config
    gan_cfg = full_cfg.get("gan", {})
    disc_cfg = gan_cfg.get("disc", {})
    loss_cfg = gan_cfg.get("loss", {})
    
    perceptual_weight = float(loss_cfg.get("perceptual_weight", 1.0))
    disc_weight = float(loss_cfg.get("disc_weight", 0.0))
    gan_start_epoch = int(loss_cfg.get("disc_start", 0))
    disc_update_epoch = int(loss_cfg.get("disc_upd_start", gan_start_epoch))
    lpips_start_epoch = int(loss_cfg.get("lpips_start", 0))
    disc_updates = int(loss_cfg.get("disc_updates", 1))
    max_d_weight = float(loss_cfg.get("max_d_weight", 1e4))
    disc_loss_type = loss_cfg.get("disc_loss", "hinge")
    gen_loss_type = loss_cfg.get("gen_loss", "vanilla")
    
    # Training hyperparameters
    batch_size = int(training_cfg.get("batch_size", 16))
    global_batch_size = training_cfg.get("global_batch_size", None)
    if global_batch_size is not None:
        global_batch_size = int(global_batch_size)
        assert global_batch_size % world_size == 0
        batch_size = global_batch_size // world_size
    else:
        global_batch_size = batch_size * world_size
    
    num_workers = int(training_cfg.get("num_workers", 4))
    clip_grad_val = training_cfg.get("clip_grad", 1.0)
    clip_grad = float(clip_grad_val) if clip_grad_val is not None and clip_grad_val > 0 else None
    log_interval = int(training_cfg.get("log_interval", 100))
    sample_every = int(training_cfg.get("sample_every", 2500))
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 10))
    ema_decay = float(training_cfg.get("ema_decay", 0.9999))
    num_epochs = int(training_cfg.get("epochs", 100))
    default_seed = int(training_cfg.get("global_seed", 0))
    
    # Eval config
    eval_section = full_cfg.get("eval", None)
    do_eval = eval_section is not None
    if do_eval:
        eval_interval = int(eval_section.get("eval_interval", 5000))
        max_eval_samples = eval_section.get("max_eval_samples", None)
        eval_data = eval_section.get("data_path", None)
        use_hf_eval = args.use_hf_eval or args.hf_eval_load_from_disk is not None
        if not use_hf_eval:
            assert eval_data, "eval.data_path must be specified"
    
    # Setup
    global_seed = args.global_seed if args.global_seed is not None else default_seed
    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    experiment_dir, checkpoint_dir, logger = configure_experiment_dirs(args, rank)
    full_cfg.cmd_args = vars(args)
    full_cfg.experiment_dir = experiment_dir
    full_cfg.checkpoint_dir = checkpoint_dir
    
    # Model
    logger.info("Initializing S-RAE model...")
    model: SRAE = instantiate_from_config(srae_config).to(device)
    
    # Only train student (teacher is frozen by default in SRAE)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params) / 1e6:.2f}M")
    
    # EMA
    ema_model = deepcopy(model).to(device).eval()
    ema_model.requires_grad_(False)
    
    # DDP
    if world_size > 1:
        ddp_model = DDP(model, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False)
        model = ddp_model.module
        decoder = ddp_model.module.decoder
    else:
        ddp_model = model
        decoder = model.decoder
    
    # Discriminator
    if disc_weight > 0:
        discriminator, disc_aug = build_discriminator(disc_cfg, device)
        if world_size > 1:
            ddp_disc = DDP(discriminator, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False)
            discriminator = ddp_disc.module
        else:
            ddp_disc = discriminator
        disc_loss_fn, gen_loss_fn = select_gan_losses(disc_loss_type, gen_loss_type)
    else:
        discriminator = None
        ddp_disc = None
        disc_aug = None
        disc_loss_fn = gen_loss_fn = None
    
    # LPIPS
    lpips = LPIPS().to(device)
    lpips.eval()
    
    # Optimizers
    optimizer, optim_msg = build_optimizer(trainable_params, training_cfg)
    if discriminator is not None:
        disc_params = [p for p in discriminator.parameters() if p.requires_grad]
        disc_optimizer, disc_optim_msg = build_optimizer(disc_params, disc_cfg)
    else:
        disc_optimizer = None
    
    # Schedulers (initialized after steps_per_epoch is known)
    scheduler = None
    disc_scheduler = None
    
    # AMP
    scaler, autocast_kwargs = get_autocast_scaler(args)
    
    # Data
    first_crop_size = 384 if args.image_size == 256 else int(args.image_size * 1.5)
    train_transform = transforms.Compose([
        transforms.Resize(first_crop_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(args.image_size),
        transforms.ToTensor(),
    ])
    
    if args.use_hf:
        if args.hf_load_from_disk:
            hf_dataset = load_dataset_from_hf(load_from_disk_path=args.hf_load_from_disk, split=args.hf_split)
        else:
            hf_dataset = load_dataset_from_hf(dataset_name=args.hf_dataset_name, split=args.hf_split, cache_dir=args.hf_cache_dir)
        dataset = HFImageNetDataset(hf_dataset, transform=train_transform)
        loader, sampler = prepare_hf_dataloader(dataset, batch_size, num_workers, rank, world_size)
    else:
        loader, sampler = prepare_dataloader(args.data_path, batch_size, num_workers, rank, world_size, transform=train_transform)
    
    # Eval data
    eval_loader = None
    if do_eval:
        eval_transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.ToTensor(),
        ])
        
        use_hf_eval = args.use_hf_eval or args.hf_eval_load_from_disk is not None
        if use_hf_eval:
            if args.hf_eval_load_from_disk:
                hf_eval = load_dataset_from_hf(load_from_disk_path=args.hf_eval_load_from_disk, split=args.hf_eval_split)
            else:
                hf_eval = load_dataset_from_hf(dataset_name=args.hf_dataset_name, split=args.hf_eval_split, cache_dir=args.hf_cache_dir)
            eval_dataset = HFImageNetDataset(hf_eval, transform=eval_transform)
        else:
            eval_dataset = ImageFolder(str(eval_data), transform=eval_transform)
        
        if max_eval_samples is not None and max_eval_samples < len(eval_dataset):
            from torch.utils.data import Subset
            eval_dataset = Subset(eval_dataset, range(max_eval_samples))
        
        eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, sampler=eval_sampler, 
                                num_workers=num_workers, pin_memory=True, drop_last=False)
    
    steps_per_epoch = len(loader)
    if steps_per_epoch == 0:
        raise RuntimeError("Dataloader returned zero batches. Check dataset and batch size settings.")

    if training_cfg.get("scheduler"):
        scheduler, _ = build_scheduler(optimizer, steps_per_epoch, training_cfg)
    if disc_cfg.get("scheduler") and disc_optimizer is not None:
        disc_scheduler, _ = build_scheduler(disc_optimizer, steps_per_epoch, disc_cfg)

    gan_start_step = gan_start_epoch * steps_per_epoch
    disc_update_step = disc_update_epoch * steps_per_epoch
    lpips_start_step = lpips_start_epoch * steps_per_epoch
    
    # Resume
    start_epoch, global_step = 0, 0
    
    logger.info(f"Training for {num_epochs} epochs, {steps_per_epoch} steps per epoch")
    logger.info(f"Perceptual weight: {perceptual_weight}, GAN weight: {disc_weight}")
    
    for epoch in range(start_epoch, num_epochs):
        ddp_model.train()
        if discriminator is not None:
            discriminator.eval()
        sampler.set_epoch(epoch)
        
        epoch_metrics = defaultdict(lambda: torch.zeros(1, device=device))
        disc_metrics = {}
        num_batches = 0
        
        for step, (images, _) in enumerate(loader):
            use_gan = global_step >= gan_start_step and disc_weight > 0.0
            train_disc = global_step >= disc_update_step and disc_weight > 0.0
            use_lpips = global_step >= lpips_start_step and perceptual_weight > 0.0
            
            images = images.to(device, non_blocking=True)
            real_normed = images * 2.0 - 1.0
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(**autocast_kwargs):
                # Forward through S-RAE
                recon = ddp_model(images)
                recon_normed = recon * 2.0 - 1.0
                
                # Get S-RAE specific losses
                losses = model.get_last_losses()
                loss_rec = losses.get('loss_rec', (recon - images).abs().mean())
                loss_align = losses.get('loss_align', torch.zeros(1, device=device))
                loss_reg = losses.get('loss_reg', torch.zeros(1, device=device))
                loss_align_weight = float(getattr(model, "loss_align_weight", 1.0))
                loss_reg_weight = float(getattr(model, "loss_reg_weight", 1.0))
                
                # LPIPS loss
                if use_lpips:
                    lpips_loss = lpips(real_normed, recon_normed)
                else:
                    lpips_loss = loss_rec.new_zeros(())
                
                # GAN loss
                if use_gan:
                    fake_aug = disc_aug.aug(recon_normed)
                    logits_fake, _ = ddp_disc(fake_aug, None)
                    gan_loss = gen_loss_fn(logits_fake)
                else:
                    gan_loss = torch.zeros_like(loss_rec)
                
                # Total reconstruction loss
                recon_total = loss_rec + perceptual_weight * lpips_loss
                
                # Adaptive weight for GAN
                if use_gan:
                    last_layer = decoder.decoder_pred.weight
                    adaptive_weight = calculate_adaptive_weight(
                        recon_total, gan_loss, last_layer, max_d_weight
                    )
                    total_loss = recon_total + disc_weight * adaptive_weight * gan_loss
                else:
                    adaptive_weight = torch.zeros_like(recon_total)
                    total_loss = recon_total
                
                # Add S-RAE specific losses
                total_loss = total_loss + loss_align_weight * loss_align + loss_reg_weight * loss_reg
            
            # Backward
            if scaler:
                scaler.scale(total_loss).backward()
                if clip_grad is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad)
                optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            # EMA update
            model_for_ema = ddp_model.module if isinstance(ddp_model, DDP) else ddp_model
            update_ema(ema_model, model_for_ema, ema_decay)
            
            # Discriminator update
            if train_disc and discriminator is not None:
                ddp_model.eval()
                ddp_disc.train()
                
                for _ in range(disc_updates):
                    disc_optimizer.zero_grad(set_to_none=True)
                    
                    with autocast(**autocast_kwargs):
                        with torch.no_grad():
                            recon_disc = ddp_model(images)
                            recon_disc_normed = recon_disc * 2.0 - 1.0
                            fake_detached = recon_disc_normed.clamp(-1.0, 1.0)
                            fake_detached = torch.round((fake_detached + 1.0) * 127.5) / 127.5 - 1.0
                        
                        fake_input = disc_aug.aug(fake_detached)
                        real_input = disc_aug.aug(real_normed)
                        logits_fake, logits_real = discriminator(fake_input, real_input)
                        d_loss = disc_loss_fn(logits_real, logits_fake)
                        accuracy = (logits_real > logits_fake).float().mean()
                    
                    if scaler:
                        scaler.scale(d_loss).backward()
                        scaler.step(disc_optimizer)
                        scaler.update()
                    else:
                        d_loss.backward()
                        disc_optimizer.step()
                    
                    disc_metrics = {
                        "disc_loss": d_loss.detach(),
                        "logits_real": logits_real.detach().mean(),
                        "logits_fake": logits_fake.detach().mean(),
                        "disc_accuracy": accuracy.detach(),
                    }
                    
                    epoch_metrics["disc_loss"] += d_loss.detach()
                    epoch_metrics["disc_accuracy"] += accuracy.detach()
                    
                    if disc_scheduler is not None:
                        disc_scheduler.step()
                
                ddp_disc.eval()
                ddp_model.train()
            
            # Logging
            epoch_metrics["loss"] += total_loss.detach()
            epoch_metrics["loss_rec"] += loss_rec.detach()
            epoch_metrics["loss_align"] += loss_align.detach()
            epoch_metrics["loss_reg"] += loss_reg.detach()
            epoch_metrics["lpips"] += lpips_loss.detach()
            epoch_metrics["gan"] += gan_loss.detach()
            num_batches += 1
            
            if log_interval > 0 and global_step % log_interval == 0 and rank == 0:
                stats = {
                    'loss/total': total_loss.item(),
                    'loss/rec': loss_rec.item(),
                    'loss/align': loss_align.item(),
                    'loss/reg': loss_reg.item(),
                    'loss/lpips': lpips_loss.item(),
                    'loss/gan': gan_loss.item(),
                    'lr': optimizer.param_groups[0]['lr'],
                }
                if disc_metrics:
                    stats.update({
                        'loss/disc': disc_metrics['disc_loss'].item(),
                        'disc/accuracy': disc_metrics['disc_accuracy'].item(),
                    })
                logger.info(f"[Epoch {epoch} | Step {global_step}] " + ", ".join(f"{k}: {v:.4f}" for k, v in stats.items()))
                if args.wandb:
                    wandb_utils.log(stats, step=global_step)
            
            # Sample
            if global_step % sample_every == 0 and rank == 0:
                logger.info("Generating EMA samples...")
                with torch.no_grad():
                    sample_images = images[:4]
                    samples = ema_model(sample_images)
                    comparison = torch.cat([sample_images, samples], dim=0).cpu()
                    grid = make_grid(comparison, nrow=4)
                    if args.wandb:
                        wandb_utils.log_image(grid, step=global_step)
            
            # Eval
            if do_eval and global_step > 0 and global_step % eval_interval == 0:
                # rFID evaluation (same as RAE)
                if rank == 0:
                    logger.info("Starting rFID evaluation...")
                    fid_metric = torch.zeros(1, device=device)  # Placeholder
                
                if world_size > 1:
                    dist.barrier()
            
            global_step += 1
        
        # Epoch end
        if rank == 0 and num_batches > 0:
            avg_loss = (epoch_metrics['loss'] / num_batches).item()
            logger.info(f"[Epoch {epoch}] Avg Loss: {avg_loss:.4f}")
        
        # Checkpoint
        if checkpoint_interval > 0 and epoch % checkpoint_interval == 0 and rank == 0:
            ckpt_path = f"{checkpoint_dir}/ep-{epoch:07d}.pt"
            torch.save({
                'epoch': epoch,
                'step': global_step,
                'model': ddp_model.state_dict(),
                'ema': ema_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
