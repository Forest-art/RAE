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
from typing import Dict, Optional, Tuple
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
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
from eval import evaluate_reconstruction_distributed
from eval.fid import calculate_rfid
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
    parser.add_argument("--eval-every-steps", type=int, default=None, help="Run periodic rFID every N steps (0 disables). Defaults to periodic_eval.every_steps in config when omitted.")
    parser.add_argument("--eval-num-samples", type=int, default=None, help="Samples for periodic rFID. Use <=0 for full split. Defaults to periodic_eval.num_samples in config when omitted.")
    parser.add_argument("--eval-split", type=str, choices=["val", "train", "dummy"], default=None, help="Split for periodic rFID. Defaults to periodic_eval.split in config when omitted.")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Batch size for periodic rFID. Defaults to periodic_eval.batch_size or training batch size.")
    args = parser.parse_args()
    
    if args.data_path is None and not args.use_hf:
        parser.error("--data-path is required unless using --use-hf")
    if args.eval_every_steps is not None and args.eval_every_steps < 0:
        parser.error("--eval-every-steps must be >= 0")
    if args.eval_batch_size is not None and args.eval_batch_size <= 0:
        parser.error("--eval-batch-size must be > 0 when provided")
    
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


def _to_uint8_nhwc(images: torch.Tensor) -> np.ndarray:
    return images.clamp(0, 1).mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()


def _forward_reconstruction_for_eval(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    # S-RAE reconstruction metrics should use unmasked forward when available.
    try:
        return model(images, mask_ratio=0.0)
    except TypeError:
        return model(images)


@torch.no_grad()
def run_periodic_rfid_eval(
    model: torch.nn.Module,
    split_name: str,
    rank: int,
    world_size: int,
    device: torch.device,
    autocast_kwargs: dict,
    num_samples: int,
    eval_batch_size: int,
    eval_dataset=None,
    dummy_images: Optional[torch.Tensor] = None,
) -> Tuple[Optional[float], Optional[str]]:
    rfid_value: Optional[float] = None
    error_message: Optional[str] = None
    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()

    if world_size > 1:
        dist.barrier()
    try:
        if rank == 0:
            bs = max(1, int(eval_batch_size))
            if split_name == "dummy":
                if dummy_images is None or dummy_images.numel() == 0:
                    raise RuntimeError("Dummy periodic eval requires current batch images.")
                if num_samples <= 0:
                    eval_images = dummy_images
                else:
                    eval_images = dummy_images[:num_samples]
                eval_images = eval_images.to(device, non_blocking=True)
                with autocast(**autocast_kwargs):
                    recon = _forward_reconstruction_for_eval(model, eval_images)
                ref_arr = _to_uint8_nhwc(eval_images)
                rec_arr = _to_uint8_nhwc(recon)
            else:
                if eval_dataset is None:
                    raise RuntimeError(f"Periodic eval dataset is not available for split '{split_name}'.")
                n = len(eval_dataset) if num_samples <= 0 else min(num_samples, len(eval_dataset))
                if n <= 0:
                    raise RuntimeError("Periodic eval dataset is empty.")
                subset = Subset(eval_dataset, range(n))
                eval_loader = DataLoader(
                    subset,
                    batch_size=bs,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True,
                    drop_last=False,
                )
                ref_images, rec_images = [], []
                for images, _ in eval_loader:
                    images = images.to(device, non_blocking=True)
                    with autocast(**autocast_kwargs):
                        recon = _forward_reconstruction_for_eval(model, images)
                    ref_images.append(_to_uint8_nhwc(images))
                    rec_images.append(_to_uint8_nhwc(recon))
                ref_arr = np.concatenate(ref_images, axis=0)
                rec_arr = np.concatenate(rec_images, axis=0)

            device_str = "cuda" if device.type == "cuda" else "cpu"
            rfid_value = float(calculate_rfid(ref_arr, rec_arr, bs=bs, device=device_str))
    except Exception as exc:
        error_message = str(exc)
    finally:
        if world_size > 1:
            dist.barrier()
        if was_training and hasattr(model, "train"):
            model.train()

    return rfid_value, error_message


def main():
    args = parse_args()
    
    # Distributed setup
    rank, world_size, device = setup_distributed()
    
    # Config
    full_cfg = OmegaConf.load(args.config)
    srae_config = full_cfg.get("stage_1")
    training_section = full_cfg.get("training", None)
    training_cfg = OmegaConf.to_container(training_section, resolve=True) if training_section is not None else {}
    training_cfg = dict(training_cfg) if isinstance(training_cfg, dict) else {}
    
    # GAN config
    gan_section = full_cfg.get("gan", None)
    gan_cfg = OmegaConf.to_container(gan_section, resolve=True) if gan_section is not None else {}
    gan_cfg = dict(gan_cfg) if isinstance(gan_cfg, dict) else {}
    if not gan_cfg:
        raise ValueError("Config must define a top-level 'gan' section for S-RAE training.")
    disc_cfg = gan_cfg.get("disc", {})
    disc_cfg = dict(disc_cfg) if isinstance(disc_cfg, dict) else {}
    if not disc_cfg:
        raise ValueError("gan.disc configuration is required for S-RAE training.")
    loss_cfg = gan_cfg.get("loss", {})
    loss_cfg = dict(loss_cfg) if isinstance(loss_cfg, dict) else {}
    
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
    raw_rec_loss_mode = str(training_cfg.get("rec_loss_mode", "pixel_l1")).strip().lower()
    if raw_rec_loss_mode in {"pixel_l1", "pixel", "l1", "full_l1"}:
        rec_loss_mode = "pixel_l1"
    elif raw_rec_loss_mode in {"masked_mse", "patch_mse", "mse"}:
        rec_loss_mode = "masked_mse"
    else:
        raise ValueError(
            f"Unsupported training.rec_loss_mode='{raw_rec_loss_mode}'. "
            "Expected one of: pixel_l1, masked_mse."
        )
    
    # Periodic eval config
    periodic_eval_section = full_cfg.get("periodic_eval", None)
    periodic_eval_cfg = OmegaConf.to_container(periodic_eval_section, resolve=True) if periodic_eval_section is not None else {}
    periodic_eval_cfg = dict(periodic_eval_cfg) if isinstance(periodic_eval_cfg, dict) else {}
    periodic_eval_every_steps = int(
        args.eval_every_steps
        if args.eval_every_steps is not None
        else periodic_eval_cfg.get("every_steps", 0)
    )
    periodic_eval_num_samples = int(
        args.eval_num_samples
        if args.eval_num_samples is not None
        else periodic_eval_cfg.get("num_samples", 32)
    )
    periodic_eval_split = (
        args.eval_split
        if args.eval_split is not None
        else periodic_eval_cfg.get("split", "val")
    )
    periodic_eval_batch_size = int(
        args.eval_batch_size
        if args.eval_batch_size is not None
        else periodic_eval_cfg.get("batch_size", batch_size)
    )
    if periodic_eval_every_steps < 0:
        raise ValueError("Periodic evaluation every_steps must be >= 0.")
    if periodic_eval_split not in {"val", "train", "dummy"}:
        raise ValueError(
            f"Invalid periodic evaluation split '{periodic_eval_split}'. "
            "Expected one of: val, train, dummy."
        )
    if periodic_eval_batch_size <= 0:
        raise ValueError("Periodic evaluation batch_size must be > 0.")

    # Full eval config
    do_eval = False
    eval_setup_error: Optional[str] = None
    eval_setup_note: Optional[str] = None
    eval_interval = 0
    eval_model = False
    eval_metrics = ("rfid", "psnr", "ssim")
    eval_data = None
    max_eval_samples = None
    reference_npz_path = None
    eval_section = full_cfg.get("eval", None)
    if eval_section:
        eval_interval = int(eval_section.get("eval_interval", 5000))
        eval_model = bool(eval_section.get("eval_model", False))
        eval_metrics = tuple(eval_section.get("metrics", ("rfid", "psnr", "ssim")))
        eval_data = eval_section.get("data_path", None)
        max_eval_samples = eval_section.get("max_eval_samples", None)
        reference_npz_path = eval_section.get("reference_npz_path", None)
        use_hf_eval = args.use_hf_eval or args.hf_eval_load_from_disk is not None
        missing_fields = []
        if eval_interval > 0:
            if not use_hf_eval and not eval_data:
                missing_fields.append("eval.data_path (or --use-hf-eval/--hf-eval-load-from-disk)")
            if len(eval_metrics) == 0:
                missing_fields.append("eval.metrics")
            if reference_npz_path and not os.path.exists(reference_npz_path):
                eval_setup_note = (
                    f"eval.reference_npz_path not found at {reference_npz_path}; "
                    "falling back to online reference images from the eval dataset."
                )
                reference_npz_path = None
            elif reference_npz_path is None:
                eval_setup_note = "Using online reference images from the eval dataset (no eval.reference_npz_path provided)."
            do_eval = len(missing_fields) == 0
            if not do_eval:
                eval_setup_error = (
                    "Disabling full reconstruction evaluation because: "
                    + "; ".join(missing_fields)
                )
    
    # Setup
    global_seed = args.global_seed if args.global_seed is not None else default_seed
    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    experiment_dir, checkpoint_dir, logger = configure_experiment_dirs(args, rank)
    if eval_setup_error and rank == 0:
        logger.warning(eval_setup_error)
    if eval_setup_note and rank == 0 and do_eval:
        logger.info(eval_setup_note)
    full_cfg.cmd_args = vars(args)
    full_cfg.experiment_dir = experiment_dir
    full_cfg.checkpoint_dir = checkpoint_dir
    
    # Model
    logger.info("Initializing S-RAE model...")
    model: SRAE = instantiate_from_config(srae_config).to(device)
    effective_image_size = int(args.image_size)
    model_image_size = int(getattr(model, "img_size", effective_image_size))
    if model_image_size != effective_image_size and rank == 0:
        logger.warning(
            f"Requested --image-size={effective_image_size} but S-RAE model reconstructs img_size={model_image_size}. "
            "Training/eval loaders will follow --image-size."
        )
    
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
    first_crop_size = 384 if effective_image_size == 256 else int(effective_image_size * 1.5)
    train_transform = transforms.Compose([
        transforms.Resize(first_crop_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(effective_image_size),
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
    
    train_dataset = loader.dataset
    eval_dataset = None
    if do_eval:
        eval_transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, effective_image_size)),
            transforms.ToTensor(),
        ])

        use_hf_eval = args.use_hf_eval or args.hf_eval_load_from_disk is not None
        if use_hf_eval:
            if args.hf_eval_load_from_disk:
                hf_eval_dataset = load_dataset_from_hf(
                    load_from_disk_path=args.hf_eval_load_from_disk,
                    split=args.hf_eval_split,
                )
            else:
                hf_eval_dataset = load_dataset_from_hf(
                    dataset_name=args.hf_dataset_name,
                    split=args.hf_eval_split,
                    cache_dir=args.hf_cache_dir,
                )
            eval_dataset = HFImageNetDataset(hf_eval_dataset, transform=eval_transform)
        else:
            eval_dataset = ImageFolder(str(eval_data), transform=eval_transform)

        if max_eval_samples is not None and max_eval_samples < len(eval_dataset):
            eval_dataset = Subset(eval_dataset, range(max_eval_samples))

        if rank == 0:
            logger.info(f"Evaluation dataset loaded, containing {len(eval_dataset)} images.")

    periodic_eval_enabled = periodic_eval_every_steps > 0
    periodic_eval_dataset = None
    if periodic_eval_enabled:
        try:
            periodic_eval_transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, effective_image_size)),
                transforms.ToTensor(),
            ])
            if periodic_eval_split == "dummy":
                periodic_eval_dataset = None
            elif periodic_eval_split == "train":
                periodic_eval_dataset = train_dataset
            else:
                if eval_dataset is not None:
                    periodic_eval_dataset = eval_dataset
                elif args.use_hf or args.use_hf_eval or args.hf_eval_load_from_disk is not None:
                    target_split = args.hf_eval_split
                    hf_source_path = args.hf_eval_load_from_disk or args.hf_load_from_disk
                    if hf_source_path:
                        hf_eval_dataset = load_dataset_from_hf(
                            load_from_disk_path=hf_source_path,
                            split=target_split,
                        )
                    else:
                        hf_eval_dataset = load_dataset_from_hf(
                            dataset_name=args.hf_dataset_name,
                            split=target_split,
                            cache_dir=args.hf_cache_dir,
                        )
                    periodic_eval_dataset = HFImageNetDataset(hf_eval_dataset, transform=periodic_eval_transform)
                else:
                    if args.data_path is None:
                        raise FileNotFoundError("Validation data path is unavailable for periodic eval split=val.")
                    val_path = args.data_path.parent / "val"
                    if not val_path.exists():
                        val_path = args.data_path.parent / "validation"
                    if not val_path.exists():
                        raise FileNotFoundError(
                            "Could not infer validation path for periodic eval split=val. "
                            "Expected sibling folder named 'val' or 'validation'."
                        )
                    periodic_eval_dataset = ImageFolder(str(val_path), transform=periodic_eval_transform)

            if rank == 0:
                if periodic_eval_split == "dummy":
                    sample_desc = "full current batch" if periodic_eval_num_samples <= 0 else str(periodic_eval_num_samples)
                    logger.info(
                        f"Periodic rFID enabled every {periodic_eval_every_steps} steps on split=dummy "
                        f"(num_samples={sample_desc}, batch_size={periodic_eval_batch_size})."
                    )
                else:
                    chosen_samples = len(periodic_eval_dataset) if periodic_eval_num_samples <= 0 else min(periodic_eval_num_samples, len(periodic_eval_dataset))
                    sample_desc = f"{chosen_samples} (full dataset)" if periodic_eval_num_samples <= 0 else str(chosen_samples)
                    logger.info(
                        f"Periodic rFID enabled every {periodic_eval_every_steps} steps on split={periodic_eval_split} "
                        f"with {sample_desc} samples (batch_size={periodic_eval_batch_size})."
                    )
        except Exception as exc:
            periodic_eval_enabled = False
            if rank == 0:
                logger.error(f"Disabling periodic rFID evaluation due to setup error: {exc}")
    
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
    logger.info(f"Reconstruction loss mode: {rec_loss_mode} (RAE-compatible is pixel_l1)")
    loss_rec_weight = float(getattr(model, "loss_rec_weight", 1.0))
    loss_align_weight = float(getattr(model, "loss_align_weight", 1.0))
    loss_reg_weight = float(getattr(model, "loss_reg_weight", 1.0))
    logger.info(
        f"S-RAE loss weights: rec={loss_rec_weight}, align={loss_align_weight}, reg={loss_reg_weight}"
    )
    
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
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(**autocast_kwargs):
                # Forward through S-RAE
                recon = ddp_model(images)
                if recon.shape[-2:] != images.shape[-2:]:
                    recon_target = F.interpolate(images, size=recon.shape[-2:], mode="bicubic", align_corners=False)
                else:
                    recon_target = images
                real_normed = recon_target * 2.0 - 1.0
                recon_normed = recon * 2.0 - 1.0
                
                # Get S-RAE specific losses
                losses = model.get_last_losses()
                loss_rec_masked_mse = losses.get('loss_rec', None)
                if loss_rec_masked_mse is None:
                    loss_rec_masked_mse = recon.new_zeros(())
                loss_rec_pixel_l1 = (recon - recon_target).abs().mean()
                if rec_loss_mode == "masked_mse":
                    loss_rec = loss_rec_masked_mse
                else:
                    loss_rec = loss_rec_pixel_l1
                loss_align = losses.get('loss_align', None)
                if loss_align is None:
                    loss_align = loss_rec.new_zeros(())
                loss_reg = losses.get('loss_reg', None)
                if loss_reg is None:
                    loss_reg = loss_rec.new_zeros(())
                
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
                recon_total = loss_rec_weight * loss_rec + perceptual_weight * lpips_loss
                
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
            epoch_metrics["loss_rec_pixel_l1"] += loss_rec_pixel_l1.detach()
            epoch_metrics["loss_rec_masked_mse"] += loss_rec_masked_mse.detach()
            epoch_metrics["loss_align"] += loss_align.detach()
            epoch_metrics["loss_reg"] += loss_reg.detach()
            epoch_metrics["lpips"] += lpips_loss.detach()
            epoch_metrics["gan"] += gan_loss.detach()
            num_batches += 1
            
            if log_interval > 0 and global_step % log_interval == 0 and rank == 0:
                stats = {
                    'loss/total': total_loss.item(),
                    'loss/rec': loss_rec.item(),
                    'loss/rec_pixel_l1': loss_rec_pixel_l1.item(),
                    'loss/rec_masked_mse': loss_rec_masked_mse.item(),
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
                    samples = _forward_reconstruction_for_eval(ema_model, sample_images)
                    comparison = torch.cat([sample_images, samples], dim=0).cpu()
                    grid = make_grid(comparison, nrow=4)
                    if args.wandb:
                        wandb_utils.log_image(grid, step=global_step)
            
            # Full eval
            if do_eval and (eval_interval > 0 and global_step % eval_interval == 0):
                logger.info("Starting evaluation...")
                current_model = ddp_model.module if isinstance(ddp_model, DDP) else ddp_model
                eval_models = [(current_model, "model")]
                for eval_mod, mod_name in eval_models:
                    eval_stats = evaluate_reconstruction_distributed(
                        eval_mod,
                        eval_dataset,
                        len(eval_dataset),
                        rank=rank,
                        world_size=world_size,
                        device=device,
                        batch_size=batch_size,
                        metrics_to_compute=eval_metrics,
                        experiment_dir=experiment_dir,
                        global_step=global_step,
                        autocast_kwargs=autocast_kwargs,
                        reference_npz_path=reference_npz_path,
                        forward_kwargs={"mask_ratio": 0.0},
                    )
                    eval_stats = {f"eval_{mod_name}/{k}": v for k, v in eval_stats.items()} if eval_stats is not None else {}
                    if args.wandb:
                        wandb_utils.log(eval_stats, step=global_step)
                logger.info("Evaluation done.")

            # Periodic eval
            if periodic_eval_enabled and global_step > 0 and (global_step % periodic_eval_every_steps == 0):
                eval_model_ref = ddp_model.module if isinstance(ddp_model, DDP) else ddp_model
                eval_rfid, eval_error = run_periodic_rfid_eval(
                    model=eval_model_ref,
                    split_name=periodic_eval_split,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    autocast_kwargs=autocast_kwargs,
                    num_samples=periodic_eval_num_samples,
                    eval_batch_size=periodic_eval_batch_size,
                    eval_dataset=periodic_eval_dataset,
                    dummy_images=images if periodic_eval_split == "dummy" else None,
                )
                if rank == 0:
                    if eval_error is not None:
                        logger.error(f"Periodic rFID evaluation failed at step {global_step}: {eval_error}")
                    elif eval_rfid is not None:
                        periodic_stats = {"eval_periodic/rfid": eval_rfid}
                        logger.info(
                            f"[Epoch {epoch} | Step {global_step}] "
                            f"eval_periodic/rfid: {eval_rfid:.6f}"
                        )
                        if args.wandb:
                            wandb_utils.log(periodic_stats, step=global_step)
            
            global_step += 1
        
        # Epoch end
        if rank == 0 and num_batches > 0:
            avg_loss = (epoch_metrics['loss'] / num_batches).item()
            avg_rec = (epoch_metrics['loss_rec'] / num_batches).item()
            avg_rec_l1 = (epoch_metrics['loss_rec_pixel_l1'] / num_batches).item()
            avg_rec_masked = (epoch_metrics['loss_rec_masked_mse'] / num_batches).item()
            logger.info(
                f"[Epoch {epoch}] Avg Loss: {avg_loss:.4f}, "
                f"Avg Rec({rec_loss_mode}): {avg_rec:.4f}, "
                f"Avg Rec(pixel_l1): {avg_rec_l1:.4f}, "
                f"Avg Rec(masked_mse): {avg_rec_masked:.4f}"
            )
        
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
