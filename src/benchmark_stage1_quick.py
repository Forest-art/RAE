"""
Quick multi-model Stage-1 benchmark.

Runs fast reconstruction + understanding evaluation for multiple model checkpoints.
Supports single-GPU and torchrun multi-GPU execution.

Model spec format:
  --model <name>::<config_path>::<checkpoint_path>
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from omegaconf import OmegaConf

from dataset import HFImageNetDataset, load_dataset_from_hf
from eval import evaluate_reconstruction_distributed
from eval.understanding import KNNEvaluator, LinearProbeEvaluator
from utils.dist_utils import cleanup_distributed, setup_distributed
from utils.model_utils import instantiate_from_config


def parse_args():
    parser = argparse.ArgumentParser(description="Quick Stage-1 benchmark (multi-model, multi-GPU)")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec: name::config_path::checkpoint_path (can be repeated)",
    )

    parser.add_argument("--use-hf", action="store_true", help="Use HuggingFace datasets from disk.")
    parser.add_argument("--train-data-path", type=str, default=None, help="ImageFolder train path (local mode).")
    parser.add_argument("--val-data-path", type=str, default=None, help="ImageFolder val path (local mode).")
    parser.add_argument("--hf-train-path", type=str, default=None, help="HF dataset save_to_disk path for train.")
    parser.add_argument("--hf-val-path", type=str, default=None, help="HF dataset save_to_disk path for val.")
    parser.add_argument("--hf-train-split", type=str, default="train")
    parser.add_argument("--hf-val-split", type=str, default="validation")

    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")

    parser.add_argument("--recon-num-samples", type=int, default=2048, help="<=0 means full val split.")
    parser.add_argument("--recon-metrics", nargs="+", default=["psnr", "ssim", "rfid"])

    parser.add_argument("--understanding-train-samples", type=int, default=50000, help="<=0 means full train split.")
    parser.add_argument("--understanding-val-samples", type=int, default=5000, help="<=0 means full val split.")
    parser.add_argument("--skip-knn", action="store_true")
    parser.add_argument("--skip-linear", action="store_true")
    parser.add_argument("--knn-k", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--linear-epochs", type=int, default=20)
    parser.add_argument("--linear-lr", type=float, default=1e-3)

    parser.add_argument(
        "--checkpoint-key",
        choices=["auto", "ema", "model", "state_dict"],
        default="auto",
        help="Which key to load from checkpoint dict.",
    )
    parser.add_argument("--output", type=str, required=True, help="Output JSON path.")
    return parser.parse_args()


def parse_model_spec(spec: str) -> Dict[str, str]:
    parts = spec.split("::")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid --model '{spec}'. Expected format: name::config_path::checkpoint_path"
        )
    return {"name": parts[0], "config": parts[1], "checkpoint": parts[2]}


def subset_dataset(dataset, max_samples: int):
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, range(max_samples))


def shard_dataset_for_rank(dataset, rank: int, world_size: int):
    if world_size <= 1:
        return dataset
    indices = list(range(rank, len(dataset), world_size))
    return Subset(dataset, indices)


def maybe_strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def select_state_dict(raw_ckpt, key: str):
    if not isinstance(raw_ckpt, dict):
        return raw_ckpt

    if key != "auto":
        if key in raw_ckpt:
            return raw_ckpt[key]
        return raw_ckpt

    for candidate in ("ema", "model", "state_dict"):
        if candidate in raw_ckpt:
            return raw_ckpt[candidate]
    return raw_ckpt


@torch.no_grad()
def infer_encoder_dim(model, device: torch.device, image_size: int) -> int:
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    if hasattr(model, "forward_student"):
        z, _, _, _ = model.forward_student(dummy, mask_ratio=0.0)
        z = z.mean(dim=1)
    elif hasattr(model, "encode"):
        z = model.encode(dummy)
    else:
        z = model(dummy)
        if isinstance(z, dict):
            z = z.get("latent", z.get("z", z))
    if z.dim() > 2:
        z = z.flatten(1)
    return int(z.shape[1])


def build_datasets(args, image_size: int):
    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    if args.use_hf:
        if not args.hf_train_path or not args.hf_val_path:
            raise ValueError("--use-hf requires --hf-train-path and --hf-val-path.")
        train_data = load_dataset_from_hf(load_from_disk_path=args.hf_train_path, split=args.hf_train_split)
        val_data = load_dataset_from_hf(load_from_disk_path=args.hf_val_path, split=args.hf_val_split)
        train_dataset = HFImageNetDataset(train_data, transform=transform)
        val_dataset = HFImageNetDataset(val_data, transform=transform)
    else:
        if not args.train_data_path or not args.val_data_path:
            raise ValueError("Local mode requires --train-data-path and --val-data-path.")
        train_dataset = ImageFolder(args.train_data_path, transform=transform)
        val_dataset = ImageFolder(args.val_data_path, transform=transform)

    return train_dataset, val_dataset


def load_stage1_model(config_path: str, checkpoint_path: str, checkpoint_key: str, device: torch.device):
    cfg = OmegaConf.load(config_path)
    stage1_cfg = cfg.get("stage_1")
    if stage1_cfg is None:
        raise ValueError(f"Config has no stage_1 section: {config_path}")

    model = instantiate_from_config(stage1_cfg).to(device)
    raw_ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = select_state_dict(raw_ckpt, checkpoint_key)
    if isinstance(state_dict, dict):
        state_dict = maybe_strip_module_prefix(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, missing, unexpected


def get_autocast_kwargs(precision: str, device: torch.device) -> dict:
    if device.type == "cuda" and precision == "bf16":
        return {"enabled": True, "dtype": torch.bfloat16}
    return {"enabled": False}


def main():
    args = parse_args()
    rank, world_size, device = setup_distributed()

    model_specs = [parse_model_spec(s) for s in args.model]
    eval_knn = not args.skip_knn
    eval_linear = not args.skip_linear
    if not eval_knn and not eval_linear and rank == 0:
        print("Both understanding metrics are disabled (--skip-knn and --skip-linear).")

    train_full, val_full = build_datasets(args, args.image_size)
    recon_val = subset_dataset(val_full, args.recon_num_samples)

    understand_train = subset_dataset(train_full, args.understanding_train_samples)
    understand_val = subset_dataset(val_full, args.understanding_val_samples)
    understand_train_shard = shard_dataset_for_rank(understand_train, rank, world_size)
    understand_val_shard = shard_dataset_for_rank(understand_val, rank, world_size)

    train_loader = DataLoader(
        understand_train_shard,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        understand_val_shard,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    autocast_kwargs = get_autocast_kwargs(args.precision, device)

    if rank == 0:
        print(f"World size: {world_size}")
        print(f"Models: {[m['name'] for m in model_specs]}")
        print(f"Recon eval samples: {len(recon_val)}")
        print(f"Understanding train samples: {len(understand_train)}")
        print(f"Understanding val samples: {len(understand_val)}")

    results = {
        "meta": {
            "world_size": world_size,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "recon_metrics": args.recon_metrics,
            "knn_k": args.knn_k,
            "linear_epochs": args.linear_epochs,
            "linear_lr": args.linear_lr,
        },
        "models": {},
    }

    for spec in model_specs:
        model_name = spec["name"]
        if rank == 0:
            print(f"\n===== Evaluating {model_name} =====")

        model, missing, unexpected = load_stage1_model(
            spec["config"], spec["checkpoint"], args.checkpoint_key, device
        )
        if rank == 0 and (missing or unexpected):
            print(
                f"[{model_name}] load_state_dict strict=False: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )

        model_result: Dict[str, Dict] = {}

        forward_kwargs = {"mask_ratio": 0.0} if hasattr(model, "forward_student") else {}
        recon_stats = evaluate_reconstruction_distributed(
            model=model,
            val_dataset=recon_val,
            num_samples=len(recon_val),
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
            device=device,
            experiment_dir=str(output_path.parent),
            global_step=0,
            autocast_kwargs=autocast_kwargs,
            metrics_to_compute=args.recon_metrics,
            reference_npz_path=None,
            forward_kwargs=forward_kwargs,
        )
        if rank == 0 and recon_stats is not None:
            model_result["reconstruction"] = recon_stats

        if eval_knn or eval_linear:
            encoder_dim = infer_encoder_dim(model, device, args.image_size)
        else:
            encoder_dim = None

        if eval_knn:
            knn_evaluator = KNNEvaluator(
                k=max(args.knn_k),
                distance="cosine",
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            knn_stats = knn_evaluator.run_full_evaluation(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                k_list=args.knn_k,
            )
            if rank == 0 and knn_stats:
                model_result["knn"] = {f"k={k}": v for k, v in knn_stats.items()}
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        if eval_linear and encoder_dim is not None:
            linear_evaluator = LinearProbeEvaluator(
                encoder_dim=encoder_dim,
                num_classes=args.num_classes,
                device=device,
                epochs=args.linear_epochs,
                lr=args.linear_lr,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            linear_stats = linear_evaluator.run_full_evaluation(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
            )
            if rank == 0 and linear_stats:
                model_result["linear_probe"] = linear_stats
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        if rank == 0:
            results["models"][model_name] = model_result
            with output_path.open("w") as f:
                json.dump(results, f, indent=2)
            print(f"[{model_name}] done.")

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    if rank == 0:
        print(f"\nBenchmark results saved to {output_path}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
