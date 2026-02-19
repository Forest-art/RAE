## Diffusion Transformers with Representation Autoencoders (RAE)<br><sub>Official PyTorch Implementation</sub>

### [Paper](https://arxiv.org/abs/2510.11690) | [Project Page](https://rae-dit.github.io/)

This repository contains the PyTorch/GPU codebase for **Diffusion Transformers with Representation Autoencoders**.

It now includes two Stage-1 autoencoder variants:

- **RAE**: frozen representation encoder (for example DINOv2 / SigLIP2 / MAE) + trainable ViT decoder.
- **SRAE**: Semantic-Guided Regularized Autoencoder with teacher-student alignment and bottleneck regularization.

Both Stage-1 models can be used as tokenizers for Stage-2 latent diffusion (DiT^DH).

For the old code structure, see the [deprecated branch](https://github.com/bytetriper/RAE/tree/deprecated-gpu).
For JAX/TPU implementation, see [diffuse_nnx](https://github.com/willisma/diffuse_nnx).

## Environment

```bash
conda create -n rae python=3.10 -y
conda activate rae
pip install uv

# Example: PyTorch 2.8.0 + CUDA 12.9
uv pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
uv pip install -r requirements.txt
```

## Local Smoke Test (dummy_data)

Minimal CPU-friendly smoke test that exercises the dummy HF dataset + dataloader.

```bash
uv pip install -r requirements.txt
uv pip install datasets pillow

bash scripts/smoke_dummy.sh
```

If your login node lacks `torch`/`torchvision`, run the same command on a GPU node.
On clusters, a minimal Slurm batch job can call `bash scripts/smoke_dummy.sh`.

## Data And Pretrained Models

### Download pretrained collections

```bash
pip install huggingface_hub
hf download nyu-visionx/RAE-collections --local-dir models
```

### Dataset format

Default data format in this repo is HuggingFace `datasets` saved by `save_to_disk`.
Use one dataset root for both train/validation splits:

```text
/path/to/hf_imagenet/
  dataset_dict.json
  train/
  validation/
  ...
```

Expected columns are `image` and `label`.
All Stage-1 training/evaluation commands below use `--use-hf` + `--hf-load-from-disk` style arguments.

## Script Index

### Training

- `src/train_stage1.py`: Stage-1 **RAE** training.
- `src/train_srae.py`: Stage-1 **SRAE** training.
- `src/train.py`: Stage-2 latent diffusion (DiT^DH) training.

### Sampling And Reconstruction

- `src/stage1_sample.py`: single-image Stage-1 reconstruction.
- `src/stage1_sample_ddp.py`: distributed Stage-1 reconstruction + `.npz` packing.
- `src/sample.py`: quick Stage-2 sampling on one device.
- `src/sample_ddp.py`: distributed Stage-2 sampling + FID-ready `.npz`.

### Evaluation And Utilities

- `src/eval_understanding.py`: encoder understanding evaluation (KNN / linear probe).
- `src/benchmark_stage1_quick.py`: quick Stage-1 multi-model benchmark (reconstruction + understanding), supports multi-GPU.
- `src/calculate_stat.py`: latent mean/variance estimation for normalization stats.
- `pack_images.py`: pack image folders to `.npz`.
- `scripts/create_dummy_data.py`: create local dummy HF dataset.
- `scripts/test_srae_structure.py`: SRAE structure sanity test (no backbone download).
- `scripts/test_srae_model.py`: SRAE forward/encode/decode sanity test.

## Common Training Setup

All training entrypoints use `EXPERIMENT_NAME` to create output folders:

```bash
export EXPERIMENT_NAME=<your_experiment_name>
```

Outputs are written to:

```text
<results-dir>/<EXPERIMENT_NAME>/
  checkpoints/
  log.txt
  config.yaml
  src/
```

Optional Weights and Biases logging:

```bash
export ENTITY=<wandb_entity>
export PROJECT=<wandb_project>
# add --wandb to training command
```

## Stage-1 RAE Training

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/train_stage1.py \
  --config configs/stage1/training/DINOv2-B_decB.yaml \
  --use-hf \
  --hf-load-from-disk /path/to/hf_imagenet \
  --hf-split train \
  --results-dir ckpts/stage1 \
  --image-size 256 \
  --precision bf16
```

Notes:

- `train_stage1.py` supports auto-resume from the latest checkpoint in the experiment folder.
- Add `--compile` to enable `torch.compile` on model forward/encode.
- You can use HuggingFace dataset mode with `--use-hf` and related flags.

### Stage-1 RAE (HF `load_from_disk` + Online Full Eval)

Complete training command with HuggingFace local dataset loading, full online eval (no `reference_npz_path`), and full periodic eval:

```bash
export EXPERIMENT_NAME=rae_dino_decb_hf_full_eval_$(date +%Y%m%d_%H%M%S)

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/train_stage1.py \
  --config configs/stage1/training/DINOv2-B_decB.yaml \
  --use-hf \
  --hf-load-from-disk /path/to/hf_imagenet \
  --hf-split train \
  --use-hf-eval \
  --hf-eval-load-from-disk /path/to/hf_imagenet \
  --hf-eval-split validation \
  --results-dir ckpts/stage1 \
  --image-size 256 \
  --precision bf16 \
  --eval-every-steps 2500 \
  --eval-split val \
  --eval-num-samples -1
```

Notes:

- `--hf-load-from-disk` expects a dataset saved with `datasets.save_to_disk` (Dataset or DatasetDict).
- Full eval can run online from the eval dataset directly; `eval.reference_npz_path` is optional.
- `--eval-num-samples -1` means periodic eval uses the full selected split.

## Stage-1 SRAE Training

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/train_srae.py \
  --config configs/stage1/training/SRAE-B_decB.yaml \
  --use-hf \
  --hf-load-from-disk /path/to/hf_imagenet \
  --hf-split train \
  --use-hf-eval \
  --hf-eval-load-from-disk /path/to/hf_imagenet \
  --hf-eval-split validation \
  --results-dir ckpts/stage1 \
  --image-size 256 \
  --precision bf16
```

HuggingFace dataset example:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/train_srae.py \
  --config configs/stage1/training/SRAE-B_decB.yaml \
  --use-hf \
  --hf-load-from-disk /path/to/hf_imagenet \
  --hf-split train \
  --use-hf-eval \
  --hf-eval-load-from-disk /path/to/hf_imagenet \
  --hf-eval-split validation \
  --results-dir ckpts/stage1
```

Notes:

- `train_srae.py` saves checkpoints, but currently starts from epoch 0 on relaunch (no automatic resume flow like `train_stage1.py`).

## Stage-2 DiT Training

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/train.py \
  --config configs/stage2/training/ImageNet256/DiTDH-S_DINOv2-B.yaml \
  --data-path /path/to/imagenet/train \
  --results-dir ckpts/stage2 \
  --image-size 256 \
  --precision fp32 \
  --compile
```

Notes:

- `train.py` currently requires `--compile`.
- `train.py` supports auto-resume from latest checkpoint in the experiment folder.
- `train.py` currently reads ImageFolder (`--data-path`). If your source is HF `load_from_disk`, export train images once before Stage-2 training.

## Reconstruction And Sampling

### Stage-1 single image reconstruction

```bash
python src/stage1_sample.py \
  --config configs/stage1/pretrained/DINOv2-B.yaml \
  --image assets/pixabay_cat.png \
  --output recon.png
```

### Stage-1 distributed reconstruction (for rFID/FID)

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/stage1_sample_ddp.py \
  --config configs/stage1/pretrained/DINOv2-B.yaml \
  --data-path /path/to/imagenet/val \
  --sample-dir recon_samples \
  --per-proc-batch-size 32 \
  --num-samples 50000 \
  --precision fp32
```

Note: `stage1_sample_ddp.py` currently reads ImageFolder input (`--data-path`).

### Stage-2 quick sampling

```bash
python src/sample.py \
  --config configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B.yaml \
  --seed 42
```

### Stage-2 distributed sampling (for gFID)

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/sample_ddp.py \
  --config configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B.yaml \
  --sample-dir samples \
  --num-fid-samples 50000 \
  --per-proc-batch-size 125 \
  --label-sampling equal \
  --precision fp32
```

## Evaluation Scripts

### 1. Encoder understanding (KNN / Linear Probe)

Linear probe:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/eval_understanding.py \
  --config configs/stage1/training/SRAE-B_decB.yaml \
  --checkpoint ckpts/stage1/<exp>/checkpoints/ep-0000010.pt \
  --use-hf \
  --train-data-path /path/to/hf_imagenet \
  --data-path /path/to/hf_imagenet \
  --hf-split validation \
  --num-classes 1000 \
  --eval-linear \
  --linear-epochs 100 \
  --linear-lr 0.001 \
  --batch-size 256 \
  --output results_linear.json
```

KNN:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/eval_understanding.py \
  --config configs/stage1/training/SRAE-B_decB.yaml \
  --checkpoint ckpts/stage1/<exp>/checkpoints/ep-0000010.pt \
  --use-hf \
  --train-data-path /path/to/hf_imagenet \
  --data-path /path/to/hf_imagenet \
  --hf-split validation \
  --num-classes 1000 \
  --eval-knn \
  --knn-k 1 5 10 20 100 200 \
  --batch-size 256 \
  --output results_knn.json
```

Run both:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/eval_understanding.py \
  --config configs/stage1/training/SRAE-B_decB.yaml \
  --checkpoint ckpts/stage1/<exp>/checkpoints/ep-0000010.pt \
  --use-hf \
  --train-data-path /path/to/hf_imagenet \
  --data-path /path/to/hf_imagenet \
  --hf-split validation \
  --num-classes 1000 \
  --eval-linear \
  --eval-knn \
  --output results_understanding.json
```

### 2. Quick Stage-1 benchmark (RAE/SRAE multi-model)

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/benchmark_stage1_quick.py \
  --model rae::configs/stage1/training/DINOv2-B_decB.yaml::ckpts/stage1/<rae_exp>/checkpoints/ep-0000010.pt \
  --model srae::configs/stage1/training/SRAE-B_decB.yaml::ckpts/stage1/<srae_exp>/checkpoints/ep-0000010.pt \
  --use-hf \
  --hf-train-path /path/to/hf_imagenet \
  --hf-val-path /path/to/hf_imagenet \
  --hf-train-split train \
  --hf-val-split validation \
  --num-classes 1000 \
  --recon-num-samples 5000 \
  --understanding-train-samples 50000 \
  --understanding-val-samples 5000 \
  --knn-k 1 5 10 20 \
  --linear-epochs 20 \
  --output benchmark_stage1_quick.json
```

### 3. Recommended Pipeline (HF datasets, copy-paste)

Run reconstruction + understanding together for a single checkpoint:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/benchmark_stage1_quick.py \
  --model srae::configs/stage1/training/SRAE-B_decB.yaml::ckpts/stage1/<srae_exp>/checkpoints/ep-0000010.pt \
  --use-hf \
  --hf-train-path /path/to/hf_imagenet \
  --hf-val-path /path/to/hf_imagenet \
  --hf-train-split train \
  --hf-val-split validation \
  --num-classes 1000 \
  --recon-num-samples -1 \
  --understanding-train-samples 50000 \
  --understanding-val-samples 5000 \
  --knn-k 1 5 10 20 \
  --linear-epochs 20 \
  --output srae_stage1_eval.json
```

Run generation sampling for Stage-2:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/sample_ddp.py \
  --config configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B.yaml \
  --sample-dir samples \
  --num-fid-samples 50000 \
  --per-proc-batch-size 125 \
  --label-sampling equal \
  --precision fp32
```

Important:

- Keep `--num-classes` consistent with dataset labels (ImageNet-1K uses `1000`).
- `--recon-num-samples -1` means full-split reconstruction evaluation.

### 4. FID with ADM evaluator

```bash
git clone https://github.com/openai/guided-diffusion.git
cd guided-diffusion/evaluation

conda create -n adm-fid python=3.10 -y
conda activate adm-fid
pip install 'tensorflow[and-cuda]'==2.19 scipy requests tqdm

wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
python evaluator.py VIRTUAL_imagenet256_labeled.npz /path/to/samples.npz
```

## Utility Commands

### Calculate latent statistics

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/calculate_stat.py \
  --config configs/stage1/pretrained/DINOv2-B.yaml \
  --data-path /path/to/imagenet/train \
  --sample-dir stats \
  --per-proc-batch-size 256 \
  --image-size 256 \
  --precision fp32
```

Note: `calculate_stat.py` currently reads ImageFolder input (`--data-path`).

### Pack image folders into `.npz`

```bash
python pack_images.py <sample_dir> [image_size] [save_dir]
```

### Build dummy dataset for quick checks

```bash
python scripts/create_dummy_data.py \
  --num-samples 100 \
  --image-size 256 \
  --num-classes 10 \
  --output-dir ./dummy_data
```

### SRAE sanity tests

```bash
python scripts/test_srae_structure.py
python scripts/test_srae_model.py
```

## Config References

- Stage-1 pretrained configs: `configs/stage1/pretrained/`
- Stage-1 training configs (RAE + SRAE): `configs/stage1/training/`
- Stage-2 training configs: `configs/stage2/training/`
- Stage-2 sampling configs: `configs/stage2/sampling/`

### Baseline Presets (Encoder Backbone)

- DINOv2 (existing):
  - `configs/stage1/training/DINOv2-B_decB.yaml`
  - `configs/stage1/training/DINOv2-B_decXL.yaml`
  - `configs/stage2/training/ImageNet256/DiTDH-S_DINOv2-B.yaml`
  - `configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B.yaml`
- SigLIP2 (new):
  - `configs/stage1/training/SigLIP2-B_decB.yaml`
  - `configs/stage1/training/SigLIP2-B_decXL.yaml`
  - `configs/stage2/training/ImageNet256/DiTDH-S_SigLIP2-B.yaml`
  - `configs/stage2/training/ImageNet256/DiTDH-XL_SigLIP2-B.yaml`
- MAE (new):
  - `configs/stage1/training/MAE-B_decB.yaml`
  - `configs/stage1/training/MAE-B_decXL.yaml`
  - `configs/stage2/training/ImageNet256/DiTDH-S_MAE-B.yaml`
  - `configs/stage2/training/ImageNet256/DiTDH-XL_MAE-B.yaml`

For extra details, also see:

- `SRAE_README.md`
- `EVAL_README.md`

## Acknowledgement

This codebase builds on:

- [SiT](https://github.com/willisma/sit)
- [DDT](https://github.com/MCG-NJU/DDT)
- [LightningDiT](https://github.com/hustvl/LightningDiT/)
- [MAE](https://github.com/facebookresearch/mae)
