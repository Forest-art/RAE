# S-RAE (Semantic-Guided Regularized Autoencoder)

S-RAE 是一个统一的自监督学习框架，结合了 DINOv2 的语义理解能力和 MAE 的图像生成能力。

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                        S-RAE                                 │
├─────────────────────────────────────────────────────────────┤
│  Teacher (Frozen DINOv2)          Student (Trainable)       │
│  ┌──────────────────────┐        ┌──────────────────────┐  │
│  │ Full Image           │        │ Masked Image (75%)   │  │
│  │ ↓                    │        │ ↓                    │  │
│  │ DINOv2 Encoder       │        │ DINOv2 Encoder       │  │
│  │ ↓                    │        │ ↓                    │  │
│  │ Global Feature       │◄───────│ Projector (MLP)      │  │
│  │ (CLS token)          │        │   (Alignment Loss)   │  │
│  └──────────────────────┘        └──────────┬───────────┘  │
│                                             │               │
│                              ┌──────────────▼───────────┐  │
│                              │   Bottleneck (z)         │  │
│                              │   Linear + L2 Norm       │  │
│                              │   + L2 Regularization    │  │
│                              └──────────────┬───────────┘  │
│                                             │               │
│                              ┌──────────────▼───────────┐  │
│                              │   MAE Decoder            │  │
│                              │   (Reconstruction)       │  │
│                              └──────────────┬───────────┘  │
│                                             │               │
│                              Reconstructed Image           │
└─────────────────────────────────────────────────────────────┘
```

## 损失函数

1. **Reconstruction Loss** (`loss_rec`): MSE on masked patches
2. **Alignment Loss** (`loss_align`): Cosine similarity between Student projector and Teacher
3. **Regularization Loss** (`loss_reg`): L2 penalty on bottleneck latent

## 使用方法

### 1. 训练 S-RAE

使用专用训练脚本：

```bash
cd RAE/src

# 单卡训练
python train_srae.py \
    --config ../configs/stage1/training/SRAE-B_decB.yaml \
    --data-path /path/to/imagenet/train \
    --results-dir ../experiments/srae \
    --image-size 256

# 多卡训练
torchrun --nproc_per_node=8 train_srae.py \
    --config ../configs/stage1/training/SRAE-B_decB.yaml \
    --data-path /path/to/imagenet/train \
    --results-dir ../experiments/srae \
    --image-size 256
```

### 2. 使用 HuggingFace 数据集

```bash
python train_srae.py \
    --config ../configs/stage1/training/SRAE-B_decB.yaml \
    --use-hf \
    --hf-dataset-name imagenet-1k \
    --hf-split train \
    --use-hf-eval \
    --hf-eval-split validation
```

### 3. 配置文件

主要配置参数 (`configs/stage1/training/SRAE-B_decB.yaml`):

```yaml
stage_1:
  target: stage1.SRAE
  params:
    dinov2_model_name: "facebook/dinov2-base"
    mask_ratio: 0.75
    bottleneck_dim: 768
    use_l2_norm: true
    projector_hidden_dim: 4096
    decoder_num_layers: 8
    decoder_dim: 512
    loss_rec_weight: 1.0
    loss_align_weight: 0.1
    loss_reg_weight: 0.01
```

### 4. 与原始 RAE 兼容

S-RAE 也可以通过原始训练脚本 `train_stage1.py` 训练（仅使用重建损失）：

```bash
python train_stage1.py \
    --config ../configs/stage1/training/SRAE-B_decB.yaml \
    --data-path /path/to/imagenet/train
```

## 关键特性

### Masking 策略
- 随机 masking，默认 75% 的 patches 被 mask
- 使用 DINOv2 的 patch embedding 和 position embedding
- Student 只处理可见 patches（高效）

### Bottleneck 设计
- Linear projection: 768 → 768 (可配置)
- L2 归一化 (可选) 或 LayerNorm
- L2 正则化损失保持空间紧凑

### Projector 设计
- 2层 MLP: 768 → 4096 → 768
- GELU 激活 + LayerNorm
- 仅用于对齐损失，不用于解码

## 性能优化建议

1. **batch size**: 使用较大的 batch (256-512) 以获得稳定的对比学习
2. **mask ratio**: 默认 75%，可根据需要调整
3. **loss weights**: 
   - 初期：`loss_align_weight=0.1` 较高，加速语义学习
   - 后期：可降低到 0.05，侧重重建质量
4. **warmup**: 建议 5-10 epochs warmup

## 与 RAE 的区别

| 特性 | RAE | S-RAE |
|------|-----|-------|
| Encoder | DINOv2 (冻结) | DINOv2-based (可训练) |
| Masking | 无 | 75% 随机 mask |
| Latent | 连续 + noise | 连续 + L2 norm |
| 语义引导 | 无 | Teacher-Student 对齐 |
| 损失 | L1 + LPIPS + GAN | MSE + Align + L2 |
| 重建质量 | 高 | 中高 (平衡语义) |
| 语义特征 | 一般 | 优秀 |

## 参考文献

- DINOv2: "DINOv2: Learning Robust Visual Features without Supervision"
- MAE: "Masked Autoencoders Are Scalable Vision Learners"
