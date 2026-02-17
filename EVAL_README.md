# Understanding Evaluation

Evaluation tools for measuring encoder representation quality.

## Supported Metrics

### 1. Linear Probing

Trains a linear classifier on frozen encoder features.

```bash
python eval_understanding.py \
    --config configs/stage1/training/SRAE-B_decB.yaml \
    --checkpoint ckpts/srae/ep-0000010.pt \
    --train-data-path data/imagenet/train \
    --data-path data/imagenet/val \
    --num-classes 1000 \
    --eval-linear \
    --linear-epochs 100 \
    --linear-lr 0.001 \
    --batch-size 256
```

**Output:**
- `top1`: Top-1 accuracy (%)
- `top5`: Top-5 accuracy (%)
- `train_acc`: Final training accuracy
- `best_val_acc`: Best validation accuracy during training

### 2. KNN Evaluation

Uses K-Nearest Neighbors on frozen features with cosine similarity.

```bash
python eval_understanding.py \
    --config configs/stage1/training/SRAE-B_decB.yaml \
    --checkpoint ckpts/srae/ep-0000010.pt \
    --train-data-path data/imagenet/train \
    --data-path data/imagenet/val \
    --num-classes 1000 \
    --eval-knn \
    --knn-k 1 5 10 20 100 200 \
    --batch-size 256
```

**Output:**
- For each k: `top1` and `top5` accuracy (%)

### 3. Run Both

```bash
python eval_understanding.py \
    --config configs/stage1/training/SRAE-B_decB.yaml \
    --checkpoint ckpts/srae/ep-0000010.pt \
    --train-data-path data/imagenet/train \
    --data-path data/imagenet/val \
    --num-classes 1000 \
    --eval-linear \
    --eval-knn \
    --output results_understanding.json
```

## HuggingFace Dataset

```bash
python eval_understanding.py \
    --config configs/stage1/training/SRAE-B_decB.yaml \
    --checkpoint ckpts/srae/ep-0000010.pt \
    --data-path /path/to/hf/dataset \
    --train-data-path /path/to/hf/dataset \
    --use-hf \
    --hf-split validation \
    --eval-linear \
    --eval-knn
```

## Implementation Details

### Feature Extraction

The evaluators automatically detect the encoder type:

- **RAE**: Uses `model.encode(images)` → returns latent
- **S-RAE**: Uses `model.forward_student(images, mask_ratio=0.0)` → pools patch features
- **Generic**: Falls back to `model(images)` and flattens output

### Linear Probing

- Optimizer: SGD with momentum=0.9
- Scheduler: Cosine Annealing
- Loss: Cross Entropy
- Features are frozen, only linear classifier is trained

### KNN

- Distance: Cosine similarity (features are L2-normalized)
- Voting: Majority vote among k nearest neighbors
- Memory efficient: Processes test set in batches

## Expected Performance

| Method | ImageNet-1K (Pre-trained DINOv2) |
|--------|----------------------------------|
| Linear Probe | ~80-82% Top-1 |
| KNN (k=20) | ~75-78% Top-1 |
| KNN (k=200) | ~76-79% Top-1 |

Note: S-RAE should improve over frozen DINOv2 due to end-to-end training.

## Multi-GPU Support

The evaluation script automatically uses all available GPUs:

```bash
torchrun --nproc_per_node=8 eval_understanding.py \
    --config ... \
    --checkpoint ... \
    --eval-linear
```

Features are extracted in parallel across GPUs.
