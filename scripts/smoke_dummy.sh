#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DUMMY_DIR="${REPO_DIR}/dummy_data"

if [[ ! -d "${DUMMY_DIR}/dummy_imagenet" ]]; then
  python "${REPO_DIR}/scripts/create_dummy_data.py" \
    --num-samples 8 \
    --image-size 32 \
    --num-classes 2 \
    --output-dir "${DUMMY_DIR}"
fi

REPO_DIR="${REPO_DIR}" python - <<'PY'
import os
import sys

repo_dir = os.environ["REPO_DIR"]
sys.path.insert(0, os.path.join(repo_dir, "src"))

from datasets import load_from_disk
from dataset import HFImageNetDataset
from torch.utils.data import DataLoader
from torchvision import transforms

dummy_path = os.path.join(repo_dir, "dummy_data", "dummy_imagenet")
dataset = load_from_disk(dummy_path)["train"]
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

wrapped = HFImageNetDataset(dataset, transform=transform)
loader = DataLoader(wrapped, batch_size=2, shuffle=False, num_workers=0)
images, labels = next(iter(loader))
print(f"Loaded batch: images={tuple(images.shape)}, labels={tuple(labels.shape)}")
PY
