"""
Test loading dataset from local disk.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import HFImageNetDataset, load_dataset_from_hf
from torchvision import transforms
from datasets import Dataset, DatasetDict
from PIL import Image
import numpy as np

# Create a dummy dataset and save to disk
print("Creating dummy dataset...")

# Create dummy images
def create_dummy_image(idx):
    """Create a dummy RGB image"""
    img_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    return Image.fromarray(img_array)

# Create dataset
data = {
    'image': [create_dummy_image(i) for i in range(20)],
    'label': [i % 10 for i in range(20)]
}

dummy_dataset = Dataset.from_dict(data)
dummy_dataset_dict = DatasetDict({'train': dummy_dataset, 'validation': dummy_dataset})

# Save to disk
output_dir = "./dummy_hf_dataset"
print(f"Saving dummy dataset to: {output_dir}")
dummy_dataset_dict.save_to_disk(output_dir)

# Test loading from disk
print("\nTesting load_from_disk...")
try:
    # Load train split
    train_dataset = load_dataset_from_hf(
        load_from_disk_path=output_dir,
        split='train'
    )
    print(f"Train dataset loaded! Length: {len(train_dataset)}")
    
    # Load validation split
    val_dataset = load_dataset_from_hf(
        load_from_disk_path=output_dir,
        split='validation'
    )
    print(f"Validation dataset loaded! Length: {len(val_dataset)}")
    
    # Test transform
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    
    # Create PyTorch dataset
    hf_torch_dataset = HFImageNetDataset(train_dataset, transform=transform)
    print(f"PyTorch dataset created! Length: {len(hf_torch_dataset)}")
    
    # Test getting an item
    image, label = hf_torch_dataset[0]
    print(f"Sample image shape: {image.shape}, label: {label}")
    print("✓ load_from_disk test passed!")
    
    # Cleanup
    import shutil
    shutil.rmtree(output_dir)
    print(f"\nCleaned up: {output_dir}")
    
except Exception as e:
    print(f"✗ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
