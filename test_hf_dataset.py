"""
Test script for HuggingFace dataset integration.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import HFImageNetDataset, load_dataset_from_hf
from torchvision import transforms

# Test loading HuggingFace dataset with a public dataset (CIFAR-10 as example)
print("Testing HuggingFace dataset loading...")
print("Using CIFAR-10 as a public dataset example (similar structure to ImageNet)")

try:
    # Try to load CIFAR-10 as a public dataset with similar structure
    # Use streaming mode to avoid downloading the entire dataset
    from datasets import load_dataset
    dataset = load_dataset(
        "cifar10", 
        split="train",
        streaming=True
    )
    
    # Take only first 10 samples for testing
    dataset = dataset.take(10)
    
    print(f"Dataset loaded successfully (streaming mode)!")
    
    # Convert to list for indexing (only for testing)
    print("Converting to list for testing...")
    samples = list(dataset)
    print(f"Total samples: {len(samples)}")
    
    # Check dataset structure
    sample = samples[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image type: {type(sample['img'])}")
    print(f"Label: {sample['label']}")
    
    # Create a list-based dataset
    class SimpleDataset:
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            item = self.data[idx]
            return {'image': item['img'], 'label': item['label']}
    
    # Test transform
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    
    # Create wrapped dataset
    wrapped_dataset = SimpleDataset(samples)
    
    # Create PyTorch dataset
    hf_torch_dataset = HFImageNetDataset(wrapped_dataset, transform=transform)
    print(f"PyTorch dataset created! Length: {len(hf_torch_dataset)}")
    
    # Test getting an item
    image, label = hf_torch_dataset[0]
    print(f"Sample image shape: {image.shape}, label: {label}")
    print("✓ HuggingFace dataset integration test passed!")
    
    # Test with ImageNet-style dataset config (if available)
    print("\nNote: For ImageNet-1k, you need:")
    print("  1. HuggingFace account with access to imagenet-1k dataset")
    print("  2. Set HF_TOKEN environment variable or pass token parameter")
    print("  3. Use: load_dataset_from_hf('imagenet-1k', split='train', token=your_token)")
    
except Exception as e:
    print(f"✗ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
