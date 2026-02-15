"""
Create dummy ImageNet-like dataset for testing S-RAE training.
Generates HF datasets format that can be loaded with load_from_disk.
"""

import os
import random
from PIL import Image
from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage
import argparse


def create_dummy_dataset(num_samples=100, image_size=256, num_classes=10, output_dir="./dummy_data"):
    """Create a dummy ImageNet-style dataset in HF format."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create train split
    print(f"Creating {num_samples} training samples...")
    train_images = []
    train_labels = []
    
    for i in range(num_samples):
        # Create random RGB image using PIL
        img = Image.new('RGB', (image_size, image_size))
        pixels = img.load()
        for y in range(image_size):
            for x in range(image_size):
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                pixels[x, y] = (r, g, b)
        
        train_images.append(img)
        train_labels.append(i % num_classes)
    
    # Create validation split (smaller)
    val_num = max(1, num_samples // 5)
    print(f"Creating {val_num} validation samples...")
    val_images = []
    val_labels = []
    
    for i in range(val_num):
        img = Image.new('RGB', (image_size, image_size))
        pixels = img.load()
        for y in range(image_size):
            for x in range(image_size):
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                pixels[x, y] = (r, g, b)
        
        val_images.append(img)
        val_labels.append(i % num_classes)
    
    # Create HF datasets
    features = Features({
        "image": HFImage(),
        "label": Value("int64"),
    })
    
    train_dataset = Dataset.from_dict({
        "image": train_images,
        "label": train_labels,
    }, features=features)
    
    val_dataset = Dataset.from_dict({
        "image": val_images,
        "label": val_labels,
    }, features=features)
    
    # Create DatasetDict
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
    })
    
    # Save to disk
    output_path = os.path.join(output_dir, "dummy_imagenet")
    dataset_dict.save_to_disk(output_path)
    print(f"Dataset saved to: {output_path}")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=100, help="Number of training samples")
    parser.add_argument("--image-size", type=int, default=256, help="Image size")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--output-dir", type=str, default="./dummy_data", help="Output directory")
    args = parser.parse_args()
    
    create_dummy_dataset(
        num_samples=args.num_samples,
        image_size=args.image_size,
        num_classes=args.num_classes,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
