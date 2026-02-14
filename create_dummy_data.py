"""
Create dummy ImageNet-like data for testing.
"""

import os
import argparse
from PIL import Image
import random


def create_dummy_imagenet(output_dir, num_train=100, num_val=20, image_size=256, num_classes=10):
    """
    Create dummy ImageNet-like dataset structure.
    
    Args:
        output_dir: Output directory
        num_train: Number of training images per class
        num_val: Number of validation images per class
        image_size: Image size
        num_classes: Number of classes
    """
    splits = {
        'train': num_train,
        'val': num_val
    }
    
    for split, num_images in splits.items():
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        for class_idx in range(num_classes):
            class_dir = os.path.join(split_dir, f"class_{class_idx:03d}")
            os.makedirs(class_dir, exist_ok=True)
            
            for img_idx in range(num_images):
                # Create random RGB image
                img = Image.new('RGB', (image_size, image_size), color=(
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                ))
                
                # Save image
                img_path = os.path.join(class_dir, f"img_{img_idx:04d}.png")
                img.save(img_path)
    
    print(f"Dummy dataset created at: {output_dir}")
    print(f"  Train: {num_classes} classes x {num_train} images = {num_classes * num_train} images")
    print(f"  Val: {num_classes} classes x {num_val} images = {num_classes * num_val} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dummy ImageNet-like dataset")
    parser.add_argument("--output-dir", type=str, default="./dummy_data", help="Output directory")
    parser.add_argument("--num-train", type=int, default=10, help="Number of training images per class")
    parser.add_argument("--num-val", type=int, default=5, help="Number of validation images per class")
    parser.add_argument("--image-size", type=int, default=256, help="Image size")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of classes")
    args = parser.parse_args()
    
    create_dummy_imagenet(
        args.output_dir,
        args.num_train,
        args.num_val,
        args.image_size,
        args.num_classes
    )
