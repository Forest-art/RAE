"""
Example script to save HuggingFace dataset to local disk.
This is useful for preparing datasets on a machine with internet access,
then using them on a machine without internet access.
"""

import argparse
from datasets import load_dataset


def save_imagenet_to_disk(output_dir, dataset_name="imagenet-1k", token=None):
    """
    Save ImageNet dataset to local disk.
    
    Args:
        output_dir: Directory to save the dataset
        dataset_name: Name of the dataset on HuggingFace
        token: HuggingFace API token for gated datasets
    """
    print(f"Loading dataset: {dataset_name}")
    
    # Load full dataset (both train and validation splits)
    dataset = load_dataset(dataset_name, token=token)
    
    print(f"Dataset loaded!")
    print(f"Splits: {list(dataset.keys())}")
    for split in dataset.keys():
        print(f"  {split}: {len(dataset[split])} samples")
    
    # Save to disk
    print(f"Saving to: {output_dir}")
    dataset.save_to_disk(output_dir)
    print("Done!")
    
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save HuggingFace dataset to local disk")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory to save dataset")
    parser.add_argument("--dataset-name", type=str, default="imagenet-1k", help="HuggingFace dataset name")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace API token")
    args = parser.parse_args()
    
    save_imagenet_to_disk(args.output_dir, args.dataset_name, args.token)
