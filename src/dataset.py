"""
Dataset classes for RAE training.
Supports both local ImageFolder and HuggingFace datasets.
"""

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class HFImageNetDataset(Dataset):
    """
    HuggingFace ImageNet dataset wrapper.
    
    Args:
        hf_dataset: HuggingFace dataset with 'image' and 'label' columns
        transform: Optional transform to be applied on the image
    """
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            image = item['image']  # get PIL Image
            label = item['label']
            
            # ImageNet -> transforms for RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            if self.transform:
                image = self.transform(image)
                
            return image, label

        except Exception as e:
            print(f"Error processing sample at index {idx}: {e}")
            # Skip this sample and recursively call the next one
            return self.__getitem__((idx + 1) % len(self.dataset))


def load_dataset_from_hf(dataset_name="imagenet-1k", split="train", cache_dir=None, token=None, load_from_disk_path=None):
    """
    Load ImageNet dataset from HuggingFace or from local disk.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace (default: imagenet-1k)
        split: Dataset split to load (train/validation)
        cache_dir: Optional cache directory for downloaded data
        token: HuggingFace API token for gated datasets (e.g., ImageNet)
        load_from_disk_path: Path to locally saved dataset (using datasets.save_to_disk)
        
    Returns:
        HuggingFace dataset object
    """
    from datasets import load_dataset, load_from_disk
    
    if load_from_disk_path is not None:
        # Load from local disk
        dataset = load_from_disk(load_from_disk_path)
        # Handle split if the loaded dataset is a DatasetDict
        if hasattr(dataset, 'keys') and split in dataset:
            dataset = dataset[split]
        return dataset
    else:
        # Load from HuggingFace Hub
        dataset = load_dataset(
            dataset_name,
            split=split,
            cache_dir=cache_dir,
            token=token
        )
        return dataset
