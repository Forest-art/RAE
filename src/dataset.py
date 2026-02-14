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


def load_dataset_from_hf(dataset_name="imagenet-1k", split="train", cache_dir=None):
    """
    Load ImageNet dataset from HuggingFace.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace (default: imagenet-1k)
        split: Dataset split to load (train/validation)
        cache_dir: Optional cache directory for downloaded data
        
    Returns:
        HuggingFace dataset object
    """
    from datasets import load_dataset
    
    dataset = load_dataset(
        dataset_name,
        split=split,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    return dataset
