import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.config import DataConfig
from .base_data_module import BaseDataModule
import torchvision.transforms.functional as F
from torchvision.transforms import RandomErasing, RandAugment, ColorJitter, InterpolationMode


class CIFAR100DataModule(BaseDataModule):
    """Lightning DataModule for CIFAR100"""
    
    def __init__(self, mean: tuple = (0.5071, 0.4867, 0.4408), std: tuple = (0.2675, 0.2565, 0.2761), config: DataConfig = None):
        """
        Args:
            config: DataConfig object containing data configuration
        """
        super().__init__(config)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        
        # Data augmentation for training
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224),
                              interpolation=InterpolationMode.BICUBIC),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),  # ColorJitter
            # transforms.RandomRotation(90),
            transforms.Lambda(lambda img: F.rotate(img, angle=90)),
            RandAugment(
            num_ops=2,  # timm typically uses 2 ops for 'rand' variant
            magnitude=9,
            interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
            RandomErasing(
            p=0.25,
            scale=(0.02, 0.33),  # Default timm values
            ratio=(0.3, 3.3),    # Default timm values
            value='random',      # 'pixel' mode in timm = 'random' in torchvision
            inplace=False
            ),
        ])
        
        # No augmentation for validation/test
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                # mean=(0.485, 0.456, 0.406),
                # std=(0.229, 0.224, 0.225),
                mean=mean,
                std=std,
            ),
        ])
        
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = datasets.CIFAR100(
                self.config.data_dir, train=True, transform=self.train_transform
            )
            self.val_dataset = datasets.CIFAR100(
                self.config.data_dir, train=False, transform=self.test_transform
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = datasets.CIFAR100(
                self.config.data_dir, train=False, transform=self.test_transform
            )