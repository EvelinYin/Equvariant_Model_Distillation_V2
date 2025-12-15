import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.config import DataConfig
from .base_data_module import BaseDataModule

class MNISTDataModule(BaseDataModule):
    """Lightning DataModule for MNIST"""
    
    def __init__(self, mean: tuple = (0.1307,), std: tuple = (0.3081,) , config: DataConfig = None):
        """
        Args:
            config: DataConfig object containing data configuration
        """
        super().__init__(config)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        
    def setup(self, stage: str = None):
        """Setup datasets"""

        
        if stage == "fit" or stage is None:
            self.train_dataset = datasets.MNIST(
                f'{self.config.data_dir}/MNIST',
                train=True,
                download=True,
                transform=self.transform
            )
            self.val_dataset = datasets.MNIST(
                f'{self.config.data_dir}/MNIST',
                train=False,
                transform=self.transform
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = datasets.MNIST(
                f'{self.config.data_dir}/MNIST',
                train=False,
                transform=self.transform
            )


