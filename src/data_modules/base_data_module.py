import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.config import DataConfig


class BaseDataModule(pl.LightningDataModule):
    """Base Lightning DataModule"""
    
    def __init__(self, config: DataConfig):
        """
        Args:
            config: DataConfig object containing data configuration
        """
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self, stage: str = None):
        """Setup datasets"""
        raise NotImplementedError("Child class must implement setup()")
    
    def train_dataloader(self):
        """Return training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
    
    def val_dataloader(self):
        """Return validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
    
    def test_dataloader(self):
        """Return test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True
        )