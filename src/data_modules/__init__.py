from .cifar100_data_module import CIFAR100DataModule
from .mnist_data_module import MNISTDataModule
from src.config import DataConfig

# 1. Map string names to the class constructors
_DATA_MODULES = {
    'mnist': MNISTDataModule,
    'cifar100': CIFAR100DataModule,
}

def available_datamodules():
    return list(_DATA_MODULES.keys())

def get_datamodule(config: DataConfig):
    """
    Factory to initialize a LightningDataModule.
    
    Args:
        name (str): The key (e.g., 'mnist', 'cifar100')
        **kwargs: Arguments passed to the DataModule (batch_size, data_dir, etc.)
    """
    name = config.dataset_name.lower()
    if name not in _DATA_MODULES:
        raise ValueError(f"DataModule '{name}' not found. Available: {available_datamodules()}")
    
    if config.imagnet_normalization:
        means = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        return _DATA_MODULES[name](mean=means, std=std, config=config)
    
    # Instantiate the class with the arguments
    datamodule_class = _DATA_MODULES[name]
    return datamodule_class(config=config)