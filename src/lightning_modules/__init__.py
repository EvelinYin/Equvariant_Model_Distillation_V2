from .parallel_distillation_module import StudentParallelLayerLightningModule
from src.config import Config
from typing import Union
import torch.nn as nn

# TODO: add other modules
_MODULES = {
    "parallel_distillation": StudentParallelLayerLightningModule,
}

def available_models():
    return list(_MODULES.keys())

def get_lightining_modules(strategy: str, config: Config,
                           model: nn.Module, teacher_model: nn.Module = None, 
                           ):
    if strategy not in _MODULES:
        raise ValueError(f"Model '{strategy}' not found. Available: {available_models()}")
    
    
    if strategy == 'parallel_distillation':
        params = {
            'model': model,
            'train_config': config.student_train,
            'teacher_model': teacher_model,
            'num_classes': config.data.num_classes,
            'parallel_layer_distillation_config': config.parallel_layer_distillation,
        }
    else:
        breakpoint()
    
    # Initialize the specific model class with kwargs
    return _MODULES[strategy](**params)