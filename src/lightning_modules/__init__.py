from .parallel_distillation_module import StudentParallelLayerLightningModule
from .training_module import LightningTrainingModule
from src.config import Config
from typing import Union
import torch.nn as nn

# TODO: add other modules
_MODULES = {
    "parallel_distillation": StudentParallelLayerLightningModule,
    "equ_naive_distillation": LightningTrainingModule,  
    "equ_train_on_GT": LightningTrainingModule,
    "non_equ_train_on_GT": LightningTrainingModule,
}

def available_models():
    return list(_MODULES.keys())

def get_lightining_modules(strategy: str, config: Config,
                           model: nn.Module, teacher_model: nn.Module = None, 
                           ):
    if strategy not in _MODULES:
        raise ValueError(f"Model '{strategy}' not found. Available: {available_models()}")
    
    params = {
        'model': model,
        'num_classes': config.data.num_classes
    }
    
    if strategy == 'parallel_distillation':
        params['train_config'] = config.student_train
        params['teacher_model'] = teacher_model
        params['parallel_layer_distillation_config'] = config.parallel_layer_distillation

    elif strategy == 'equ_naive_distillation':
        params['train_config'] = config.student_train
        params['teacher_model'] = teacher_model 

        
    elif strategy == 'equ_train_on_GT':
        params['train_config'] = config.student_train

    elif strategy == 'non_equ_train_on_GT':
        params['train_config'] = config.teacher_train
        
    else:
        breakpoint()
    
    # Initialize the specific model class with kwargs
    return _MODULES[strategy](**params)