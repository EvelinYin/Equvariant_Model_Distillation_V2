from .ViT.equ_vit import EquViT
from .ViT.pretrained_HF import PretrainedViT
from .ResNet50.equ_resnet import EquResNet
from src.config import StudentModelConfig, TeacherModelConfig
from typing import Union


# TODO: add cnn
_MODELS = {
    # "base_vit": BaseViT,
    "pretrained_ViT": PretrainedViT,
    "equ_vit": EquViT,
    "equ_resnet50": EquResNet,
}

def available_models():
    return list(_MODELS.keys())

def get_model(name: str, config: Union[StudentModelConfig, TeacherModelConfig]):
    if name not in _MODELS:
        raise ValueError(f"Model '{name}' not found. Available: {available_models()}")
    
    # TODO: add resnet configs
    
    if name == "pretrained_ViT":
        params = config.pretrained_vit_config
    elif "vit" in name:
        params = config.vit_config
    elif "cnn" in name:
        params = config.cnn_config
    
    # Initialize the specific model class with kwargs
    return _MODELS[name](**params.__dict__)