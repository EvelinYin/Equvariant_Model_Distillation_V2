import torch
import os
from src.config import Config
import collections.abc
from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse

def to_2tuple(x):
    return _ntuple(2)(x)


def setup_training_directories(config: Config):
    """Create necessary directories for checkpoints and outputs"""
    os.makedirs(config.logging.outputs_dir, exist_ok=True)


def get_BNC_features(features) -> torch.Tensor:
    final_features = None
    if features.ndim == 5:
        # B, 2, C, H, W
        C = features.size(1) * features.size(2)
        final_features = features.view(features.size(0), C, features.size(3), features.size(4))
        return final_features.flatten(-2).permute(0,2,1)  # B, H*W, C

    elif features.ndim == 4:
        # B,2C,H,W
        return features.flatten(-2).permute(0,2,1)  # B, H*W, C
        
    elif features.ndim == 3:
        return features  # B, N, C
    

def load_model_checkpoint(model: torch.nn.Module, ckpt_path: str):
    """Load model weights from checkpoint"""
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded model weights from {ckpt_path}")
        return model
    except Exception as e:
        breakpoint()