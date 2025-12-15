import torch
import torch.nn as nn
from torchvision import datasets, transforms
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_lightning import get_pretrained_teacher_model, get_data_module
from config import Config, DataConfig
from tqdm import tqdm

def get_average_cls_token(model, dataloader, layer_idx, device='cuda'):
    """
    Accumulate cls_token from model inference and return its average.
    
    Args:
        model: Pretrained model
        dataloader: DataLoader for CIFAR100
        device: Device to run inference on
    
    Returns:
        Average cls_token across all batches
    """
    model.eval()
    model.to(device)
    
    cls_tokens = []
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader):
            images = images.to(device)
            
            def hook_fn(module, input):
                cls_tokens.append(input[0].detach().cpu())
            
            handle = model.get_submodule(layer_idx).register_forward_pre_hook(hook_fn)
            
            _ = model(images)
            
            handle.remove()
            
            
                
    # Concatenate all cls_tokens and compute mean
    all_cls_tokens = torch.cat(cls_tokens, dim=0)
    avg_cls_token = all_cls_tokens.mean(dim=0)

    return avg_cls_token


if __name__ == "__main__":
    
    batch_size = 1024
    config = Config(data=DataConfig(batch_size=batch_size))
    data_module = get_data_module(config)
    data_module.setup()
    dataloader = data_module.train_dataloader()


    teacher = get_pretrained_teacher_model(model_name="google/vit-base-patch16-224-in21k", num_classes=100, device='cpu')
        
    teacher_checkpoint = torch.load("/home/yin178/Equvariant_Model_Distillation/outputs/CIFAR100/pretrained_ViT/teacher/google/vit-base-patch16-224/best.ckpt")
    
    teacher_checkpoint['state_dict'] = {k.replace('model.m', 'm'): v for k, v in teacher_checkpoint['state_dict'].items()}
    
    # breakpoint()
    teacher.load_state_dict(teacher_checkpoint['state_dict'])
    teacher.double()
    
    
    
    layer_idx = 'model.classifier'

    # Get average cls_token
    avg_cls_token = get_average_cls_token(teacher, dataloader, layer_idx, device='cuda')
    print("Average CLS Token:", avg_cls_token)
    torch.save(avg_cls_token[0], "./outputs/CIFAR100/cls_features/cls_features.pt")
    breakpoint()