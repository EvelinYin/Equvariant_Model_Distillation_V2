import torch
import torch.nn as nn
from torchvision import datasets, transforms
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ViT.student_model import ViT_FlippingInvStudentModel
from train_lightning import get_student_model, get_data_module
from config import Config, DataConfig
from tqdm import tqdm

def get_all_cls_token(model, dataloader, device='cuda'):
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

            cls_token = model(images)
            cls_tokens.append(cls_token.detach().cpu())
            
                

    all_cls_tokens = torch.cat(cls_tokens, dim=0)
    
    return all_cls_tokens


if __name__ == "__main__":
    
    batch_size = 512
    config = Config(data=DataConfig(batch_size=batch_size))
    data_module = get_data_module(config)
    data_module.setup()
    dataloader = data_module.train_dataloader()

    
    depth = 12
    img_size = 224
    nheads = 12
    embed_dim = 768
    num_classes = 100
    patch_size = 16
    
    student = ViT_FlippingInvStudentModel(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=3,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            n_heads=nheads,
            mlp_ratio=4,
            # pos_embed="None-equ",
            pos_embed="SymmetricPosEmbed",
            attention_per_channel=True,
            group_attn_channel_pooling=False,
            linear_pooling=True
        )
        
    student = student.to(torch.float64)
    # Load pretrained weights if available
    student_checkpoint = torch.load('/home/yin178/Equvariant_Model_Distillation/outputs/CIFAR100/pretrained_ViT/student/initialization/real_zero_init.ckpt')
    # student_checkpoint = torch.load('/home/yin178/Equvariant_Model_Distillation/outputs/CIFAR100/pretrained_ViT/student/initialization/nonequ_pos_embed_real_zero_init.ckpt')
    
    student_checkpoint['state_dict'] = {k: v for k, v in student_checkpoint['state_dict'].items() if k not in ['non_equ_pos_embed', 'add_pos_embed.pos_embed_left', 'add_pos_embed.cls_pos_half']}
    # student_checkpoint['state_dict'] = {k: v for k, v in student_checkpoint['state_dict'].items()}
    
    student.load_state_dict(student_checkpoint['state_dict'], strict=False)
    student.eval()
    
    
    
    # layer_idx = 'model.classifier'

    # Get average cls_token
    avg_cls_token = get_all_cls_token(student, dataloader, device='cuda')
    torch.save(avg_cls_token, "./outputs/CIFAR100/cls_features/cls_features.pt")
    breakpoint()