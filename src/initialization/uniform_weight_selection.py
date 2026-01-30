import argparse
import numpy as np
import torch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from src.utils import clean_state_dict
from src.models.ViT.pretrained_HF import PretrainedViT
from src.equ_lib.groups.rot90_group import Rot90Group
from src.equ_lib.groups.flipping_group import FlipGroup
from src.equ_lib.groups.rot45_group import Rot45Group


def conv_identity_weight(out_c, in_c, k=3):
    w = torch.zeros(out_c, in_c, k, k)
    for o in range(out_c):
        i = o % in_c
        w[o, i, k//2, k//2] = 1.0
    return w



def uniform_element_selection(wt, s_shape):
    assert wt.dim() == len(s_shape), "Tensors have different number of dimensions"
    ws = wt.clone()
    for dim in range(wt.dim()):
        assert wt.shape[dim] >= s_shape[dim], "Teacher's dimension should not be smaller than student's dimension"  # determine whether teacher is larger than student on this dimension
        if wt.shape[dim] % s_shape[dim] == 0:
            step = wt.shape[dim] // s_shape[dim]
            indices = torch.arange(s_shape[dim]) * step
        else:
            indices = torch.round(torch.linspace(0, wt.shape[dim]-1, s_shape[dim])).long()
        indices = indices.to(ws.device)
        ws = torch.index_select(ws, dim, indices)
    assert ws.shape == s_shape
    return ws



def pretrained_vit_initialization(teacher_model, teacher_ckpt_path, student_model, group, device=torch.device("cpu")):
     # Load teacher checkpoint
    if teacher_ckpt_path is not None:
        teacher_ckpt = torch.load(teacher_ckpt_path, map_location='cpu')
    
        # Handle both direct state_dict and nested checkpoints
        if isinstance(teacher_ckpt, dict) and 'state_dict' in teacher_ckpt:
            teacher_state_dict = teacher_ckpt['state_dict']
        else:
            teacher_state_dict = teacher_ckpt

        try:
            # Load teacher weights
            teacher_model.load_state_dict(teacher_state_dict, strict=False)
        except:
            cleaned_state_dict = clean_state_dict(teacher_state_dict)
            errors, unexpected = teacher_model.load_state_dict(cleaned_state_dict, strict=False)
            if unexpected:
                print(f"Ignored the following keys: {unexpected}")
                print(f"Loaded teacher model weights from {teacher_ckpt_path}")
        
    
    teacher_model = teacher_model.model.model
    
    teacher_model.eval().to(device)
    
    student_state_dict = student_model.state_dict()
    # new = student_state_dict.copy()
    
    num_elements = group.order
    
    
    # cls_token 
    student_model.cls_token.data = uniform_element_selection(teacher_model.vit.embeddings.cls_token.data, student_state_dict['cls_token'].shape)
    
    
    # pos embeddings
    # Get teacher positional embeddings
    teacher_pos_embed = teacher_model.vit.embeddings.position_embeddings.data  # Shape: (1, num_patches+1, embed_dim)

    # Check if student model has symmetric positional embeddings
    if hasattr(student_model.pos_embed, '_create_rotation_grid'):
        # Rotation90SymmetricPosEmbed or Rotation45SymmetricPosEmbed case
        # Extract spatial embeddings (excluding cls token)
        teacher_spatial_pos = teacher_pos_embed[:, 1:, :]  # (1, num_patches, embed_dim)
        B, N, C_teacher = teacher_spatial_pos.shape
        H_teacher = W_teacher = int(N ** 0.5)
        
        # Reshape to spatial grid
        teacher_grid = teacher_spatial_pos.reshape(1, H_teacher, W_teacher, C_teacher)
        
        if isinstance(student_model.pos_embed, type(student_model.pos_embed)) and 'Rotation90' in type(student_model.pos_embed).__name__:
            # Rotation90SymmetricPosEmbed: learn upper-right triangle
            target_shape = student_model.pos_embed.learnable_region.data.shape
            
            # Extract upper-right triangle region from teacher
            H_student = student_model.pos_embed.H
            teacher_triangle = []
            for i in range(H_student):
                for j in range(i, H_student):
                    teacher_triangle.append(teacher_grid[:, i, j, :])
            teacher_triangle = torch.stack(teacher_triangle, dim=1)  # (1, num_learnable, C_teacher)
            
            # Uniform selection
            selected_region = uniform_element_selection(teacher_triangle, target_shape)
            student_model.pos_embed.learnable_region.data = selected_region
        else:
            # Rotation45SymmetricPosEmbed: learn wedge region
            target_shape = student_model.pos_embed.learnable_wedge.data.shape
            
            # For simplicity, extract a wedge-like region from teacher
            # This is approximate since 45-degree rotations don't align perfectly with grid
            H_student = student_model.pos_embed.H
            num_wedge_points = target_shape[1]
            
            # Sample uniformly from teacher grid in a wedge-like pattern
            teacher_flat = teacher_grid.reshape(1, -1, C_teacher)
            selected_wedge = uniform_element_selection(teacher_flat[:, :num_wedge_points, :], target_shape)
            student_model.pos_embed.learnable_wedge.data = selected_wedge
        
        # Initialize CLS token positional embedding
        teacher_cls_pos = teacher_pos_embed[:, 0:1, :]  # (1, 1, embed_dim)
        target_cls_shape = (1, 1, student_model.pos_embed.C // group.order)
        selected_cls = uniform_element_selection(teacher_cls_pos, target_cls_shape)
        student_model.pos_embed.cls_pos_base.data = selected_cls
    else:
        # Standard non-equivariant positional embedding
        target_shape = student_model.pos_embed.pos_embed.data.shape
        selected_pos = uniform_element_selection(teacher_pos_embed, target_shape)
        student_model.pos_embed.pos_embed.data = selected_pos


    
    
    # Lifting layer
    H,W,_,_ = student_model.patch_embed.liftinglayer.kernel.weight.shape
    student_model.patch_embed.liftinglayer.kernel.weight.data = conv_identity_weight(H,W,k=3)
    student_model.patch_embed.liftinglayer.bias.data.zero_()
    

     # Patch embedding group conv layer ()
    # C_big = teacher_model.vit.embeddings.patch_embeddings.projection.out_channels
    # C_small = C_big // scale_factor
    
    W = teacher_model.vit.embeddings.patch_embeddings.projection.weight.data
    W = uniform_element_selection(W, student_model.patch_embed.grouplayer.kernel.weight.data[:,:,0].shape)
    student_model.patch_embed.grouplayer.kernel.weight.data = torch.zeros_like(student_model.patch_embed.grouplayer.kernel.weight.data)
    student_model.patch_embed.grouplayer.kernel.weight.data[:,:,0::2] = W.unsqueeze(2)
    
    student_model.patch_embed.grouplayer.bias.data = torch.zeros_like(student_model.patch_embed.grouplayer.bias.data)
    student_model.patch_embed.grouplayer.bias.data = uniform_element_selection(teacher_model.vit.embeddings.patch_embeddings.projection.bias.data,
                                                                                student_model.patch_embed.grouplayer.bias.data.shape)
    # breakpoint()
    # Transformer blocks
    for block_idx, block_big, block_small in zip(
        range(len(teacher_model.vit.encoder.layer)),
        teacher_model.vit.encoder.layer,
        student_model.blocks
    ):
        
        
        for g_i in range(num_elements):
            block_small.attn.shared_q.learnable_weights[g_i].data.zero_()
            block_small.attn.shared_k.learnable_weights[g_i].data.zero_()
            block_small.attn.shared_v.learnable_weights[g_i].data.zero_()
            block_small.attn.proj.learnable_weights[g_i].data.zero_()
            block_small.mlp.fc1.learnable_weights[g_i].data.zero_()
            block_small.mlp.fc2.learnable_weights[g_i].data.zero_()
            
            
            
        block_small.attn.shared_q.learnable_bias.data.zero_()
        block_small.attn.shared_k.learnable_bias.data.zero_()
        block_small.attn.shared_v.learnable_bias.data.zero_()
        
        block_small.attn.proj.learnable_bias.data.zero_()
        

        block_small.mlp.fc1.learnable_bias.data.zero_()
        block_small.mlp.fc2.learnable_bias.data.zero_()
    
    
        # For qkv
        block_small.attn.shared_q.learnable_weights[0].data = uniform_element_selection(
            block_big.attention.attention.query.weight.data, block_small.attn.shared_q.learnable_weights[0].data.shape)
        block_small.attn.shared_q.learnable_bias.data = uniform_element_selection(
            block_big.attention.attention.query.bias.data, block_small.attn.shared_q.learnable_bias.data.shape)
        
        block_small.attn.shared_k.learnable_weights[0].data = uniform_element_selection(                                                    
            block_big.attention.attention.key.weight.data, block_small.attn.shared_k.learnable_weights[0].data.shape)
        block_small.attn.shared_k.learnable_bias.data = uniform_element_selection(
            block_big.attention.attention.key.bias.data, block_small.attn.shared_k.learnable_bias.data.shape)
        
        block_small.attn.shared_v.learnable_weights[0].data = uniform_element_selection(                                                    
            block_big.attention.attention.value.weight.data, block_small.attn.shared_v.learnable_weights[0].data.shape)
        block_small.attn.shared_v.learnable_bias.data = uniform_element_selection(
            block_big.attention.attention.value.bias.data, block_small.attn.shared_v.learnable_bias.data.shape)
        
    
        # For attn proj
        block_small.attn.proj.learnable_weights[0].data = uniform_element_selection(
            block_big.attention.output.dense.weight.data, block_small.attn.proj.learnable_weights[0].data.shape)
        block_small.attn.proj.learnable_bias.data = uniform_element_selection(
            block_big.attention.output.dense.bias.data, block_small.attn.proj.learnable_bias.data.shape)
        
        
        # For mlp fc1 and fc2
        block_small.mlp.fc1.learnable_weights[0].data = uniform_element_selection(
            block_big.intermediate.dense.weight.data, block_small.mlp.fc1.learnable_weights[0].data.shape)
        block_small.mlp.fc1.learnable_bias.data = uniform_element_selection(
            block_big.intermediate.dense.bias.data, block_small.mlp.fc1.learnable_bias.data.shape)
        
        # breakpoint()
        block_small.mlp.fc2.learnable_weights[0].data = uniform_element_selection(
            block_big.output.dense.weight.data, block_small.mlp.fc2.learnable_weights[0].data.shape)
        block_small.mlp.fc2.learnable_bias.data = uniform_element_selection(
            block_big.output.dense.bias.data, block_small.mlp.fc2.learnable_bias.data.shape)
        
    
    
        # LayerNorms
        block_small.norm1.norm.weight.data = uniform_element_selection(
            block_big.layernorm_before.weight.data, block_small.norm1.norm.weight.data.shape)
        block_small.norm1.norm.bias.data = uniform_element_selection(
            block_big.layernorm_before.bias.data, block_small.norm1.norm.bias.data.shape)
        
        block_small.norm2.norm.weight.data = uniform_element_selection(
            block_big.layernorm_after.weight.data, block_small.norm2.norm.weight.data.shape)
        block_small.norm2.norm.bias.data = uniform_element_selection(
            block_big.layernorm_after.bias.data, block_small.norm2.norm.bias.data.shape)
        
    
    
    # Final LayerNorm
    student_model.norm.norm.weight.data = uniform_element_selection(
        teacher_model.vit.layernorm.weight.data, student_model.norm.norm.weight.data.shape)
    student_model.norm.norm.bias.data = uniform_element_selection(
        teacher_model.vit.layernorm.bias.data, student_model.norm.norm.bias.data.shape)
    
    
    # # Classifier head
    # student_model.head.weight.data = uniform_element_selection(
    #     teacher_model.classifier.weight.data, student_model.head.weight.data.shape)
    # student_model.head.bias.data = uniform_element_selection(
    #     teacher_model.classifier.bias.data, student_model.head.bias.data.shape)
    
    
    return student_model.state_dict()




def init_vit_tiny(teacher_model, student_model, device=torch.device("cpu")):
    teacher_state_dict = teacher_model.state_dict()
    student_state_dict = student_model.state_dict()
    
    weight_selection = {}
    for key in student_state_dict.keys():
        if "head" in key:
            continue
        weight_selection[key] = uniform_element_selection(teacher_state_dict[key], student_state_dict[key].shape)
    
    return weight_selection
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


    

def check_state_dicts_match(dict1, dict2):
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    if keys1 != keys2:
        missing_in_1 = keys2 - keys1
        missing_in_2 = keys1 - keys2
        print(f"Key mismatch:\nMissing in dict1: {missing_in_1}\nMissing in dict2: {missing_in_2}")
    for key in keys1:
        # if "patch_embed" in key:
            # continue
        try:
            if dict1[key].shape != dict2[key].shape:
                breakpoint()
                raise ValueError(f"Shape mismatch for key '{key}': {dict1[key].shape} vs {dict2[key].shape}")
        except:
            breakpoint()
    print("State dicts match: all keys and shapes are identical.")



    
    



if __name__ == "__main__":


    from src.models.ViT.equ_vit import EquViT
    from src.data_modules.cifar100_data_module import CIFAR100DataModule
    from src.config import DataConfig
    from tqdm import tqdm

        
    ################Configs for intialization#######################
    # teacher_model = PretrainedViT(model_name="google/vit-base-patch16-224", num_classes=100)
    teacher_model = PretrainedViT(model_name="WinKawaks/vit-small-patch16-224", num_classes=100)
    
    # teacher_ckpt_path = "/home/yin178/Equvariant_Model_Distillation/outputs/CIFAR100/ViT/teacher/pretrained_finetuned/epoch=07.ckpt"
    # teacher_ckpt_path = "/home/yin178/Equvariant_Model_Distillation/outputs/CIFAR100/pretrained_ViT/teacher/google/vit-base-patch16-224/best_fixed.ckpt"
    # teacher_ckpt_path = None
    teacher_ckpt_path = "/home/yin178/Equvariant_Model_Distillation_V2/outputs/cifar100/teacher/pretrained_ViT/non_equ_train_on_GT/teacher_vit_small_weight_selection/checkpoints/best.ckpt"
    precision = torch.float32
    # embed_dim = 192
    # embed_dim = 288
    # embed_dim = 192
    embed_dim = 96
    # embed_dim = 768
    scale_factor = 384 // embed_dim
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    group = Rot45Group()
    student_model = EquViT(  
            img_size=224,
            patch_size=16,
            in_channels=3,
            num_classes=100,
            embed_dim=embed_dim,
            depth=12,
            n_heads=3,
            mlp_ratio=4.0,
            pos_embed='SymmetricPosEmbed',
            # pos_embed='None-equ',
            attention_per_channel=True, 
            linear_pooling=False,
            group=group
            )

    
    data_config = DataConfig(
        dataset_name="cifar100",
        data_dir="./datasets",
        batch_size=64,
        num_workers=4,
        num_classes=100
    )
    
    data_module = CIFAR100DataModule(config=data_config)
    data_module.setup(stage='fit')
    calibration_loader = data_module.train_dataloader()
    
    preserved_state_dict = student_model.state_dict().copy()
    
    # copied_state_ckpt = pretrained_vit_initialize_student_from_teacher(teacher_model, student_model, teacher_ckpt_path,
    #                                                                    calibration_loader=calibration_loader, scale_factor=scale_factor, \
    #                                                                     device=device)
    
    # student_model = PretrainedViT(model_name="WinKawaks/vit-tiny-patch16-224", num_classes=100)
    # preserved_state_dict = student_model.state_dict().copy()
    

    copied_state_ckpt = pretrained_vit_initialization(teacher_model, teacher_ckpt_path, student_model, group=group,
                                device=device)
    
    # copied_state_ckpt = init_vit_tiny(teacher_model, student_model, device=device)
    
    check_state_dicts_match(copied_state_ckpt, preserved_state_dict)
    
    
    student_model_ckpt = {}
    student_model_ckpt['state_dict'] = copied_state_ckpt
    # output_path = "/home/yin178/Equvariant_Model_Distillation/outputs/CIFAR100/pretrained_ViT/student/initialization/half_channel/zero_init.ckpt"
    # output_path = "./outputs/CIFAR100/pretrained_ViT/student/initialization/double_channel/zero_init_v2.ckpt"
    # output_path = "./outputs/CIFAR100/pretrained_ViT/student/initialization/half_channel/192_zero_init_uniform_selection.ckpt"
    output_path = "./outputs/CIFAR100/pretrained_ViT/student/initialization/rot45/half_channel/96_zero_init_uniform_selection.ckpt"
    # output_path = "./outputs/CIFAR100/pretrained_ViT/student/initialization/vit_tiny_teacher/192_zero_init_uniform_selection.ckpt"
    
    
    
    # output_path = "/home/yin178/Equvariant_Model_Distillation/outputs/CIFAR100/pretrained_ViT/student/initialization/nonequ_pos_embed_real_zero_init.ckpt"
    breakpoint()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(student_model_ckpt, output_path)
    print("Student model initialized from teacher model.")
