import torch
import torchvision.models as torch_models
import torch.nn as nn

from typing import Dict, Any
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.ViT.pretrained_HF import PretrainedViT



def conv_identity_weight(out_c, in_c, k=3):
    w = torch.zeros(out_c, in_c, k, k)
    for o in range(out_c):
        i = o % in_c
        w[o, i, k//2, k//2] = 1.0
    return w




def copy_conv_weight(equ_weight, pretrained_weight):
    C1, C2, K, K = equ_weight.size()
    expanded = torch.zeros(C1//2, C2, K, K, device=equ_weight.device)
    expanded[:, ::2, :, :] = pretrained_weight
    equ_weight[0::2] = expanded
    return equ_weight




def cnn_initialize_student_from_teacher(teacher_model: nn.Module, student_model: nn.Module, teacher_ckpt_path: str) -> nn.Module:
    """
    Load teacher model checkpoint and initialize student model weights from teacher.
    
    Args:
        teacher_model: Teacher model architecture
        student_model: Student model architecture
        teacher_ckpt_path: Path to teacher model checkpoint
    
    Returns:
        Student model with weights initialized from teacher
    """
    # Load teacher checkpoint
    teacher_ckpt = torch.load(teacher_ckpt_path, map_location='cpu')
    
    # Handle both direct state_dict and nested checkpoints
    if isinstance(teacher_ckpt, dict) and 'state_dict' in teacher_ckpt:
        teacher_state_dict = teacher_ckpt['state_dict']
    else:
        teacher_state_dict = teacher_ckpt
    
    # Remove 'model.' prefix from checkpoint keys if present
    teacher_state_dict = {k.replace('model.', '', 1): v for k, v in teacher_state_dict.items()}
    # Load teacher weights
    teacher_model.load_state_dict(teacher_state_dict)
    
    # Copy teacher weights to student model, layer by layer
    student_state_dict = student_model.state_dict()
    
    for name, param in student_model.named_parameters():
        if "group_cls_token"  in name:
            student_state_dict[name] = torch.load(cls_feat_path).clone()
        elif "liftinglayer" in name:
            if "weight" in name:
                student_state_dict[name] = conv_identity_weight(1, 1, 3)
            elif "bias" in name:
                student_state_dict[name] = torch.zeros_like(param)
        elif "grouplayer" in name:
            if "weight" in name:
                C_out, C_in, kh, kw = param.shape
                student_state_dict[name] = torch.zeros_like(param)
                # student_state_dict[name][0::2, 0, :, :][:C_out//2] = teacher_state_dict[name.replace("grouplayer.", "")][:, 0, :, :].clone()
                student_state_dict[name][0::2, 0::2, :, :] = teacher_state_dict[name.replace("grouplayer.", "")].clone()

                
            elif "bias" in name:
                C_out = param.shape[0]
                student_state_dict[name] = torch.zeros_like(param)
                student_state_dict[name] = teacher_state_dict[name.replace("grouplayer.", "")].clone()
        elif "conv_layers" in name:
            if "weight" in name:
                C_out, C_in, kh, kw = param.shape
                student_state_dict[name] = torch.zeros_like(param)
                student_state_dict[name][0::2, 0::2, :, :] = teacher_state_dict[name].clone()
            elif "bias" in name:
                C_out = param.shape[0]
                student_state_dict[name] = torch.zeros_like(param)
                student_state_dict[name] = teacher_state_dict[name].clone()
        elif "norm" in name:
            if "weight" in name:
                C_out = param.shape[0]
                student_state_dict[name] = torch.ones_like(param)
                student_state_dict[name] = teacher_state_dict[name.replace("big_norm.", "")].clone()
            elif "bias" in name:
                C_out = param.shape[0]
                student_state_dict[name] = teacher_state_dict[name.replace("big_norm.", "")].clone()
        elif "fc" in name:
            if "weight" in name:
                C_out, C_in = param.shape
                student_state_dict[name] = torch.zeros_like(param)
                student_state_dict[name][:, :C_in] = teacher_state_dict[name].clone()
            elif "bias" in name:
                student_state_dict[name] = teacher_state_dict[name].clone()

    student_model.load_state_dict(student_state_dict)
    
    teacher_model.eval()
    student_model.eval()
    
    # x = torch.randn(1, 1, 28, 28).to(torch.float64)
    x = torch.randn(1, 1, 28, 28)
    
    y_student = student_model(x)
    y_teacher = teacher_model(x)
    
    return student_state_dict
    
    breakpoint()
    _, C, _, _ = y_student.shape
    print(torch.norm(y_student[:, :C//2] - y_teacher).item())
    
    breakpoint()
    
    return student_model

def pretrained_vit_initialize_student_from_teacher(teacher_model: nn.Module, student_model: nn.Module, teacher_ckpt_path: str, cls_feat_path: str, precision=torch.float16, scale_factor=1) -> nn.Module:
    """
    Load teacher model checkpoint and initialize student model weights from teacher.
    
    Args:
        teacher_model: Teacher model architecture
        student_model: Student model architecture
        teacher_ckpt_path: Path to teacher model checkpoint
    Returns:
        Student model with weights initialized from teacher
    """
     # Load teacher checkpoint
    teacher_ckpt = torch.load(teacher_ckpt_path, map_location='cpu')
    
    # Handle both direct state_dict and nested checkpoints
    if isinstance(teacher_ckpt, dict) and 'state_dict' in teacher_ckpt:
        teacher_state_dict = teacher_ckpt['state_dict']
    else:
        teacher_state_dict = teacher_ckpt
    
    # Remove 'model.' prefix from checkpoint keys if present
    

    # teacher_state_dict = {k.replace('model.m', 'm', 1): v for k, v in teacher_state_dict.items()}
    
    # Load teacher weights
    teacher_model.load_state_dict(teacher_state_dict)
    
    # Copy teacher weights to student model, layer by layer
    student_state_dict = student_model.state_dict()
    
    copied_state_dict = student_state_dict.copy()
    
    for name, param in student_model.named_parameters():
        # breakpoint()
        
        # This is only for the normal cls token
        if '.cls_token' in name:
            C = param.shape[0]
            copied_state_dict[name] = teacher_state_dict['model.model.vit.embeddings.cls_token'][:C//scale_factor].clone()
        
        elif 'non_equ_pos_embed' in name:
            breakpoint()
            teacher_doubled_pos_embd = torch.cat([teacher_state_dict['model.model.vit.embeddings.position_embeddings'], torch.zeros_like(teacher_state_dict['model.model.vit.embeddings.position_embeddings'])], dim=-1)
            copied_state_dict[name] = teacher_doubled_pos_embd.clone()
        
        elif "cls_pooling_layer" in name:
            breakpoint()
            copied_state_dict[name] = torch.zeros_like(param, dtype=precision)
            if "weight" in name:
                C_out, C_in = param.shape
                identity_block = torch.eye(C_out, dtype=precision)
                zero_block = torch.zeros(C_out, C_out, dtype=precision)
                copied_state_dict[name]= torch.cat([identity_block, zero_block], dim=1)
                copied_state_dict[name] = copied_state_dict[name].to(param.dtype)
        
        elif "liftinglayer" in name:
            if "weight" in name:
                H, W, _, _, = param.shape
                copied_state_dict[name] = conv_identity_weight(H, W, 3)
            elif "bias" in name:
                copied_state_dict[name] = torch.zeros_like(param)
                # copied_state_dict[name] += 1e-5
        elif "grouplayer" in name:
            if "weight" in name:
                teacher_name = name.replace("patch_embed.grouplayer.kernel", "model.model.vit.embeddings.patch_embeddings.projection")
                copied_state_dict[name] = torch.zeros_like(param)
                # copied_state_dict[name] += 1e-5
                C = param.shape[0]
                # breakpoint()
                # copied_state_dict[name][0::2, 0::2, :, :] = teacher_state_dict[teacher_name][:C//scale_factor].clone()
                copied_state_dict[name][:, :, 0::2, :, :] = teacher_state_dict[teacher_name][:C//scale_factor].unsqueeze(2).clone()
                
            elif "bias" in name:
                teacher_name = name.replace("patch_embed.grouplayer", "model.model.vit.embeddings.patch_embeddings.projection")
                C_out = param.shape[0]
                copied_state_dict[name] = torch.zeros_like(param)
                # copied_state_dict[name] += 1e-5
                copied_state_dict[name] = teacher_state_dict[teacher_name][:C_out].clone() 
        # elif 'add_pos_embed' in name:
        #     teacher_name = 'encoder.pos_embedding'
        #     student_state_dict[name] = teacher_state_dict[teacher_name].clone()
        
        elif 'norm' in name:
            
            teacher_name = name.replace("blocks.", "model.model.vit.encoder.layer.")
            if "norm.norm" in name:
                # This is the last layer norm
                teacher_name = name.replace("norm.norm", "model.model.vit.layernorm")
            elif "norm1" in name:
                teacher_name = teacher_name.replace(".norm1.norm", ".layernorm_before")
            elif "norm2" in name:
                teacher_name = teacher_name.replace(".norm2.norm", ".layernorm_after")

            C_s = student_state_dict[name].shape[0]
            
            try:
            
                C_t = teacher_state_dict[teacher_name].shape[0]
                copied_state_dict[name] = teacher_state_dict[teacher_name][:C_t//scale_factor].clone()
            except:
                print("Error in copying norm weights")
                breakpoint()

        
        elif 'attn' in name:
            if ('q.learnable_weights.1' in name or 'k.learnable_weights.1' in name or 'v.learnable_weights.1' in name or 'proj.learnable_weights.1' in name) \
               and 'learnable_bias' not in name:
                copied_state_dict[name] = torch.zeros_like(param)
                # copied_state_dict[name] += 1e-5
                
            elif '.learnable_weights.0' in name or 'learnable_bias' in name:
                teacher_name = name.replace("blocks.", "model.model.vit.encoder.layer.")
                if 'bias' not in name:
                    if 'shared' in name:
                        C_s, P = param.shape

                        if 'shared_q' in name:
                            teacher_name = teacher_name.replace("attn.shared_q.learnable_weights.0", "attention.attention.query.weight")
                            copied_state_dict[name] = teacher_state_dict[teacher_name][:C_s, :P].clone()
                        elif 'shared_k' in name:
                            teacher_name = teacher_name.replace("attn.shared_k.learnable_weights.0", "attention.attention.key.weight")
                            copied_state_dict[name] = teacher_state_dict[teacher_name][:C_s, :P].clone()
                        elif 'shared_v' in name:
                            teacher_name = teacher_name.replace("attn.shared_v.learnable_weights.0", "attention.attention.value.weight")
                            copied_state_dict[name] = teacher_state_dict[teacher_name][:C_s, :P].clone()
                        else:
                            print("1")
                            breakpoint()
                                
                                
                    elif 'proj' in name:
                        teacher_name = teacher_name.replace("attn.proj.learnable_weights.0", "attention.output.dense.weight")
                        C, P = param.shape
                        copied_state_dict[name] = teacher_state_dict[teacher_name][:C, :P].clone()
                    
                    else:
                        print("2")
                        
                        breakpoint()
                        
                elif 'bias' in name:
                    if 'shared' in name:
                        C_s = param.shape[0]
                        if 'shared_q' in name:
                            teacher_name = teacher_name.replace("attn.shared_q.learnable_bias", "attention.attention.query.bias")
                            copied_state_dict[name] = teacher_state_dict[teacher_name][:C_s].clone()
                        elif 'shared_k' in name:
                            teacher_name = teacher_name.replace("attn.shared_k.learnable_bias", "attention.attention.key.bias")
                            copied_state_dict[name] = teacher_state_dict[teacher_name][:C_s].clone()
                        elif 'shared_v' in name:
                            teacher_name = teacher_name.replace("attn.shared_v.learnable_bias", "attention.attention.value.bias")
                            copied_state_dict[name] = teacher_state_dict[teacher_name][:C_s].clone()
                        else:
                            print("3")
                            
                            breakpoint()
                                
                    elif 'proj' in name:
                        teacher_name = teacher_name.replace("attn.proj.learnable_bias", "attention.output.dense.bias")
                        C = param.shape[0]
                        copied_state_dict[name] = teacher_state_dict[teacher_name][:C].clone()
                    else:
                        print("4")
                        
                        breakpoint()
                else:
                    print("5")
                    
                    breakpoint()
                        
            else:
                print("6")
                
                breakpoint()
                
        elif 'mlp' in name:
            if '.learnable_weights.1' in name and 'learnable_bias' not in name:
                copied_state_dict[name] = torch.zeros_like(param)
                # copied_state_dict[name] += 1e-5
            elif '.learnable_weights.0' in name or 'learnable_bias' in name:
                teacher_name = name.replace("blocks.", "model.model.vit.encoder.layer.")
                if 'bias' not in name:
                    if 'fc1' in name:
                        teacher_name = teacher_name.replace("mlp.fc1.learnable_weights.0", "intermediate.dense.weight")
                        C, P = param.shape
                        copied_state_dict[name] = teacher_state_dict[teacher_name][:C, :P].clone()
                    elif 'fc2' in name:
                        teacher_name = teacher_name.replace("mlp.fc2.learnable_weights.0", "output.dense.weight")
                        C, P = param.shape
                        copied_state_dict[name] = teacher_state_dict[teacher_name][:C, :P].clone()
                    else:
                        breakpoint()
                        
                elif 'bias' in name:
                    if 'fc1' in name:
                        teacher_name = teacher_name.replace("mlp.fc1.learnable_bias", "intermediate.dense.bias")
                        C = param.shape[0]
                        copied_state_dict[name] = teacher_state_dict[teacher_name][:C].clone()
                    elif 'fc2' in name:
                        teacher_name = teacher_name.replace("mlp.fc2.learnable_bias", "output.dense.bias")
                        C = param.shape[0]
                        copied_state_dict[name] = teacher_state_dict[teacher_name][:C].clone()
                
            else:
                print("7")
                breakpoint()
        elif 'head' in name:
            teacher_name = name.replace("head", "model.model.classifier")
            if param.ndim == 2:
                C = param.shape[1]
                copied_state_dict[name] = teacher_state_dict[teacher_name][:, :C].clone()
            elif param.ndim ==1:
                copied_state_dict[name] = teacher_state_dict[teacher_name].clone()


    return copied_state_dict
            










#############This is one using pretrained vit from torchvision###############
# def pretrained_vit_initialize_student_from_teacher(teacher_model: nn.Module, student_model: nn.Module, teacher_ckpt_path: str) -> nn.Module:
#     """
#     Load teacher model checkpoint and initialize student model weights from teacher.
    
#     Args:
#         teacher_model: Teacher model architecture
#         student_model: Student model architecture
#         teacher_ckpt_path: Path to teacher model checkpoint
#     Returns:
#         Student model with weights initialized from teacher
#     """
#      # Load teacher checkpoint
#     teacher_ckpt = torch.load(teacher_ckpt_path, map_location='cpu')
    
#     # Handle both direct state_dict and nested checkpoints
#     if isinstance(teacher_ckpt, dict) and 'state_dict' in teacher_ckpt:
#         teacher_state_dict = teacher_ckpt['state_dict']
#     else:
#         teacher_state_dict = teacher_ckpt
    
#     # Remove 'model.' prefix from checkpoint keys if present
#     teacher_state_dict = {k.replace('model.', '', 1): v for k, v in teacher_state_dict.items()}
#     # Load teacher weights
#     teacher_model.load_state_dict(teacher_state_dict)
    
#     # Copy teacher weights to student model, layer by layer
#     student_state_dict = student_model.state_dict()
    
#     copied_state_dict = student_state_dict.copy()
    
#     for name, param in student_model.named_parameters():
        
#         if name == 'cls_token':
#             copied_state_dict[name] = teacher_state_dict['class_token'].clone()
        
#         elif "liftinglayer" in name:
#             if "weight" in name:
#                 H, W, _, _, = param.shape
#                 copied_state_dict[name] = conv_identity_weight(H, W, 3)
#             elif "bias" in name:
#                 copied_state_dict[name] = torch.zeros_like(param)
#         elif "grouplayer" in name:
#             teacher_name = name.replace("patch_embed.proj.grouplayer", "conv_proj")
#             if "weight" in name:
#                 copied_state_dict[name] = torch.zeros_like(param)
#                 copied_state_dict[name][0::2, 0::2, :, :] = teacher_state_dict[teacher_name].clone()
#             elif "bias" in name:
#                 C_out = param.shape[0]
#                 copied_state_dict[name] = torch.zeros_like(param)
#                 copied_state_dict[name] = teacher_state_dict[teacher_name].clone() 
#         # elif 'add_pos_embed' in name:
#         #     teacher_name = 'encoder.pos_embedding'
#         #     student_state_dict[name] = teacher_state_dict[teacher_name].clone()
        
#         elif 'norm' in name:
#             teacher_name = name.replace("blocks.", "encoder.layers.encoder_layer_")
#             if "norm.big_norm" in name:
#                 teacher_name = teacher_name.replace("norm.big_norm", "encoder.ln")
#             else:
#                 teacher_name = teacher_name.replace(".norm", ".ln_")
#                 teacher_name = teacher_name.replace("big_norm.", "")
            
#             C = student_state_dict[name].shape[0]
#             copied_state_dict[name][:C//2] = teacher_state_dict[teacher_name][:C//2].clone()

        
#         elif 'attn' in name:
#             if '.b' in name or 'bias_b' in name:
#                 copied_state_dict[name] = torch.zeros_like(param)
                
#             elif '.a' in name:
#                 teacher_name = name.replace("blocks.", "encoder.layers.encoder_layer_")
#                 if 'bias' not in name:
#                     if 'shared' in name:
#                         C_s, P = param.shape

#                         if 'shared_q' in name:
#                             teacher_name = teacher_name.replace("attn.shared_q.a", "self_attention.in_proj_weight")
#                             copied_state_dict[name] = teacher_state_dict[teacher_name][:C_s, :P].clone()
#                         elif 'shared_k' in name:
#                             teacher_name = teacher_name.replace("attn.shared_k.a", "self_attention.in_proj_weight")
#                             copied_state_dict[name] = teacher_state_dict[teacher_name][C_s:C_s*2, :P].clone()
#                         elif 'shared_v' in name:
#                             teacher_name = teacher_name.replace("attn.shared_v.a", "self_attention.in_proj_weight")
#                             copied_state_dict[name] = teacher_state_dict[teacher_name][C_s*2:, :P].clone()
#                         else:
#                             breakpoint()
                                
                                
#                     elif 'proj' in name:
#                         teacher_name = teacher_name.replace("attn.proj.a", "self_attention.out_proj.weight")
#                         C, P = param.shape
#                         copied_state_dict[name] = teacher_state_dict[teacher_name][:C, :P].clone()
                    
#                     else:
#                         breakpoint()
                        
#                 elif 'bias' in name:
#                     if 'qkv' in name:
#                         C_s = param.shape[0]
#                         if 'shared_q' in name:
#                             teacher_name = name.replace("attn.shared_q.bias_a", "self_attention.in_proj_bias")
#                             copied_state_dict[name] = teacher_state_dict[teacher_name][:C_s].clone()
#                         elif 'shared_k' in name:
#                             teacher_name = name.replace("attn.shared_k.bias_a", "self_attention.in_proj_bias")
#                             copied_state_dict[name] = teacher_state_dict[teacher_name][C_s:C_s*2].clone()
#                         elif 'shared_v' in name:
#                             teacher_name = name.replace("attn.shared_v.bias_a", "self_attention.in_proj_bias")
#                             copied_state_dict[name] = teacher_state_dict[teacher_name][C_s*2:].clone()
#                         else:
#                             breakpoint()
                                
#                     elif 'proj' in name:
#                         teacher_name = teacher_name.replace("attn.proj.bias_a", "self_attention.out_proj.bias")
#                         C = param.shape[0]
#                         copied_state_dict[name] = teacher_state_dict[teacher_name][:C].clone()
#                     else:
#                         breakpoint()
                        
#             else:
#                 breakpoint()
                
#         elif 'mlp' in name:
#             if '.b' in name:
#                 copied_state_dict[name] = torch.zeros_like(param)
#             elif '.a' in name:
#                 teacher_name = name.replace("blocks.", "encoder.layers.encoder_layer_")
#                 if 'bias' not in name:
#                     if 'fc1' in name:
#                         teacher_name = teacher_name.replace("fc1.a", "0.weight")
#                         C, P = param.shape
#                         copied_state_dict[name] = teacher_state_dict[teacher_name][:C, :P].clone()
#                     elif 'fc2' in name:
#                         teacher_name = teacher_name.replace("fc2.a", "3.weight")
#                         C, P = param.shape
#                         copied_state_dict[name] = teacher_state_dict[teacher_name][:C, :P].clone()
#                     else:
#                         breakpoint()
                        
#                 elif 'bias' in name:
#                     if 'fc1' in name:
#                         teacher_name = teacher_name.replace("fc1.bias_a", "0.bias")
#                         C = param.shape[0]
#                         copied_state_dict[name] = teacher_state_dict[teacher_name][:C].clone()
#                     elif 'fc2' in name:
#                         teacher_name = teacher_name.replace("fc2.bias_a", "3.bias")
#                         C = param.shape[0]
#                         copied_state_dict[name] = teacher_state_dict[teacher_name][:C].clone()
                
#             else:
#                 breakpoint()
#         elif 'head' in name:
#             teacher_name = name.replace("head", "heads.head")
#             copied_state_dict[name] = teacher_state_dict[teacher_name].clone()
    

#     return copied_state_dict
            
                
    
    

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
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    # from src.models.CNN.teacher_model import CNN_TeacherModel
    # from src.models.CNN.student_model import CNN_FlippingInvStudentModel

    # teacher_model = CNN_TeacherModel(num_conv_layers=5, base_channels=32, num_classes=10)
    # student_model = CNN_FlippingInvStudentModel(num_conv_layers=5, base_channels=32, num_classes=10)
    # teacher_ckpt_path = "/home/yin178/Equvariant_Model_Distillation/outputs/MNIST/CNN/teacher/conv5/epoch=14.ckpt"
    
    # student_state_ckpt = cnn_initialize_student_from_teacher(teacher_model, student_model, teacher_ckpt_path)
    
    
    from src.models.ViT.equ_vit import EquViT
    
    # teacher_model = torch_models.vit_b_16(pretrained=True)
    # teacher_model.heads.head = nn.Linear(768, 100)
    
    
    
    
    # teacher_model = get_pretrained_teacher_model(model_name="google/vit-base-patch16-224-in21k", num_classes=100)
    
    teacher_model = PretrainedViT(model_name="google/vit-base-patch16-224", num_classes=100)
    # embed_dim = 384
    embed_dim = 768
    scale_factor = 768 // embed_dim
    
    student_model = EquViT(  
            img_size=224,
            patch_size=16,
            in_channels=3,
            num_classes=100,
            embed_dim=embed_dim,
            depth=12,
            n_heads=12,
            mlp_ratio=4.0,
            pos_embed='SymmetricPosEmbed',
            # pos_embed='None-equ',
            attention_per_channel=True, 
            linear_pooling=False
            )
    # teacher_ckpt_path = "/home/yin178/Equvariant_Model_Distillation/outputs/CIFAR100/ViT/teacher/pretrained_finetuned/epoch=07.ckpt"
    teacher_ckpt_path = "/home/yin178/Equvariant_Model_Distillation/outputs/CIFAR100/pretrained_ViT/teacher/google/vit-base-patch16-224/best_fixed.ckpt"
    cls_feat_path = "./outputs/CIFAR100/cls_features/cls_features.pt"

    cls_feat = torch.load(cls_feat_path)

    # breakpoint()
    # student_model.linear_pooling_layer.ls_init(cls_feat)
    precision = torch.float32
    
    copied_state_ckpt = pretrained_vit_initialize_student_from_teacher(teacher_model, student_model, teacher_ckpt_path,
                                                                       cls_feat_path, precision, scale_factor=scale_factor)
    
    check_state_dicts_match(copied_state_ckpt, student_model.state_dict())
    
    
    student_model_ckpt = {}
    student_model_ckpt['state_dict'] = copied_state_ckpt
    # output_path = "/home/yin178/Equvariant_Model_Distillation/outputs/CIFAR100/pretrained_ViT/student/initialization/half_channel/zero_init.ckpt"
    output_path = "./outputs/CIFAR100/pretrained_ViT/student/initialization/double_channel/zero_init_v2.ckpt"
    
    # output_path = "/home/yin178/Equvariant_Model_Distillation/outputs/CIFAR100/pretrained_ViT/student/initialization/nonequ_pos_embed_real_zero_init.ckpt"
    breakpoint()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(student_model_ckpt, output_path)
    print("Student model initialized from teacher model.")