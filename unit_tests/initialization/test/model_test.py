import unittest
import torch
import random
# from models import *
# from structures import *
from torchvision import models as torch_models

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.models.ViT.equ_vit import EquViT
from src.models.ViT.pretrained_HF import PretrainedViT





def flip_channel(x, dim_size, f_dim, all_dim):
        """
        Flips the channel dimension of the input tensor.
        """
        if all_dim==3:
            if f_dim==1:
                flipped_x_a = x[:, :dim_size, :]
                flipped_x_b = x[:, dim_size:]

            elif f_dim==2:
                flipped_x_a = x[:, :, :dim_size]
                flipped_x_b = x[:, :, dim_size:]

        elif all_dim==4:
            if f_dim==-1:
                flipped_x_a = x[:, :, :, :dim_size]
                flipped_x_b = x[:, :, :, dim_size:]
        else: 
            raise ValueError("Check dim value.")


        flipped_x = torch.cat((flipped_x_b, flipped_x_a), dim=f_dim)

        return flipped_x


def permute_patch_embd(x, patch_size, B, H, W, enc_embed_dim):
    assert x.dim() == 3, "x should be of shape (B, N, 2C)"

    if H % patch_size == 0:
        H_batched = int(H/patch_size)
        W_batched = int(W/patch_size)

        permuted_x_embd_lifted = x.view(B, H_batched, W_batched, 2*enc_embed_dim)
        permuted_x_embd_lifted = permuted_x_embd_lifted.permute(0, 3, 1, 2).flip(dims=(-1,))
        permuted_x_embd_lifted = permuted_x_embd_lifted.permute(0, 2, 3, 1).reshape(B, H_batched*W_batched, 2*enc_embed_dim)
        permuted_x_embd_lifted = flip_channel(permuted_x_embd_lifted, dim_size=enc_embed_dim, f_dim=2, all_dim=3)
    
    else:
        H_batched = int((H-1)/patch_size)
        W_batched = int((W-1)/patch_size)

        last_row = x[:, -1, :].unsqueeze(1)
        patch_row = x[:, :-1, :]

        permuted_patch_embd_lifted = patch_row.view(B, H_batched, W_batched, 2*enc_embed_dim)
        permuted_patch_embd_lifted = permuted_patch_embd_lifted.permute(0, 3, 1, 2).flip(dims=(-1,))
        permuted_patch_embd_lifted = permuted_patch_embd_lifted.permute(0, 2, 3, 1).reshape(B, H_batched*W_batched, 2*enc_embed_dim)
        permuted_patch_embd_lifted = flip_channel(permuted_patch_embd_lifted, dim_size=enc_embed_dim, f_dim=2, all_dim=3)

        permuted_last_embd = flip_channel(last_row, dim_size=enc_embed_dim, f_dim=2, all_dim=3)

        permuted_x_embd_lifted = torch.cat((permuted_patch_embd_lifted, permuted_last_embd), dim=1)


    return permuted_x_embd_lifted






class TestModels(unittest.TestCase):
    def _test_student_model_ViT(self):
        depth = 12
        img_size = 224
        nheads = 12
        embed_dim = 768
        num_classes = 100
        patch_size = 16
        
        student = EquViT(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=3,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            n_heads=nheads,
            mlp_ratio=4,
            pos_embed="None-equ",
            # pos_embed="SymmetricPosEmbed"
            attention_per_channel=True
        )
        
        student = student.to(torch.float64)
        # Load pretrained weights if available
        # student_checkpoint = torch.load('/home/yin178/Equvariant_Model_Distillation/outputs/CIFAR100/pretrained_ViT/student/initialization/real_zero_init.ckpt')
        student_checkpoint = torch.load("/home/yin178/Equvariant_Model_Distillation_V2/outputs/CIFAR100/pretrained_ViT/student/initialization/double_channel/zero_init_v2.ckpt")
        
        # student_checkpoint = torch.load('/home/yin178/Equvariant_Model_Distillation/outputs/CIFAR100/pretrained_ViT/student/initialization/nonequ_pos_embed_real_zero_init.ckpt')
        
        student_checkpoint['state_dict'] = {k: v for k, v in student_checkpoint['state_dict'].items() if k not in ['non_equ_pos_embed', 'add_pos_embed.pos_embed_left', 'add_pos_embed.cls_pos_half']}
        # student_checkpoint['state_dict'] = {k: v for k, v in student_checkpoint['state_dict'].items()}
        
        student.load_state_dict(student_checkpoint['state_dict'], strict=False)
        student.double()
        student.eval()
        student = student.to('cpu')

        
        # teacher = torch_models.vit_b_16(pretrained=True)
        # teacher.heads.head = nn.Linear(teacher.heads.head.in_features, num_classes)
        
        teacher = PretrainedViT(model_name="google/vit-base-patch16-224", num_classes=100)
        
        teacher_checkpoint = torch.load("/home/yin178/Equvariant_Model_Distillation_V2/outputs/CIFAR100/pretrained_ViT/teacher/google/vit-base-patch16-224/best_fixed.ckpt")
        
        # teacher_checkpoint['state_dict'] = {k.replace('model.m', 'm'): v for k, v in teacher_checkpoint['state_dict'].items()}
        
        # breakpoint()
        teacher.load_state_dict(teacher_checkpoint['state_dict'])
        teacher.double()
        teacher.eval()
        teacher = teacher.to('cpu')
        
        # teacher_doubled_pos_embd = torch.cat([teacher.encoder.pos_embedding, torch.zeros_like(teacher.encoder.pos_embedding)], dim=-1)
        teacher_doubled_pos_embd = torch.cat([teacher.model.model.vit.embeddings.position_embeddings, torch.zeros_like(teacher.model.model.vit.embeddings.position_embeddings)], dim=-1)
        
        # breakpoint()
        student.non_equ_pos_embed.data.copy_(teacher_doubled_pos_embd)
        
        
        

        torch.manual_seed(42)
        random.seed(42)
        x = torch.randn(1, 3, img_size, img_size).to(torch.float64)
        
        
        # def replace_layer_input(model, x, layer_idx, replacement_fn):
        #     """
        #     Replace the input of a specific layer using a forward pre-hook.
        #     replacement_fn: function that takes input and returns modified input
        #     """
        #     def pre_hook_fn(module, input):
        #         modified_input = replacement_fn(input[0])
        #         return (modified_input,) + input[1:]
            
        #     handle = model.get_submodule(layer_idx).register_forward_pre_hook(pre_hook_fn)
        #     output = model(x)
        #     handle.remove()
            
        #     return output


        # def permutation_fn(x):
        #     B, C, H, W = x.shape
        #     x_zeroed = x.clone()
        #     x_zeroed[:, 1:] = 0.0
        #     return x_zeroed

        # out_s_replaced = replace_layer_input(student, x, student_layer_idx, permutation_fn)
        

        # out_s = student(x)        
        def get_layer_features(model, x, layer_idx, inject_lifting=False):
            features = []
            
            def hook_fn(module, input, output):
                features.append(output)
            
            
            def pre_hook_fn(module, input):
                features.append(input[0])
            
            def permutation_fn(x):
                B, C, _, H, W = x.shape
                x_zeroed = x.clone()
                x_zeroed[:, :, 1:] = 0.0
                return x_zeroed
            
            
            
            def inject_hook_fn(module, input):
                modified_input = permutation_fn(input[0])
                return (modified_input,)
            
            # handle1 = model.get_submodule(layer_idx).register_forward_pre_hook(pre_hook_fn)
            if inject_lifting:
                handle1 = model.get_submodule("patch_embed.grouplayer").register_forward_pre_hook(inject_hook_fn)
            
            handle2 = model.get_submodule(layer_idx).register_forward_hook(hook_fn)
            # handle2 = model.get_submodule(layer_idx).register_forward_pre_hook(pre_hook_fn)

            model(x)
            handle2.remove()
            if inject_lifting:
                handle1.remove()
            
            return features[0]


  

            # Register hook on the specific layer
            handle = model.get_submodule(layer_idx).register_forward_hook(hook_fn)
            # handle = model.get_submodule(layer_idx).register_forward_pre_hook(pre_hook_fn)
            
            model(x)
            handle.remove()
            
            return features[0]
        
        teacher_layer_idx = 'model.model.vit.encoder.layer.11'
        student_layer_idx = 'blocks.11'
        
        # teacher_layer_idx = 'model.model.vit.encoder.layer.0'
        # student_layer_idx = 'blocks.0'
        
        # teacher_layer_idx = 
        
        # teacher_layer_idx = 'model.model.vit.embeddings.patch_embeddings'
        # student_layer_idx = 'patch_embed'
       
        teacher_layer_idx = 'model.model.vit.layernorm'
        student_layer_idx = 'norm'
       
        
        # teacher_layer_idx = 'model.classifier'
        # student_layer_idx = 'head'
        
        # teacher_layer_idx = 'model.vit.embeddings.position_embeddings'
        # student_layer_idx = 'add_pos_embed'
        
        
        
        out_s = get_layer_features(student, x, student_layer_idx, inject_lifting=False)
        out_t = get_layer_features(teacher, x, teacher_layer_idx)
        
        # breakpoint()
        
        # print("Error: ", torch.norm(out_s[:,:,:768] - out_t[0]).item())
        print("Error: ", torch.norm(out_s[:,:,:768] - out_t).item())
        
        

        # y_s = student(x)
        # y_t = teacher(x)
        # print("Logits Error: ", torch.norm(y_s - y_t).item())
        
        breakpoint()
        
        assert torch.allclose( out_s[:,:,:768], out_t[0], atol=1e-5)
        
        breakpoint()
        
        
        # fx = torch.flip(x, dims=(-1,))
        # out_fx = student(fx)
        
        
        # H_batched = int(img_size/4)
        # W_batched = int(img_size/4)

        # print(f"Invariance Error -->",torch.norm(out-out_fx).item())
        # assert torch.allclose( out, out_fx, atol=1e-5)
    def _test_group_cls_pooling(self):
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
            pos_embed="None-equ",
            # pos_embed="SymmetricPosEmbed",
            attention_per_channel=True,
            group_attn_channel_pooling=True
        )
        
        student = student.to(torch.float64)
        # Load pretrained weights if available
        student_checkpoint = torch.load('/home/yin178/Equvariant_Model_Distillation/outputs/CIFAR100/pretrained_ViT/student/initialization/real_zero_init.ckpt')
        student_checkpoint['state_dict'] = {k: v for k, v in student_checkpoint['state_dict'].items() if k not in ['non_equ_pos_embed', 'add_pos_embed.pos_embed_left', 'add_pos_embed.cls_pos_half']}
        student.load_state_dict(student_checkpoint['state_dict'], strict=False)
        student.double()
        student.eval()
        
        
        # prob = student(torch.randn(1, 3, img_size, img_size).to(torch.float64))
        
        teacher = get_pretrained_teacher_model(model_name="google/vit-base-patch16-224-in21k", num_classes=100, device='cpu')
        
        teacher_checkpoint = torch.load("/home/yin178/Equvariant_Model_Distillation/outputs/CIFAR100/pretrained_ViT/teacher/google/vit-base-patch16-224/best.ckpt")
        
        teacher_checkpoint['state_dict'] = {k.replace('model.m', 'm'): v for k, v in teacher_checkpoint['state_dict'].items()}
        
        # breakpoint()
        teacher.load_state_dict(teacher_checkpoint['state_dict'])
        teacher.double()
        teacher.eval()
        
        # teacher_doubled_pos_embd = torch.cat([teacher.encoder.pos_embedding, torch.zeros_like(teacher.encoder.pos_embedding)], dim=-1)
        teacher_doubled_pos_embd = torch.cat([teacher.model.vit.embeddings.position_embeddings, torch.zeros_like(teacher.model.vit.embeddings.position_embeddings)], dim=-1)
        
        student.non_equ_pos_embed.data.copy_(teacher_doubled_pos_embd)
        
        torch.manual_seed(42)
        random.seed(42)
        x = torch.randn(1, 3, img_size, img_size).to(torch.float64)
        
        
        student_y = student(x)
        teacher_y = teacher(x)
        
        
        print("Logits Error with Group CLS Pooling: ", torch.norm(student_y - teacher_y).item())
        assert torch.allclose( student_y, teacher_y, atol=1e-5)
        
        
        # breakpoint()
    
    def test_attn(self):
        student_ckpt_path = "/home/yin178/Equvariant_Model_Distillation_V2/outputs/CIFAR100/pretrained_ViT/student/initialization/half_channel/zero_init_v2.ckpt"

        student_checkpoint = torch.load(student_ckpt_path)
        student_state_dict = student_checkpoint['state_dict']
        student_weights = student_state_dict['blocks.0.attn.shared_q.learnable_weights.0']
        student_bias = student_state_dict['blocks.0.attn.shared_q.learnable_bias']
        
        teacher_ckpt_path = "/home/yin178/Equvariant_Model_Distillation_V2/outputs/CIFAR100/pretrained_ViT/teacher/google/vit-base-patch16-224/best_fixed.ckpt"
        teacher_checkpoint = torch.load(teacher_ckpt_path)
        teacher_state_dict = teacher_checkpoint['state_dict']
        
        teacher_weights = teacher_state_dict['model.model.vit.encoder.layer.0.attention.attention.query.weight']
        teacher_bias = teacher_state_dict['model.model.vit.encoder.layer.0.attention.attention.query.bias']
        
        student_c = student_weights.shape[0]
        
        teacher_linear_layer = torch.nn.Linear(teacher_weights.shape[1], teacher_weights.shape[0])
        teacher_linear_layer.weight.data = teacher_weights
        teacher_linear_layer.bias.data = teacher_bias
        
        student_linear_layer = torch.nn.Linear(teacher_weights.shape[1], teacher_weights.shape[0])
        student_linear_layer.weight.data = torch.zeros_like(student_linear_layer.weight.data)
        student_linear_layer.weight.data[:student_c, :student_c] = student_weights
        student_linear_layer.bias.data = torch.zeros_like(student_linear_layer.bias.data)
        student_linear_layer.bias.data[:student_c] = student_bias
        
        x  = torch.randn(1, 10, teacher_weights.shape[1])
        x [:,:, student_c:] = 0.0
        out_t = teacher_linear_layer(x)
        out_s = student_linear_layer(x)
        print(f"Attn query initialized errors: {torch.norm(out_t[:,:,:student_c] - out_s[:,:,:student_c]).item()}")
        breakpoint()
        
        


if __name__ == '__main__':
    unittest.main()
