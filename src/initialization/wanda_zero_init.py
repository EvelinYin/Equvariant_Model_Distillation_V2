import torch
import torchvision.models as torch_models
import torch.nn as nn

from typing import Dict, Any
import sys
import os

from sklearn.decomposition import PCA

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


@torch.no_grad()
def wanda_channel_idx(W, X, C_row, C_col):
    """
    W: (C_out, C_in)
    X: (N*L, C_in)
    dim: 0 is row-wise, 1 is column-wise
    returns: (C_in,)
    """
    try:
        scores = W.abs() * torch.sqrt(X.reshape((1, -1)))
        col_scores = scores.sum(dim=0)
        col_keep_idx = torch.topk(col_scores, C_col, dim=0).indices
        col_keep_idx, _ = torch.sort(col_keep_idx)
        
        row_scores = scores.sum(dim=1)  
        row_keep_idx = torch.topk(row_scores, C_row, dim=0).indices
        row_keep_idx, _ = torch.sort(row_keep_idx)
    except:
        breakpoint()
    return col_keep_idx, row_keep_idx



def get_wanda_mask(W, X, sparsity_ratio):
    W_metric = torch.abs(W) * torch.sqrt(X.reshape((1,-1)))
    sort_res = torch.sort(W_metric, dim=-1, stable=True)
    indices = sort_res[1][:,:int(W_metric.shape[1]*sparsity_ratio)]
    W_mask = (torch.zeros_like(W_metric) == 1)
    W_mask.scatter_(1, indices, True)
    return W_mask



# def get_wanda_metric(W, X):
#     """
#     W: (C_out, C_in)
#     X: (N*L, C_in)
#     dim: 0 is row-wise, 1 is column-wise
#     returns: (C_in,)
#     """
#     W_metric = torch.abs(W) * torch.sqrt(X.reshape((1,-1)))
#     tmp_metric = torch.cumsum(sort_res[0], dim=1)
#     sort_res = torch.sort(W_metric, dim=-1, stable=True)
#     W_mask = (torch.zeros_like(W_metric) == 1)
#     return sort_res, W_metric, W_mask, W_metric.sum(dim=1)


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def magnitude_channel_idx(W, C_small):
    """
    W: (C_out, C_in)
    dim: 0 is row-wise, 1 is column-wise
    returns: (C_in,)
    """
        
    scores = W.abs().mean()
    keep_idx = torch.topk(scores, C_small).indices
    keep_idx, _ = torch.sort(keep_idx)
    
    return keep_idx


# ------------------------------------------------------------
# Activation collector (calibration)
# ------------------------------------------------------------

class ActivationNormAccumulator:
    def __init__(self, hidden_dim, device):
        self.sum_sq = torch.zeros(hidden_dim, device=device)
        self.nsamples = 0
        

    def update(self, x):
        # x: (B, L, C) or (B, C)
        tmp = x.shape[0]
        self.sum_sq *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
                
        # breakpoint()
        x = x.reshape(-1, x.shape[-1]).t()
        try:
            self.sum_sq += torch.norm(x, p=2, dim=1) ** 2  / self.nsamples
        
        except:
            breakpoint()




class ViTActivationCollector:
    def __init__(self, model, device):
        self.handles = []
        self.accumulators = {}

        # Transformer blocks
        for i, block in enumerate(model.vit.encoder.layer):
            key = f"layer_{i}"
            self.accumulators[key] = {}
            def make_forward_hook(k, layer_name):
                def hook(_, __, output):
                    self.accumulators[k][layer_name].update(output.detach())
                return hook

            def make_pre_hook(k, layer_name):
                if layer_name is not None:
                    def hook(_, input):
                        self.accumulators[k][layer_name].update(input[0])
                else:
                    def hook(_, input):
                        self.accumulators[k].update(input[0])
                return hook   
            
            # For norm1 layer as input to q,k,v
            C = block.layernorm_before.normalized_shape[0]
            self.accumulators[key]["norm1_out"] = ActivationNormAccumulator(C, device)
            self.handles.append(
                block.layernorm_before.register_forward_hook(make_forward_hook(key, "norm1_out"))
            )
            
            # For attn output before proj
            C = block.attention.output.dense.out_features
            self.accumulators[key]["attn_out"] = ActivationNormAccumulator(C, device)
            self.handles.append(
                block.attention.output.dense.register_forward_pre_hook(make_pre_hook(key, "attn_out"))
            )
            
            # For norm2 layer output as input to fc1
            C = block.layernorm_after.normalized_shape[0]
            self.accumulators[key]["norm2_out"] = ActivationNormAccumulator(C, device)
            self.handles.append(
                block.layernorm_after.register_forward_hook(make_forward_hook(key, "norm2_out"))
            )
            
            # For mlp1 output before fc2
            C = block.intermediate.dense.out_features
            self.accumulators[key]["mlp1_out"] = ActivationNormAccumulator(C, device)
            self.handles.append(
                block.intermediate.dense.register_forward_hook(make_forward_hook(key, "mlp1_out"))
            )
            
            
        # For classifier head
        C = model.classifier.in_features
        self.accumulators["classifier"] = ActivationNormAccumulator(C, device)
        self.handles.append(
            model.classifier.register_forward_pre_hook(make_pre_hook("classifier", None))
        )
            
            

    def remove(self):
        for h in self.handles:
            h.remove()





# ------------------------------------------------------------
# Linear shrinking helper (INVARIANT-SAFE)
# ------------------------------------------------------------

@torch.no_grad()
def shrink_linear(big, small, row_idx=None, col_idx=None):
    """
    row_idx → output channels
    col_idx → input channels
    """
    W = big.weight.data

    if row_idx is not None:
        W = W[row_idx]

    if col_idx is not None:
        W = W[:, col_idx]

    
    if row_idx is not None:
            small.learnable_weights[0].data.copy_(W)

    else:
            small.weight.data.copy_(W)

    if big.bias is not None:
        if row_idx is not None:
                small.learnable_bias.data.copy_(big.bias.data[row_idx]) 
        else:
                small.bias.data.copy_(big.bias.data)
        # else:
        #     if small.bias is not None:
        #         small.bias.data.copy_(big.bias.data)
        #     elif small.learnable_bias is not None:
        #         small.learnable_bias.data.copy_(big.bias.data)






def pretrained_vit_initialize_student_from_teacher(teacher_model: nn.Module, student_model: nn.Module, teacher_ckpt_path: str,\
                                                    calibration_loader, scale_factor=1, device="cpu") -> nn.Module:
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
    
    # Load teacher weights
    teacher_model.load_state_dict(teacher_state_dict)
    
    teacher_model = teacher_model.model.model
    
    teacher_model.eval().to(device)
    student_model.eval().to(device)

    
    # Copy teacher weights to student model, layer by layer
    student_state_dict = student_model.state_dict()
    # copied_state_dict = student_state_dict.copy()
    
    

    # cls_token 
    W = teacher_model.vit.embeddings.cls_token.data
    scores = W.abs().squeeze()
    C_small = W.shape[2] // scale_factor
    keep_idx = torch.topk(scores, C_small).indices
    keep_idx, _ = torch.sort(keep_idx)
    student_model.cls_token.data = teacher_model.vit.embeddings.cls_token.data[:, :, keep_idx]
    
    
    # Lifting layer
    H,W,_,_ = student_model.patch_embed.liftinglayer.kernel.weight.shape
    student_model.patch_embed.liftinglayer.kernel.weight.data = conv_identity_weight(H,W,k=3)
    student_model.patch_embed.liftinglayer.bias.data.zero_()
    
    
    # Patch embedding group conv layer ()
    C_big = teacher_model.vit.embeddings.patch_embeddings.projection.out_channels
    C_small = C_big // scale_factor
    
    W = teacher_model.vit.embeddings.patch_embeddings.projection.weight.data
    scores = W.abs().mean(dim=(1,2,3))
    keep_idx = torch.topk(scores, C_small).indices
    keep_idx, _ = torch.sort(keep_idx)
    student_model.patch_embed.grouplayer.kernel.weight.data = torch.zeros_like(student_model.patch_embed.grouplayer.kernel.weight.data)
    student_model.patch_embed.grouplayer.kernel.weight.data[:,:,0::2] = W[keep_idx].unsqueeze(2)
    
    student_model.patch_embed.grouplayer.bias.data = torch.zeros_like(student_model.patch_embed.grouplayer.bias.data)
    student_model.patch_embed.grouplayer.bias.data = teacher_model.vit.embeddings.patch_embeddings.projection.bias.data[keep_idx]

    
    
    collector = ViTActivationCollector(teacher_model, device)
    
    
    with torch.no_grad():
        for images, idx in tqdm(calibration_loader, desc="Collecting activations"):
            images = images.to(device)
            _ = teacher_model(images)
            # break

    collector.remove()
    

    

    # breakpoint()
    # Transformer blocks
    for block_idx, block_big, block_small in zip(
        range(len(teacher_model.vit.encoder.layer)),
        teacher_model.vit.encoder.layer,
        student_model.blocks
    ):
        # ----------------------------------------------------
        # Attention: qkv consumes residual, proj produces it
        # ----------------------------------------------------

        
        
        block_small.attn.shared_q.learnable_weights[1].data.zero_()
        block_small.attn.shared_k.learnable_weights[1].data.zero_()
        block_small.attn.shared_v.learnable_weights[1].data.zero_()
        block_small.attn.shared_q.learnable_bias.data.zero_()
        block_small.attn.shared_k.learnable_bias.data.zero_()
        block_small.attn.shared_v.learnable_bias.data.zero_()
        
        block_small.attn.proj.learnable_weights[1].data.zero_()
        block_small.attn.proj.learnable_bias.data.zero_()
        
        block_small.mlp.fc1.learnable_weights[1].data.zero_()
        block_small.mlp.fc2.learnable_weights[1].data.zero_()
        block_small.mlp.fc1.learnable_bias.data.zero_()
        block_small.mlp.fc2.learnable_bias.data.zero_()
        

        # For q,k,v
        col_keep_idx, row_keep_idx = wanda_channel_idx(
            block_big.attention.attention.query.weight.data,
            collector.accumulators[f'layer_{block_idx}']['norm1_out'].sum_sq,
            C_col=block_small.attn.dim,
            C_row=block_small.attn.dim 
        )
        
        
        shrink_linear(
            block_big.attention.attention.query,
            block_small.attn.shared_q,
            row_idx=row_keep_idx,
            col_idx=col_keep_idx
        )
        
        shrink_linear(
            block_big.attention.attention.key,
            block_small.attn.shared_k,
            row_idx=row_keep_idx,
            col_idx=col_keep_idx
        )
        
        shrink_linear(
            block_big.attention.attention.value,
            block_small.attn.shared_v,
            row_idx=row_keep_idx,
            col_idx=col_keep_idx
        )
        
        
        
        
        # For attn proj
        col_keep_idx, row_keep_idx = wanda_channel_idx(
            block_big.attention.attention.output.dense.weight.data,
            collector.accumulators[f'layer_{block_idx}']['attn_out'].sum_sq,
            C_col=block_small.attn.dim,
            C_row=block_small.attn.dim
        )
        
        
        shrink_linear(
            block_big.attention.output.dense,
            block_small.attn.proj,
            row_idx=row_keep_idx,
            col_idx=col_keep_idx
        )
        
        

        # For mlp fc1 and fc2
        col_keep_idx, row_keep_idx = wanda_channel_idx(
            block_big.intermediate.dense.weight.data,
            collector.accumulators[f'layer_{block_idx}']['norm2_out'].sum_sq,
            C_col=block_small.mlp.fc1.learnable_weights[0].shape[1],
            C_row=block_small.mlp.fc1.learnable_weights[0].shape[0]
        )
        
        
        shrink_linear(
            block_big.intermediate.dense,
            block_small.mlp.fc1,
            row_idx=row_keep_idx,
            col_idx=col_keep_idx
        )
        
        
        
        col_keep_idx, row_keep_idx = wanda_channel_idx(
            block_big.output.dense.weight.data,
            collector.accumulators[f'layer_{block_idx}']['mlp1_out'].sum_sq,
            C_col=block_small.mlp.fc2.learnable_weights[0].shape[1],
            C_row=block_small.mlp.fc2.learnable_weights[0].shape[0]
        )
        
        shrink_linear(
            block_big.output.dense,
            block_small.mlp.fc2,
            row_idx=row_keep_idx,
            col_idx=col_keep_idx
        )
        
        
        # LayerNorms
        W = block_big.layernorm_before.weight.data
        scores = W.abs()
        keep_idx = torch.topk(scores, C_small).indices
        keep_idx, _ = torch.sort(keep_idx)
        block_small.norm1.norm.weight.data.copy_(block_big.layernorm_before.weight.data[keep_idx])
        block_small.norm1.norm.bias.data.copy_(block_big.layernorm_before.bias.data[keep_idx])
        
        
        W = block_big.layernorm_after.weight.data
        scores = W.abs()
        keep_idx = torch.topk(scores, C_small).indices
        keep_idx, _ = torch.sort(keep_idx)
        block_small.norm2.norm.weight.data.copy_(block_big.layernorm_after.weight.data[keep_idx])
        block_small.norm2.norm.bias.data.copy_(block_big.layernorm_after.bias.data[keep_idx])
        
    # Final LayerNorm
    W = teacher_model.vit.layernorm.weight.data
    scores = W.abs()
    keep_idx = torch.topk(scores, C_small).indices
    keep_idx, _ = torch.sort(keep_idx)
    student_model.norm.norm.weight.data.copy_(teacher_model.vit.layernorm.weight.data[keep_idx])
    student_model.norm.norm.bias.data.copy_(teacher_model.vit.layernorm.bias.data[keep_idx])
    
    # Classifier head
    W = teacher_model.classifier.weight.data
    scores = W.abs().mean(dim=0)
    keep_idx = torch.topk(scores, C_small).indices
    keep_idx, _ = torch.sort(keep_idx)
    shrink_linear(
        teacher_model.classifier,
        student_model.head,
        row_idx=None,
        col_idx=keep_idx
    )
        
    
    return student_model.state_dict()
    

    # return copied_state_dict
            






def pretrained_vit_model_pruning(teacher_model: nn.Module, teacher_ckpt_path: str,\
                                 calibration_loader, sparsity_ratio=0.5, device="cpu") -> nn.Module:
    """
    Load teacher model checkpoint and initialize student model weights from teacher.
    
    Args:
        teacher_model: Teacher model architecture
        student_model: Student model architecture
        teacher_ckpt_path: Path to teacher model checkpoint
    Returns:
        Student model with weights initialized from teacher
    """
    
    def copy_partial_weights(weights,row_keep_idx, col_keep_idx):
        new_weights = torch.zeros_like(weights)
        if col_keep_idx is None:
            new_weights[row_keep_idx] = weights[row_keep_idx]
        else:
            new_weights[row_keep_idx[:, None], col_keep_idx] = weights[row_keep_idx[:, None], col_keep_idx]
        return new_weights

    
    

    # # cls_token 
    # W = teacher_model.vit.embeddings.cls_token.data
    # scores = W.abs().squeeze()
    # C_small = W.shape[2] // scale_factor
    # keep_idx = torch.topk(scores, C_small).indices
    # keep_idx, _ = torch.sort(keep_idx)
    # student_model.cls_token.data = teacher_model.vit.embeddings.cls_token.data[:, :, keep_idx]
    
    
    
    # Patch embedding group conv layer ()
    # C_big = teacher_model.vit.embeddings.patch_embeddings.projection.out_channels
    # C_small = C_big // scale_factor
    
    # W = teacher_model.vit.embeddings.patch_embeddings.projection.weight.data
    # scores = W.abs().mean(dim=(1,2,3))
    # keep_idx = torch.topk(scores, C_small).indices
    # keep_idx, _ = torch.sort(keep_idx)
    # new_weights = torch.zeros_like(teacher_model.vit.embeddings.patch_embeddings.projection.weight.data)
    # new_weights[keep_idx] = teacher_model.vit.embeddings.patch_embeddings.projection.weight.data[keep_idx]
    # teacher_model.vit.embeddings.patch_embeddings.projection.weight.data = new_weights.clone()
    
    # new_bias = torch.zeros_like(teacher_model.vit.embeddings.patch_embeddings.projection.bias.data) 
    # new_bias[keep_idx] = teacher_model.vit.embeddings.patch_embeddings.projection.bias.data[keep_idx]
    # teacher_model.vit.embeddings.patch_embeddings.projection.bias.data = new_bias.clone()

    
    
    collector = ViTActivationCollector(teacher_model, device)
    
    
    with torch.no_grad():
        for images, idx in tqdm(calibration_loader, desc="Collecting activations"):
            images = images.to(device)
            _ = teacher_model(images)
            break

    collector.remove()
    

    

    # breakpoint()
    # Transformer blocks
    for block_idx, block_big in zip(
        range(len(teacher_model.vit.encoder.layer)),
        teacher_model.vit.encoder.layer,
    ):
        # ----------------------------------------------------
        # Attention: qkv consumes residual, proj produces it
        # ----------------------------------------------------

        

        # For q,k,v
        # col_keep_idx, row_keep_idx = wanda_channel_idx(
        #     block_big.attention.attention.query.weight.data,
        #     collector.accumulators[f'layer_{block_idx}']['norm1_out'].sum_sq,
        #     C_col=C_small,
        #     C_row=C_small
        # )
        W_mask = get_wanda_mask(
            block_big.attention.attention.query.weight.data,
            collector.accumulators[f'layer_{block_idx}']['norm1_out'].sum_sq,
            sparsity_ratio=sparsity_ratio
        )
        block_big.attention.attention.query.weight.data[W_mask] = 0.0
        
        
        W_mask = get_wanda_mask(
            block_big.attention.attention.key.weight.data,
            collector.accumulators[f'layer_{block_idx}']['norm1_out'].sum_sq,
            sparsity_ratio=sparsity_ratio
        )
        block_big.attention.attention.key.weight.data[W_mask] = 0.0
        
        
        
        W_mask = get_wanda_mask(
            block_big.attention.attention.value.weight.data,
            collector.accumulators[f'layer_{block_idx}']['norm1_out'].sum_sq,
            sparsity_ratio=sparsity_ratio
        )
        block_big.attention.attention.value.weight.data[W_mask] = 0.0
        
        
        
        
        # For attn proj
        W_mask = get_wanda_mask(
            block_big.attention.output.dense.weight.data,
            collector.accumulators[f'layer_{block_idx}']['attn_out'].sum_sq,
            sparsity_ratio=sparsity_ratio
        )
        block_big.attention.output.dense.weight.data[W_mask] = 0.0
        
        

        # For mlp fc1 and fc2
        W_mask = get_wanda_mask(
            block_big.intermediate.dense.weight.data,
            collector.accumulators[f'layer_{block_idx}']['norm2_out'].sum_sq,
            sparsity_ratio=sparsity_ratio
        )
        block_big.intermediate.dense.weight.data[W_mask] = 0.0
        

        
        
        W_mask = get_wanda_mask(
            block_big.output.dense.weight.data,
            collector.accumulators[f'layer_{block_idx}']['mlp1_out'].sum_sq,
            sparsity_ratio=sparsity_ratio
        )
        block_big.output.dense.weight.data[W_mask] = 0.0
        

        
        
        
        
    breakpoint()
    # Classifier head
    W_mask = get_wanda_mask(
        teacher_model.classifier.weight.data,
        collector.accumulators[f'classifier'].sum_sq,
        sparsity_ratio=sparsity_ratio
    )
    teacher_model.classifier.weight.data[W_mask] = 0.0
    

    return student_model.state_dict()
    

    # return copied_state_dict
            











            
                
    
    

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
    teacher_model = PretrainedViT(model_name="google/vit-base-patch16-224", num_classes=100)
    # teacher_ckpt_path = "/home/yin178/Equvariant_Model_Distillation/outputs/CIFAR100/ViT/teacher/pretrained_finetuned/epoch=07.ckpt"
    teacher_ckpt_path = "/home/yin178/Equvariant_Model_Distillation/outputs/CIFAR100/pretrained_ViT/teacher/google/vit-base-patch16-224/best_fixed.ckpt"
    precision = torch.float32
    embed_dim = 384
    # embed_dim = 768
    scale_factor = 768 // embed_dim
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
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

    copied_state_ckpt = pretrained_vit_model_pruning(teacher_model, teacher_ckpt_path,
                                calibration_loader=calibration_loader, device=device)
    
    check_state_dicts_match(copied_state_ckpt, preserved_state_dict)
    
    
    student_model_ckpt = {}
    student_model_ckpt['state_dict'] = copied_state_ckpt
    # output_path = "/home/yin178/Equvariant_Model_Distillation/outputs/CIFAR100/pretrained_ViT/student/initialization/half_channel/zero_init.ckpt"
    # output_path = "./outputs/CIFAR100/pretrained_ViT/student/initialization/double_channel/zero_init_v2.ckpt"
    output_path = "./outputs/CIFAR100/pretrained_ViT/student/initialization/half_channel/zero_init_wanda.ckpt"
    
    
    # output_path = "/home/yin178/Equvariant_Model_Distillation/outputs/CIFAR100/pretrained_ViT/student/initialization/nonequ_pos_embed_real_zero_init.ckpt"
    breakpoint()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(student_model_ckpt, output_path)
    print("Student model initialized from teacher model.")