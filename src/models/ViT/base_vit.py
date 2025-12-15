import torch
import torch.nn as nn
import torch.nn.functional as F
from src.non_equ_lib.blocks import Mlp, Attention
from transformers import AutoModelForImageClassification

class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    def __init__(self, embed_dim=256, n_heads=8, mlp_ratio=4, qkv_bias=False, attn_drop=0.1, drop=0.1, \
                act_layer=nn.GELU, norm_layer=nn.LayerNorm, rope=None):
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.attn = Attention(dim=embed_dim, rope=rope, num_heads=n_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        

    def forward(self, x):
    
        # if self.norm1.norm.weight.dtype != x.dtype:
        #     self.norm1 = self.norm1.to(x.dtype)
        #     self.norm2 = self.norm2.to(x.dtype)

        x = self.dropout(x)
        x = x + self.attn(self.norm1(x))   
        x = x + self.mlp(self.norm2(x))
        return x




