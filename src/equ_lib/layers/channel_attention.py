import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..groups.flipping_group import FlipGroup
from ..layers.shared_weight_linear import SharedWeightLinear

class EquAttentionPerChannel(nn.Module):

    def __init__(self, dim, rope=None, num_heads=8, qkv_bias=False, 
                 attn_drop=0., proj_drop=0., temperature=1.0,
                 group=FlipGroup()):
        
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.shared_q = SharedWeightLinear(dim, dim, bias=qkv_bias, group=group)
        self.shared_k = SharedWeightLinear(dim, dim, bias=qkv_bias, group=group)
        self.shared_v = SharedWeightLinear(dim, dim, bias=qkv_bias, group=group)

        self.attn_drop = nn.Dropout(attn_drop)
        
        self.proj = SharedWeightLinear(dim, dim, bias=qkv_bias, group=group)

        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope 
        
        self.temperature = temperature
        self.group = group  

    def forward(self, x, xpos=None):
        B, N, C = x.shape
        device = x.device
        dtype = x.dtype

        
        q = self.shared_q(x)
        k = self.shared_k(x)
        v = self.shared_v(x)
        

        qkv = torch.stack([q, k, v], dim=2)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
        q, k, v = [qkv[:, :, i] for i in range(3)]


        head_dim = C // self.group.order // self.num_heads
        
        x = self.group.get_channel_attention(q, k, v, head_dim, self.temperature, attn_drop=self.attn_drop)
        x = x.transpose(1, 2).reshape(B, N, -1)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # reshape (B,N,2,C) back to (B, N, 2C)
        return x.flatten(2)