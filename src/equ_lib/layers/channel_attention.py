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
        self.attention_head_size = self.dim  // self.num_heads
        
        self.shared_q = SharedWeightLinear(dim, dim, bias=qkv_bias, group=group)
        self.shared_k = SharedWeightLinear(dim, dim, bias=qkv_bias, group=group)
        self.shared_v = SharedWeightLinear(dim, dim, bias=qkv_bias, group=group)

        self.attn_drop = nn.Dropout(attn_drop)
        
        self.proj = SharedWeightLinear(dim, dim, bias=qkv_bias, group=group)

        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope 
        
        self.temperature = temperature
        self.group = group  
    
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
                # 1. Reshape to separate the 'x' part and the 'extra' part
        #    New shape: (B, N, 2, H, D)
        #    Here, '2' represents the split between x and the extension.
        #    Note: Use self.attention_head_size (D), not the doubled size.
        
        new_x_shape = x.size()[:-1] + (self.group.order, self.num_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        
        # 2. Permute to bring Heads (H) to the front, keeping the '2' adjacent to 'D'
        #    Current dims: (0:B, 1:N, 2:2, 3:H, 4:D)
        #    Target dims:  (0:B, 3:H, 1:N, 2:2, 4:D)
        x = x.permute(0, 3, 1, 2, 4)
        
        # 3. Flatten the last two dimensions to fuse the '2' and 'D' back together
        #    Result shape: (B, H, N, 2*D)
        return x.flatten(3)

    
    def transpose_back(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Unpack the last dimension to separate the '2' (split) from 'D' (head size)
        #    Input Shape: (B, H, N, 2*D)
        #    New Shape:   (B, H, N, 2, D)
        new_shape = x.size()[:-1] + (self.group.order, self.attention_head_size)
        x = x.view(new_shape)

        # 2. Permute to restore the original order: Batch, Sequence, Split, Head, Dim
        #    Current dims: (0:B, 1:H, 2:N, 3:2, 4:D)
        #    Target dims:  (0:B, 2:N, 3:2, 1:H, 4:D)
        x = x.permute(0, 2, 3, 1, 4).contiguous()

        # 3. Flatten the last three dimensions (2, H, D) back into the full hidden size (2*C)
        #    Target Shape: (B, N, 2*C)
        #    Note: self.all_head_size is typically 'C', so we multiply by 2.
        final_shape = x.size()[:-3] + (self.group.order * self.dim,)

        return x.view(final_shape)
        
        

    def forward(self, x, xpos=None):
        B, N, C = x.shape
        device = x.device
        dtype = x.dtype

        
        q = self.shared_q(x)
        k = self.shared_k(x)
        v = self.shared_v(x)
        
        # breakpoint()

        # qkv = torch.stack([q, k, v], dim=2)
        # qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
        # q, k, v = [qkv[:, :, i] for i in range(3)]
        
        # breakpoint()
        
        
        # qt_trans = torch.load("./debug_outputs/q_trans_t.pt")
        # qt = torch.load("./debug_outputs/q_t.pt")
        
        # breakpoint()
        
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)
        

        head_dim = C // self.group.order // self.num_heads
        
        q = q * self.scale
        x = self.group.get_channel_attention(q, k, v, head_dim, self.temperature, attn_drop=self.attn_drop)
        
        x = self.transpose_back(x)

        
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # reshape (B,N,2,C) back to (B, N, 2C)
        return x.flatten(2)