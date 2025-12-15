import torch
import torch.nn as nn
from src.equ_lib.groups.flipping_group import FlipGroup

class FlippingSymmetricPosEmbed(nn.Module):
    def __init__(self, num_patches, embed_dim, group_attn_channel_pooling=False):
        super().__init__()
        
        assert embed_dim % 2 == 0, "Embedding dim must be divisible by group order for splitting"
        self.num_patches = num_patches
        self.H = int(num_patches ** 0.5)
        self.W = int(num_patches ** 0.5)
        self.C = embed_dim
        
        
        # 1. Define parameters for the LEFT half of the image
        # We calculate width of the learnable area
        self.W_half = (self.W + 1) // 2 
        
        # We learn the full 2C embedding for the left side
        # self.pos_embed_left = nn.Parameter(torch.randn(1, self.H, self.W_half, 2*self.C))
        self.pos_embed_left = nn.Parameter(torch.empty(1, self.H, self.W_half, 2*self.C).normal_(std=0.02))

        # 2. CLS token pos embed must be self-symmetric
        # We learn size C and repeat it
        self.cls_pos_half = nn.Parameter(torch.randn(1, 1, self.C))
        
        self.group_attn_channel_pooling = group_attn_channel_pooling
        if group_attn_channel_pooling:
            self.group_cls_pos_half = nn.Parameter(torch.randn(1, 1, self.C))

    def forward(self, x):
        # x shape: (B, N_patches + 1, 2C)
        # --- Construct Spatial Grid Embeddings ---
        
        # Get the left side
        left_grid = self.pos_embed_left # (1, H, W_half, 2C)
        
        # Create the right side by flipping spatially AND swapping channels
        # 1. Flip spatially (W dimension)
        right_grid_flipped = torch.flip(left_grid, dims=[2])
        
        # # 2. Swap channels: Split into (C, C), swap, concat back
        c1, c2 = right_grid_flipped.chunk(2, dim=-1)
        right_grid = torch.cat([c2, c1], dim=-1)
        # right_grid = right_grid_flipped
        
        # 3. Concatenate Left and Right
        grid = torch.cat([left_grid, right_grid], dim=2)
            
        # Flatten grid to (1, N_patches, 2C)
        pos_embed_patches = grid.flatten(1, 2)

        # --- Construct CLS Token Embedding ---
        cls_pos = torch.cat([self.cls_pos_half, self.cls_pos_half], dim=-1) # (1, 1, 2C)
        
        if self.group_attn_channel_pooling:
            group_cls_pos = torch.cat([self.group_cls_pos_half, self.group_cls_pos_half], dim=-1) # (1, 1, 2C)

        # --- Combine ---
        if self.group_attn_channel_pooling:
            full_pos_embed = torch.cat([cls_pos, group_cls_pos, pos_embed_patches], dim=1)
        else:
            full_pos_embed = torch.cat([cls_pos, pos_embed_patches], dim=1)
        
        
        return x + full_pos_embed
