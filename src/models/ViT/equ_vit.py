import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

# from blocks.equ_pos_embed import RoPE2D
from src.equ_lib.layers.equ_pos_embedding import FlippingSymmetricPosEmbed
from src.equ_lib.groups.flipping_group import FlipGroup
from .base_vit import TransformerBlock
from src.equ_lib.layers.channel_layer_norm import ChannelLayerNorm
from src.equ_lib.layers.channel_attention import EquAttentionPerChannel
from src.equ_lib.layers.shared_weight_linear import SharedWeightLinear
from src.equ_lib.equ_blocks import EquMlp, EquPatchEmbed
from src.equ_lib.layers.lifting_convolution import LiftingConvolution
# from equ_lib.layers.



class EquTransformerBlock(TransformerBlock):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, rope=None, attention_per_channel=False, group=FlipGroup()):
        super().__init__()
        self.norm1 = ChannelLayerNorm(dim, group=group)
        if attention_per_channel:
            self.attn = EquAttentionPerChannel(dim, rope=rope, num_heads=num_heads, 
                                               qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, group=group)
        else:
            breakpoint()
            # self.attn = FlippingEquAttention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = ChannelLayerNorm(dim, group=group)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = EquMlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                          act_layer=act_layer, drop=drop, group=group)
        self.dropout = nn.Dropout(drop)
        



class EquViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, stride=1, in_channels=3, embed_dim=256, \
                    depth=6, n_heads=8, mlp_ratio=4, num_classes=100, attn_drop=0.1, drop=None, \
                    norm_layer=None, pos_embed='SymmetricPosEmbed', attention_per_channel=False, \
                    group_attn_channel_pooling=False, linear_pooling=False, group=FlipGroup()   
                 ):
        super().__init__()
        print("Using ViT structure....")

        self.embed_dim = embed_dim
        

        self.patch_embed = EquPatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_channels,
                                         embed_dim=embed_dim, group=group)
        
        n_patches = self.patch_embed.n_patches
        
        if pos_embed == 'SymmetricPosEmbed':
            self.add_pos_embed = group.get_pos_embd(num_patches=n_patches, embed_dim=embed_dim, group_attn_channel_pooling=group_attn_channel_pooling)
        elif pos_embed == 'None-equ':
            self.non_equ_pos_embed = nn.Parameter(torch.empty(1, n_patches+1, embed_dim*2).normal_(std=0.02))
            self.add_pos_embed = lambda x: x + self.non_equ_pos_embed
            
        self.rope = None
        


        # if pos_embed is None:
        #     self.rope = None
        # elif pos_embed == 'BERT':
        #     self.rope = None
        #     self.pos_embed = nn.Parameter(torch.empty(1, n_patches, embed_dim).normal_(std=0.02))
        # elif pos_embed.startswith('RoPE'):
        #     freq = float(pos_embed[len('RoPE'):])
        #     self.rope = RoPE2D(freq=freq)


        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            EquTransformerBlock(embed_dim, n_heads, mlp_ratio, qkv_bias=True, 
                                rope=self.rope, attention_per_channel=attention_per_channel, group=group)
            for _ in range(depth)
        ])

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        self.group_attn_channel_pooling = group_attn_channel_pooling
        self.linear_pooling = linear_pooling
        self.linear_pooling_layer = None
        if group_attn_channel_pooling:
            self.group_cls_token = nn.Parameter(torch.randn(1, embed_dim))
        # elif linear_pooling:
        #     self.linear_pooling_layer = FlipInvariantLSLayer(embed_dim)

        # Classification head
        self.norm = ChannelLayerNorm(embed_dim, group=group)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self.group = group
        
    
        
    def forward(self, x, return_features=False):
        features = []
        
        
        # x, pos = self.patch_embed(x)  # (B, N, 2C)
        x = self.patch_embed(x)  # (B, N, 2C)
        
        
        
        B, N, C = x.shape

        
        ### Add group cls token
        if self.group_attn_channel_pooling:
            group_cls_token = torch.cat([self.group_cls_token] * self.group.order.item(), dim=-1)  # (1, 1, group_order*C)
            group_cls_token = group_cls_token.expand(B, -1, -1)
            x = torch.cat([group_cls_token, x], dim=1)
        
        ### Add cls token
        # cls_token = torch.cat([self.cls_token, self.cls_token], dim=-1)  # (1, 1, 2*C)
        cls_token = torch.cat([self.cls_token] * self.group.order.item(), dim=-1)  # (1, 1, group_order*C)
        cls_token = cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        x = self.add_pos_embed(x)
        

        # TODO: Delete this !!!!!!!!!!!!!!!!!!!!!!
        # x[:,:,C//2:] = 0.0
        
        for blk in self.blocks:
            # x = blk(x, pos)   #(B,N,2C)
            x = blk(x)   #(B,N,2C)
            features.append(x)
        
        
        x = self.norm(x)  # B,N,2C
        
        
        if self.group_attn_channel_pooling:
            group_cls_token = x[:, 1]  # group CLS token, (B,N,2C)
        
        x = x[:, 0]  # CLS token, (B,2C)
        
        if self.group_attn_channel_pooling:
            B, C2 = x.shape
            C2 = x.shape[1]
            # group_cls_token = self.group_cls_token.expand(B, -1)
            sims = []
            
            for g in range(self.group.order.item()):
                start_idx = g * (C2 // self.group.order.item())
                end_idx = (g + 1) * (C2 // self.group.order.item())
                sim_g = (x[:, start_idx:end_idx] * group_cls_token[:, start_idx:end_idx]).sum(dim=1)
                sims.append(sim_g)
            sim = torch.stack(sims, dim=-1)  # B, group_order
            prob = F.softmax(sim / 0.8, dim=-1)  # temperature
            
            
            # 1. Reshape x to separate the 4 groups
            # New shape: [Batch, order, Quarter_Size]
            x_reshaped = x.view(x.shape[0], self.group.order.item(), -1)

            # 2. Reshape prob to broadcast correctly
            # New shape: [Batch, order, 1]
            prob_reshaped = prob.unsqueeze(-1)

            # 3. Multiply and sum across the 'split' dimension (dim 1)
            # Result shape: [Batch, Quarter_Size]
            x = (x_reshaped * prob_reshaped).sum(dim=1)

        elif self.linear_pooling:
            x = self.linear_pooling_layer(x)

        # Store final features before classifier
        final_features = x 
        
        # breakpoint()
        # x = x[:, :x.size(1)//2] # Zero out the second half to get flipping invariant representation
        # x = x.view(x.size(0), 2, -1).sum(dim=1)[0] # B,C
        
        # breakpoint()
        if not self.group_attn_channel_pooling and not self.linear_pooling:
            x = x.view(x.size(0), self.group.order, -1).mean(dim=1) # B,C
            # x = x.view(x.size(0), 2, -1).sum(dim=1)[0] # B,C
        # x = x.view(x.size(0), 2, -1).max(dim=1)[0] # B,C
        
        
        
        #####This is to get rid of 2C and N dimension when not using CLS token
        # x = x.view(x.size(0), N, 2, -1).mean(dim=2).reshape(B, N, -1)  # B,N,C
        # side = int(x.size(1) ** 0.5)  # infer spatial shape if square
        # x = x.view(x.size(0), side, side, -1).permute(0, 3, 1, 2)  # -> (B, C, H, W)
        # # Global average pooling over
        # x = F.adaptive_avg_pool2d(x, output_size=1).squeeze(-1).squeeze(-1)  # -> (B, C)

        
        if self.head.weight.dtype != x.dtype:
            self.head = self.head.to(x.dtype)
        # Classifier
        logits = self.head(x)

        
        if return_features:
            return logits, final_features
        
        if self.group_attn_channel_pooling:
            return logits, sim
        else: 
            return logits
    
    