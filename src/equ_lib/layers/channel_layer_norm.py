import torch
import torch.nn as nn
import torch.nn.functional as F
from src.equ_lib.groups.flipping_group import FlipGroup

class ChannelLayerNorm(nn.Module):
    def __init__(self, dim, group=FlipGroup()):
        super().__init__()
        # Pretend we only have a single big LayerNorm
        self.norm = nn.LayerNorm(dim, eps=1e-12)
        self.group = group

    def forward(self, x):
        if self.norm.weight.dtype != x.dtype:
                    self.norm = self.norm.to(x.dtype)
        if x.dim() == 5:
            B, G, C, H, W = x.shape
            x = x.reshape(B, G*C, H, W)    #B, 2C, H, W   
                
        if x.dim() == 4:
            B, C, H, W = x.shape

            # B, n_group, C, H, W
            x = x.reshape(B, self.group.order, -1, H, W)
            
            normed_x = []
            for i in range(self.group.order):
                sliced_x = x[:, i, :, :, :]
                # normalize on ( B, H, W, C ) so the layer norm on C
                # Then reshape back to B, C, H, W
                normed_x.append(self.norm(sliced_x.permute(0,2,3,1)).permute(0,3,1,2))
            x = torch.stack(normed_x, dim=1)
            return x

        elif x.dim() == 3:

            B, N, C = x.shape
            
            x = x.reshape(B, N, self.group.order, -1)
            
            normed_x = []
            for i in range(self.group.order):
                normed_x.append(self.norm(x[:,:, i, :]))
                
            x = torch.stack(normed_x, dim=2)
            
            return x.flatten(2)

        else:
            breakpoint()