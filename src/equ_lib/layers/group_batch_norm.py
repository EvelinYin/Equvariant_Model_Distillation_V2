import torch
import torch.nn as nn

class GroupBatchNorm(nn.Module):
    def __init__(self, num_channels, num_groups=4):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_channels)
        self.num_groups = num_groups
    
    def forward(self, x):
        # x: (B, G, C, H, W)
        x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4]) # (B*G, C, H, W)
        x = self.bn(x)
        x = x.reshape(-1, self.num_groups, x.shape[1], x.shape[2], x.shape[3]) # (B, G, C, H, W)
        return x
