import torch
import torch.nn as nn
from src.equ_lib.groups.flipping_group import FlipGroup
from src.utils import to_2tuple
from src.equ_lib.layers.shared_weight_linear import SharedWeightLinear
from src.equ_lib.layers.group_convolution import GroupConvolution
from src.equ_lib.layers.lifting_convolution import LiftingConvolution



class EquMlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks"""
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, bias=True, drop=0., group=FlipGroup()):
        super().__init__()
        self.in_features = in_features
        
        self.group = group  

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.hidden_features = hidden_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = SharedWeightLinear(in_features, hidden_features, bias=bias[0], group=group)

        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])

        self.fc2 = SharedWeightLinear(hidden_features, out_features, bias=bias[1], group=group)

        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        # if self.fc1.a.dtype != x.dtype:
        #     self.fc1 = self.fc1.to(x.dtype)
        #     self.fc2 = self.fc2.to(x.dtype)

        x = self.fc1(x)
        
        x = self.act(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.drop2(x)
        return x


class EquPatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=256, padding=0, group=FlipGroup()):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.liftinglayer = LiftingConvolution(in_channels=3, out_channels=3, 
                                                kernel_size=3, stride=1, padding=1, group=group)
        self.grouplayer = GroupConvolution(in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size,
                                     stride=patch_size, padding=padding, bias=True, group=group)
        # self.proj = FlippingEquConv_Custom(in_channels=in_chans, out_channels=embed_dim, inter_ kernel_size=patch_size, stride=patch_size, bias=True)
        # self.position_getter = PositionGetter()
    

    def forward(self, x, **kw):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0, f"Input image height ({H}) is not a multiple of patch size ({self.patch_size[0]})."
        assert W % self.patch_size == 0, f"Input image width ({W}) is not a multiple of patch size ({self.patch_size[1]})."

        x = self.liftinglayer(x)
        x = self.grouplayer(x)
        # breakpoint()
        x = x.flatten(3).permute(0, 3, 1, 2).flatten(2)  # B,2,C,H,W -> B,N,2C
        return x

