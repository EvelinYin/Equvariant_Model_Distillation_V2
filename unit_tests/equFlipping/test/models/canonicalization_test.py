import torch
import math
from src.models.canonicalizer.canonicalization import CanonicalizationNetwork
from src.equ_lib.groups.flipping_group import FlipGroup


def test_canonicalization_forward():
    in_channels = 3
    out_channels = 2
    batchsize = 1
    S = 32

    flipping_group = FlipGroup()

    layer = CanonicalizationNetwork(
        group=flipping_group,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        hidden_channel_list=[8, 16],
        dropout_p=0.0,
    )

    layer.eval()
    layer = layer.to(torch.float64)

    x = torch.randn(batchsize, in_channels, S, S).to(torch.float64)
    fx = torch.flip(x, dims=(-1,))

    out_x, _, indicator_1 = layer(x)
    out_fx, _, indicator_2  = layer(fx)
    
    breakpoint()

    return torch.norm(out_x - out_fx).item()