import torch

from src.equ_lib.layers.lifting_convolution import LiftingConvolution
from src.equ_lib.groups.flipping_group import FlipGroup
from src.equ_lib.utils import gflip

def test_lifting_convolution_forward():
    in_channels = 3
    out_channels = 2
    kernel_size = 3
    batchsize = 1
    padding = 0
    S = 6

    flipping_group = FlipGroup()
    
    layer = LiftingConvolution( group=flipping_group,
                                kernel_size=kernel_size,
                                in_channels=in_channels,
                                out_channels=out_channels,
                                padding=padding)


    layer.eval()
    layer = layer.to(torch.float64)

    x = torch.randn(batchsize, in_channels, S, S).to(torch.float64)
    fx = torch.flip(x, dims=(-1,))

    out_x = layer(x)
    out_fx = layer(fx)

    f_out_x = gflip(out_x)
    
    return torch.norm(f_out_x-out_fx).item()
 
    
def test_lifting_convolution_backward():
    in_channels = 3
    out_channels = 2
    kernel_size = 3
    batchsize = 1
    padding = 0
    S = 6

    flipping_group = FlipGroup()
    
    layer = LiftingConvolution( group=flipping_group,
                                kernel_size=kernel_size,
                                in_channels=in_channels,
                                out_channels=out_channels,
                                padding=padding)


    layer = layer.to(torch.float64)
    
    
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
    
    x = torch.randn(batchsize, in_channels, S, S).to(torch.float64)
    target = torch.randn(batchsize, 2, out_channels, S - kernel_size + 1, S - kernel_size + 1).to(torch.float64)
    
    for _ in range(100):
        optimizer.zero_grad()
        output = layer(x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
    
    
    layer.eval()
    x = torch.randn(batchsize, in_channels, S, S).to(torch.float64)
    fx = torch.flip(x, dims=(-1,))

    out_x = layer(x)
    out_fx = layer(fx)

    f_out_x = gflip(out_x)
    
    return torch.norm(f_out_x-out_fx).item()