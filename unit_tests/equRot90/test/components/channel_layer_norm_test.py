import torch
import math
from src.equ_lib.layers.channel_layer_norm import ChannelLayerNorm
from src.equ_lib.groups.flipping_group import FlipGroup
from src.equ_lib.equ_utils import gflip, BN2C_to_B2CHW, B2CHW_to_BN2C

def test_channel_layer_norm_forward():
    in_channels = 4
    out_channels = 2
    N = 4
    sqrt_N = int(math.sqrt(N))
    batchsize = 1

    flipping_group = FlipGroup()
    
    layer = ChannelLayerNorm(dim=in_channels,
                            group=flipping_group)        


    layer.eval()
    layer = layer.to(torch.float64)

    x = torch.randn(batchsize, 2, in_channels, N, N).to(torch.float64)
    fx = gflip(x)

    ##########Test for B,N,2C
    reshaped_x = B2CHW_to_BN2C(x)
    reshaped_fx = B2CHW_to_BN2C(fx)
    

    out_x = layer(reshaped_x.flatten(2))
    out_fx = layer(reshaped_fx.flatten(2))
    
    # breakpoint()
    out_x = out_x.view(batchsize, N*N, 2, in_channels)
    reshaped_out_x = BN2C_to_B2CHW(out_x)
    f_out_x = gflip(reshaped_out_x)
    f_out_x = B2CHW_to_BN2C(f_out_x)
    # breakpoint()
    
    difference = 0.0
    difference += torch.norm(f_out_x.flatten(2)-out_fx).item()
    
    
    ##########Test for B,2C,H,W
    x = x.reshape(batchsize, 2*in_channels, N, N)
    fx = fx.reshape(batchsize, 2*in_channels, N, N)
    out_x = layer(x)
    out_fx = layer(fx)
    

    # B,2C,H,W --> B,2,C,sqrt(N),sqrt(N)
    out_x = out_x.view(batchsize, 2, in_channels, N, N)
    f_out_x = gflip(reshaped_out_x)
    
    difference += torch.norm(f_out_x-out_fx).item()
    return difference
 
    
def test_channel_layer_norm_backward():
    in_channels = 4
    out_channels = 2
    N = 4
    sqrt_N = int(math.sqrt(N))
    batchsize = 1

    flipping_group = FlipGroup()
    
    layer = ChannelLayerNorm(dim=in_channels,
                            group=flipping_group)   


    layer.eval()
    layer = layer.to(torch.float64)

    
    
    optimizer = torch.optim.Adam(layer.parameters(), lr=1000)
    
    x = torch.randn(batchsize, N, 2, in_channels).to(torch.float64)
    target = torch.randn(batchsize, N, 2, in_channels).to(torch.float64)
    
    for _ in range(1000):
        optimizer.zero_grad()
        output = layer(x.flatten(2))
        loss = torch.nn.functional.mse_loss(output, target.flatten(2))
        loss.backward()
        optimizer.step()
    
    

    layer.eval()
    x = torch.randn(batchsize, 2, in_channels, N, N).to(torch.float64)
    fx = gflip(x)

    ##########Test for B,N,2C
    reshaped_x = B2CHW_to_BN2C(x)
    reshaped_fx = B2CHW_to_BN2C(fx)
    

    out_x = layer(reshaped_x.flatten(2))
    out_fx = layer(reshaped_fx.flatten(2))
    
    # breakpoint()
    out_x = out_x.view(batchsize, N*N, 2, in_channels)
    reshaped_out_x = BN2C_to_B2CHW(out_x)
    f_out_x = gflip(reshaped_out_x)
    f_out_x = B2CHW_to_BN2C(f_out_x).flatten(2)
    

    return torch.norm(f_out_x-out_fx).item()