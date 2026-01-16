import torch
import math
from src.equ_lib.layers.channel_attention import EquAttentionPerChannel
from src.equ_lib.groups.flipping_group import FlipGroup
from src.equ_lib.equ_utils import gflip, BN2C_to_B2CHW, B2CHW_to_BN2C

def test_channel_attention_forward():
    in_channels = 4
    out_channels = 2
    N = 4
    sqrt_N = int(math.sqrt(N))
    batchsize = 1

    flipping_group = FlipGroup()
    
    layer = EquAttentionPerChannel(dim=in_channels,
                                           num_heads=2,
                                           group=flipping_group)        


    layer.eval()
    layer = layer.to(torch.float64)
    # breakpoint()

    x = torch.randn(batchsize, N, 2, in_channels).to(torch.float64)
    
    # B,N,2,C --> B,2,C,sqrt(N),sqrt(N)
    reshaped_x = BN2C_to_B2CHW(x)
    fx = gflip(reshaped_x)
    
    # B,2,C,sqrt(N),sqrt(N) --> B,N,2,C
    reshaped_fx = B2CHW_to_BN2C(fx)
    
    out_x = layer(x.flatten(2))
    out_fx = layer(reshaped_fx.flatten(2))
    

    # B,N,2,C --> B,2,C,sqrt(N),sqrt(N)
    out_x = out_x.view(batchsize, N, 2, in_channels)
    reshaped_out_x = BN2C_to_B2CHW(out_x)
    f_out_x = gflip(reshaped_out_x)
    
    # B,2,C,sqrt(N),sqrt(N) --> B,N,2,C
    f_out_x = B2CHW_to_BN2C(f_out_x).flatten(2)
    
    return torch.norm(f_out_x-out_fx).item()
 
    
def test_channel_attention_backward():
    in_channels = 4
    out_channels = 2
    N = 4
    sqrt_N = int(math.sqrt(N))
    batchsize = 1

    flipping_group = FlipGroup()
    
    layer = EquAttentionPerChannel(dim=in_channels,
                                           num_heads=2,
                                           group=flipping_group)    


    layer.eval()
    layer = layer.to(torch.float64)

    
    
    optimizer = torch.optim.Adam(layer.parameters(), lr=100000)
    
    x = torch.randn(batchsize, N, 2, in_channels).to(torch.float64)
    target = torch.randn(batchsize, N, 2, in_channels).to(torch.float64) * 1000000
    
    for i in range(10000):
        optimizer.zero_grad()
        output = layer(x.flatten(2))
        # loss = torch.nn.functional.mse_loss(output, target.flatten(2))
        
        # with torch.no_grad():
            # To get fx and the layer's prediction on fx
        reshaped_x = BN2C_to_B2CHW(x)
        fx = gflip(reshaped_x)
        reshaped_fx = B2CHW_to_BN2C(fx)
        out_x = layer(x.flatten(2))
        out_x = out_x.view(batchsize, N, 2, in_channels)
        f_out_x = gflip(BN2C_to_B2CHW(out_x))
        f_out_x = B2CHW_to_BN2C(f_out_x).flatten(2)
        
        
        # loss = -torch.norm(output)
        loss = -torch.norm((f_out_x-output))
        print(f"Step {i}, norm: {torch.norm(output)}")
        loss.backward()
        optimizer.step()
    
    

    layer.eval()
    x = torch.randn(batchsize, N, 2, in_channels).to(torch.float64)
    

    reshaped_x = BN2C_to_B2CHW(x)
    fx = gflip(reshaped_x)
    reshaped_fx = B2CHW_to_BN2C(fx)

    out_x = layer(x.flatten(2))
    out_fx = layer(reshaped_fx.flatten(2))
    
    out_x = out_x.view(batchsize, N, 2, in_channels)
    f_out_x = gflip(BN2C_to_B2CHW(out_x))
    f_out_x = B2CHW_to_BN2C(f_out_x).flatten(2)

    # return torch.norm(f_out_x-out_fx).item()
    return (torch.norm((f_out_x-out_fx))/torch.norm(out_fx)).item()