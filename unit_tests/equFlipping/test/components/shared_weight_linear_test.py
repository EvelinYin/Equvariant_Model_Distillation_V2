import torch
import math
from src.equ_lib.layers.shared_weight_linear import SharedWeightLinear
from src.equ_lib.groups.flipping_group import FlipGroup
from src.equ_lib.utils import gflip, BN2C_to_B2CHW, B2CHW_to_BN2C

def test_shared_weight_linear_forward():
    in_channels = 4
    out_channels = 2
    N = 4
    sqrt_N = int(math.sqrt(N))
    batchsize = 1

    flipping_group = FlipGroup()
    
    layer = SharedWeightLinear( in_channel=in_channels,
                                out_channel=out_channels,
                                group=flipping_group)


    layer.eval()
    layer = layer.to(torch.float64)

    x = torch.randn(batchsize, N, 2, in_channels).to(torch.float64)
    
    # B,N,2,C --> B,2,C,sqrt(N),sqrt(N)
    reshaped_x = BN2C_to_B2CHW(x)
    fx = gflip(reshaped_x)
    
    # B,2,C,sqrt(N),sqrt(N) --> B,N,2,C
    reshaped_fx = B2CHW_to_BN2C(fx)
    
    out_x = layer(x)
    out_fx = layer(reshaped_fx)
    

    # B,N,2,C --> B,2,C,sqrt(N),sqrt(N)
    reshaped_out_x = BN2C_to_B2CHW(out_x)
    f_out_x = gflip(reshaped_out_x)
    
    # B,2,C,sqrt(N),sqrt(N) --> B,N,2,C
    f_out_x = B2CHW_to_BN2C(f_out_x)
    
    
    return torch.norm(f_out_x-out_fx).item()
 
    
def test_shared_weight_linear_backward():
    in_channels = 4
    out_channels = 2
    N = 4
    sqrt_N = int(math.sqrt(N))
    batchsize = 1

    flipping_group = FlipGroup()
    
    layer = SharedWeightLinear( in_channel=in_channels,
                                out_channel=out_channels,
                                group=flipping_group,
                                )


    layer.eval()
    layer = layer.to(torch.float64)

    
    
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
    
    x = torch.randn(batchsize, N, 2, in_channels).to(torch.float64)
    target = torch.randn(batchsize, N, 2, out_channels).to(torch.float64)
    
    for _ in range(100):
        optimizer.zero_grad()
        output = layer(x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
    
    

    layer.eval()
    x = torch.randn(batchsize, N, 2, in_channels).to(torch.float64)
    
    reshaped_x = BN2C_to_B2CHW(x)
    fx = gflip(reshaped_x)
    reshaped_fx = B2CHW_to_BN2C(fx)

    out_x = layer(x, debug=True)
    out_fx = layer(reshaped_fx, debug=True)
    f_out_x = gflip(BN2C_to_B2CHW(out_x))
    f_out_x = B2CHW_to_BN2C(f_out_x)

    return torch.norm(f_out_x-out_fx).item()