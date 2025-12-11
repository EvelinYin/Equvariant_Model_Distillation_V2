import torch
import math
from src.models.ResNet50.equ_resnet import EquBottleneck, EquResNet
from src.equ_lib.groups.flipping_group import FlipGroup
from src.equ_lib.utils import gflip, BN2C_to_B2CHW, B2CHW_to_BN2C

def test_resnet_forward():
    in_channels = 3
    batchsize = 1
    num_classes = 10
    S = 224

    flipping_group = FlipGroup()
    
    layer = EquResNet( EquBottleneck, [3, 4, 6, 3], 
                      num_classes=num_classes, 
                      group=flipping_group)


    layer.eval()
    layer = layer.to(torch.float64)

    x = torch.randn(batchsize, in_channels, S, S).to(torch.float64)
    fx = torch.flip(x, dims=(-1,))
    
    
    out_x = layer(x)
    out_fx = layer(fx)
    
    breakpoint()
    
    return torch.norm(out_x-out_fx).item()
 
    
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