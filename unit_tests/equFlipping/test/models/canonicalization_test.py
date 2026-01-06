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
        use_equ_layers=False,
        group=flipping_group,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        hidden_channel_list=[8, 16],
        dropout_p=0.2,
    )

    layer.eval()
    layer = layer.to(torch.float64)
    
    diff = 0.0
    for i in range(100):
        x = torch.randn(batchsize, in_channels, S, S).to(torch.float64)
        fx = torch.flip(x, dims=(-1,))

        out_x, _, indicator_1 = layer(x)
        out_fx, _, indicator_2  = layer(fx)
        diff += torch.norm(out_x - out_fx).item()
    
    # breakpoint()

    return diff


   
def test_canonicalization_backward():
    in_channels = 3
    out_channels = 2
    batchsize = 1
    S = 32

    flipping_group = FlipGroup()
    
    layer = CanonicalizationNetwork(
        use_equ_layers=False,
        group=flipping_group,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        hidden_channel_list=[8, 16],
        dropout_p=0.2,
    )
    

    layer.eval()
    layer = layer.to(torch.float64).cuda()

    
    

    # target = torch.randn(batchsize, 100).to(torch.float64)
    # loss = torch.nn.functional.mse_loss(output, target)
        
    x = torch.randn(batchsize, in_channels, S, S).to(torch.float64).cuda()
    fx = torch.flip(x, dims=(-1,)).cuda()
    optimizer = torch.optim.Adam(layer.parameters(), lr=100000)
    
    for i in range(1000):
        optimizer.zero_grad()
        output, _, _ = layer(x)
        out_fx, _, _ = layer(fx)
        
        # Try to maximize the equivariance error 
        loss = -torch.norm(output-out_fx)
        print(f"Step {i}, norm: {torch.norm(output)}")
        loss.backward()
        optimizer.step()
    
    
    layer.eval()
    x = torch.randn(batchsize, in_channels, S, S).to(torch.float64).cuda()
    fx = torch.flip(x, dims=(-1,)).cuda()

    out_x, _, _ = layer(x)
    out_fx, _, _ = layer(fx)
    
    # Report the relative equivariance error
    return (torch.norm(out_x-out_fx)/torch.norm(out_fx)).item()