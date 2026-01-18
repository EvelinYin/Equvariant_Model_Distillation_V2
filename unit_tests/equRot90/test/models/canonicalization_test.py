import torch
import math
from src.models.canonicalizer.canonicalization import CanonicalizationNetwork
from src.equ_lib.groups.rot90_group import Rot90Group
from src.models.ViT.equ_vit import EquViT


def test_canonicalization_forward():
    in_channels = 3
    out_channels = 4  # Changed to 4 to match group order
    batchsize = 1
    S = 64

    rotation90_group = Rot90Group()

    layer = CanonicalizationNetwork(
        use_equ_layers=True,
        group=rotation90_group,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        hidden_channel_list=[8, 16],
        dropout_p=0.2,
    )
    
    
    model = EquViT( img_size=S,
                    patch_size=16,
                    in_channels=3,
                    num_classes=100,
                    embed_dim=768,
                    depth=3,
                    n_heads=12,
                    mlp_ratio=4,
                    pos_embed="SymmetricPosEmbed",
                    attention_per_channel=True,
                    group_attn_channel_pooling=False,
                    group=rotation90_group)
    
    

    layer.eval()
    layer = layer.to(torch.float64)
    
    model.eval()
    model = model.to(torch.float64).cuda()
    
    
    # Test all rotations: 90°, 180°, 270°
    total_diff = 0.0
    for i in range(100):
        x = torch.randn(batchsize, in_channels, S, S).to(torch.float64)
        
        # Test each rotation
        for k in [1, 2, 3]:  # k=0 is identity
            rx = torch.rot90(x, k=k, dims=(-2, -1))

            out_x, feat_x, indicator_x = layer(x)
            out_rx, feat_rx, indicator_rx = layer(rx)
            
            out_x = model(out_x)
            out_rx = model(out_rx)
            
            diff = torch.norm(out_x - out_rx).item()
            total_diff += diff

            
            
    
    # Average difference across all iterations and rotations
    avg_diff = total_diff / (100 * 3)
    
    return avg_diff


   
def test_canonicalization_backward():
    in_channels = 3
    out_channels = 4  # Changed to 4 to match group order
    batchsize = 1
    S = 32

    rotation90_group = Rot90Group()
    
    layer = CanonicalizationNetwork(
        use_equ_layers=True,
        group=rotation90_group,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        hidden_channel_list=[8, 16],
        dropout_p=0.2,
    )
    

    layer.train()  # Set to train mode for backward pass
    layer = layer.to(torch.float64).cuda()

    x = torch.randn(batchsize, in_channels, S, S).to(torch.float64).cuda()
    
    # Use 90° rotation for adversarial training
    rx = torch.rot90(x, k=1, dims=(-2, -1)).cuda()
    optimizer = torch.optim.Adam(layer.parameters(), lr=100000)
    
    for i in range(1000):
        optimizer.zero_grad()
        output, _, _ = layer(x)
        out_rx, _, _ = layer(rx)
        
        # Try to maximize the equivariance error 
        loss = -torch.norm(output - out_rx)
        if i % 100 == 0:
            print(f"Step {i}, loss: {loss.item()}, norm: {torch.norm(output).item()}")
        
        loss.backward()
        optimizer.step()
    
    
    layer.eval()
    x = torch.randn(batchsize, in_channels, S, S).to(torch.float64).cuda()
    
    # Test all rotations after adversarial training
    max_relative_error = 0.0
    for k in [1, 2, 3]:
        rx = torch.rot90(x, k=k, dims=(-2, -1)).cuda()

        out_x, _, _ = layer(x)
        out_rx, _, _ = layer(rx)
        
        # Report the relative equivariance error
        relative_error = (torch.norm(out_x - out_rx) / torch.norm(out_rx)).item()
        max_relative_error = max(max_relative_error, relative_error)
        print(f"Rotation {k*90}°: relative error = {relative_error}")
    
    return max_relative_error