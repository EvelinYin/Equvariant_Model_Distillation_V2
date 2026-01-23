import torch
import math
from src.models.ViT.equ_vit import EquViT
from src.equ_lib.groups.rot45_group import Rot45Group
from src.equ_lib.equ_utils import BN2C_to_B2CHW, B2CHW_to_BN2C, BNC_to_B2CHW, B2CHW_to_BNC

import torch
import kornia as K

def test_vit_forward():
    depth = 1
    img_size = 224
    batchsize = 2

    rotation45_group = Rot45Group()
    
    layer = EquViT( img_size=img_size,
                    patch_size=16,
                    in_channels=3,
                    num_classes=100,
                    embed_dim=768,
                    depth=depth,
                    n_heads=12,
                    mlp_ratio=4,
                    pos_embed="SymmetricPosEmbed",
                    attention_per_channel=True,
                    group_attn_channel_pooling=False,
                    group=rotation45_group)
    
    layer.eval()
    layer = layer.to(torch.float64).cuda()

    x = torch.randn(batchsize, 3, img_size, img_size).to(torch.float64).cuda()
    
    # breakpoint()
    # Test all rotations: 45°, 90°, 135°, 180°, 225°, 270°, 315°
    errors = []
    for k in range(1, 8):  # k=0 is identity, so we test k=1,2,...,7
        angle = k * 45.0  # Rotation angle in degrees
        rx = K.geometry.rotate(x, torch.tensor([angle] * batchsize).cuda().to(torch.float64))
        
        out_x = layer(x)
        out_rx = layer(rx)
        breakpoint()
        error = torch.norm(out_x - out_rx).item()
        errors.append(error)
        print(f"Rotation {angle}°: error = {error}")
    
    # Return maximum error across all rotations
    return max(errors)
    

 
    
def test_vit_backward():
    depth = 1
    img_size = 224
    batchsize = 2

    rotation45_group = Rot45Group()
    
    layer = EquViT( img_size=img_size,
                    patch_size=16,
                    in_channels=3,
                    num_classes=100,
                    embed_dim=768,
                    depth=depth,
                    n_heads=12,
                    mlp_ratio=4,
                    pos_embed="SymmetricPosEmbed",
                    attention_per_channel=True,
                    group_attn_channel_pooling=False,
                    group=rotation45_group)
    

    layer.eval()
    layer = layer.to(torch.float64).cuda()

    x = torch.randn(batchsize, 3, img_size, img_size).to(torch.float64).cuda()
    
    # Use 45° rotation for the test
    rx = K.geometry.rotate(x, torch.tensor([45.0] * batchsize).cuda()).cuda()
    optimizer = torch.optim.Adam(layer.parameters(), lr=100000)
    
    for i in range(100):
        optimizer.zero_grad()
        output = layer(x)
        out_rx = layer(rx)
        
        # Try to maximize the equivariance error 
        loss = -torch.norm(output - out_rx)
        print(f"Step {i}, norm: {torch.norm(output)}")
        loss.backward()
        optimizer.step()
    
    
    layer.eval()
    x = torch.randn(batchsize, 3, img_size, img_size).to(torch.float64).cuda()
    
    # Test all rotations after training
    errors = []
    for k in range(1, 8):  # Test all 7 non-identity rotations
        angle = k * 45.0
        rx = K.geometry.rotate(x, torch.tensor([angle] * batchsize).cuda()).cuda()
        
        out_x = layer(x)
        out_rx = layer(rx)
        
        # Report the relative equivariance error
        relative_error = (torch.norm(out_x - out_rx) / torch.norm(out_rx)).item()
        errors.append(relative_error)
        print(f"Rotation {angle}°: relative error = {relative_error}")
    
    # Return maximum relative error across all rotations
    return max(errors)