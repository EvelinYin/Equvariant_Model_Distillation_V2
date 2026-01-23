import torch
import math
from src.models.ViT.equ_vit import EquViT
from src.equ_lib.groups.rot45_group import Rot45Group
import torch.nn.functional as F

def rotate_image_pytorch(x, angle_degrees):
    """
    Rotate image using PyTorch's affine_grid and grid_sample.
    x: tensor of shape [B, C, H, W]
    angle_degrees: rotation angle in degrees (can be tensor or scalar)
    """
    batch_size = x.shape[0]
    
    # Convert to radians
    if isinstance(angle_degrees, (int, float)):
        angle = angle_degrees * (math.pi / 180)
        angles = torch.tensor([angle] * batch_size, device=x.device, dtype=x.dtype)
    else:
        angles = angle_degrees * (math.pi / 180)
    
    # Create rotation matrices
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)
    
    # Build theta matrices [B, 2, 3]
    theta = torch.zeros(batch_size, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = cos_angles
    theta[:, 0, 1] = sin_angles
    theta[:, 1, 0] = -sin_angles
    theta[:, 1, 1] = cos_angles
    
    # Generate sampling grid
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    
    # Apply rotation
    rotated = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    
    return rotated


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
    layer = layer.to(torch.float64)

    x = torch.randn(batchsize, 3, img_size, img_size).to(torch.float64)
    
    # Test all rotations: 45°, 90°, 135°, 180°, 225°, 270°, 315°
    errors = []
    for k in range(1, 8):  # k=0 is identity, so we test k=1,2,...,7
        angle = k * 45.0  # Rotation angle in degrees
        rx = rotate_image_pytorch(x, angle)
        
        out_x = layer(x)
        out_rx = layer(rx)
        
        error = torch.norm(out_x - out_rx).item()
        errors.append(error)
        print(f"Rotation {angle}°: error = {error}")
    
    # Return maximum error across all rotations
    max_error = max(errors)
    print(f"\nMaximum error across all rotations: {max_error}")
    print(f"Note: Some interpolation error is expected for non-90° rotations")
    return max_error
    

 
    
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
    rx = rotate_image_pytorch(x, 45.0)
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
        rx = rotate_image_pytorch(x, angle)
        
        out_x = layer(x)
        out_rx = layer(rx)
        
        # Report the relative equivariance error
        relative_error = (torch.norm(out_x - out_rx) / torch.norm(out_rx)).item()
        errors.append(relative_error)
        print(f"Rotation {angle}°: relative error = {relative_error}")
    
    # Return maximum relative error across all rotations
    max_error = max(errors)
    print(f"\nMaximum relative error across all rotations: {max_error}")
    print(f"Note: Some interpolation error is expected for non-90° rotations")
    return max_error