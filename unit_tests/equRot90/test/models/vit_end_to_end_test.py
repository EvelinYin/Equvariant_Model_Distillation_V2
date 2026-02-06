import torch
import math
from src.models.ViT.equ_vit import EquViT
from src.equ_lib.groups.rot90_group import Rot90Group

def test_vit_forward():
    depth = 1
    img_size = 224
    batchsize = 2

    rotation90_group = Rot90Group()
    
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
                    group=rotation90_group)
    
    layer.eval()
    layer = layer.to(torch.float64)

    x = torch.randn(batchsize, 3, img_size, img_size).to(torch.float64)
    
    # Test all rotations: 90°, 180°, 270°
    errors = []
    for k in [1, 2, 3]:  # k=0 is identity, so we test k=1,2,3
        rx = torch.rot90(x, k=k, dims=(-2, -1))
        
        out_x = layer(x)
        out_rx = layer(rx)
        
        error = torch.norm(out_x - out_rx).item()
        errors.append(error)
        print(f"Rotation {k*90}°: error = {error}")
    
    # Return maximum error across all rotations
    return max(errors)
    
    # cls_token = out_x[:, :1, :] # cls token
    # r_cls_token = out_rx[:, :1, :] 
    
    # out_x = out_x[:, 1:] # remove cls token
    # out_rx = out_rx[:, 1:] # remove cls token
    # r_out_x = rotation90_group.trans(BNC_to_B2CHW(out_x), k)
    
    # r_out_x = B2CHW_to_BNC(r_out_x)
    
    # permuted_r_cls_token = rotation90_group.roll_group(r_cls_token)
    
    # difference = torch.norm(r_out_x-out_rx).item() + torch.norm(cls_token-permuted_r_cls_token).item()
    
    # print("Difference in ViT forward:", torch.norm(r_out_x-out_rx).item())
    # print("Difference for cls token:", torch.norm(cls_token-r_cls_token).item())
    # breakpoint()
    
 
    
def test_vit_backward():
    depth = 1
    img_size = 224
    batchsize = 2

    rotation90_group = Rot90Group()
    
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
                    group=rotation90_group)
    

    layer.eval()
    layer = layer.to(torch.float64).cuda()

    x = torch.randn(batchsize, 3, img_size, img_size).to(torch.float64).cuda()
    
    # Use 90° rotation for the test
    rx = torch.rot90(x, k=1, dims=(-2, -1)).cuda()
    optimizer = torch.optim.Adam(layer.parameters(), lr=100000)
    
    for i in range(200):
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
    for k in [1, 2, 3]:
        rx = torch.rot90(x, k=k, dims=(-2, -1)).cuda()
        
        out_x = layer(x)
        out_rx = layer(rx)
        
        # Report the relative equivariance error
        relative_error = (torch.norm(out_x - out_rx) / torch.norm(out_rx)).item()
        errors.append(relative_error)
        print(f"Rotation {k*90}°: relative error = {relative_error}")
    
    # Return maximum relative error across all rotations
    return max(errors)