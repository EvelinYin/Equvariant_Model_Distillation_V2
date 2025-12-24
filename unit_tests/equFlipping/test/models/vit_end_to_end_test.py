import torch
import math
from src.models.ViT.equ_vit import EquViT
from src.equ_lib.groups.flipping_group import FlipGroup
from src.equ_lib.equ_utils import gflip, BN2C_to_B2CHW, B2CHW_to_BN2C, BNC_to_B2CHW, B2CHW_to_BNC

def test_vit_forward():
    depth = 1
    img_size = 224
    batchsize = 2

    flipping_group = FlipGroup()
    
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
                    group=flipping_group)
    
    


    layer.eval()
    layer = layer.to(torch.float64)

    x = torch.randn(batchsize, 3, img_size, img_size).to(torch.float64)
    fx = torch.flip(x, dims=(-1,))
    
    
    out_x = layer(x)
    out_fx = layer(fx)
    
    return torch.norm(out_x-out_fx).item()
    
    # cls_token = out_x[:, :1, :] # cls token
    # f_cls_token = out_fx[:, :1, :] 
    
    # out_x = out_x[:, 1:] # remove cls token
    # out_fx = out_fx[:, 1:] # remove cls token
    # f_out_x = gflip(BNC_to_B2CHW(out_x))
    
    # f_out_x = B2CHW_to_BNC(f_out_x)
    
    # permuted_f_cls_token = flipping_group.roll_group(f_cls_token)
    
    # difference = torch.norm(f_out_x-out_fx).item() + torch.norm(cls_token-permuted_f_cls_token).item()
    
    # print("Difference in ViT forward:", torch.norm(f_out_x-out_fx).item())
    # print("Difference for cls token:", torch.norm(cls_token-f_cls_token).item())
    # breakpoint()
    
 
    
def test_vit_backward():
    depth = 1
    img_size = 224
    batchsize = 2

    flipping_group = FlipGroup()
    
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
                    group=flipping_group)
    

    layer.eval()
    layer = layer.to(torch.float64).cuda()

    
    

    # target = torch.randn(batchsize, 100).to(torch.float64)
    # loss = torch.nn.functional.mse_loss(output, target)
        
    x = torch.randn(batchsize, 3, img_size, img_size).to(torch.float64).cuda()
    fx = torch.flip(x, dims=(-1,)).cuda()
    optimizer = torch.optim.Adam(layer.parameters(), lr=100000)
    
    for i in range(1000):
        optimizer.zero_grad()
        output = layer(x)
        out_fx = layer(fx)
        
        # Try to maximize the equivariance error 
        loss = -torch.norm(output-out_fx)
        print(f"Step {i}, norm: {torch.norm(output)}")
        loss.backward()
        optimizer.step()
    
    
    layer.eval()
    x = torch.randn(batchsize, 3, img_size, img_size).to(torch.float64).cuda()
    fx = torch.flip(x, dims=(-1,)).cuda()

    out_x = layer(x)
    out_fx = layer(fx)
    
    # Report the relative equivariance error
    return (torch.norm(out_x-out_fx)/torch.norm(out_fx)).item()