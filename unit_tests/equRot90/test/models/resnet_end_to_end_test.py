import torch
import math
from src.models.ResNet50.equ_resnet import EquBottleneck, EquResNet, Bottleneck, ResNet
from src.equ_lib.groups.rot90_group import Rot90Group

def test_resnet_forward():
    in_channels = 3
    batchsize = 1
    # img_size = 224
    img_size = 225
    num_classes = 10
    S = 224

    # resnet_layer = ResNet( Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    # resnet_layer.eval()
    # x = torch.randn(batchsize, in_channels, S, S).to(torch.float64)
    # resnet_layer = resnet_layer.to(torch.float64)
    # resnet_layer(x)
    # breakpoint()

    rot90_group = Rot90Group()
    
    layer = EquResNet( EquBottleneck, [3, 4, 6, 3], 
                      num_classes=num_classes, 
                      group=rot90_group)
    
    


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
 
def test_resnet_backward():
    in_channels = 3
    batchsize = 1
    # img_size = 224
    img_size = 225
    num_classes = 10
    S = 224

    rot90_group = Rot90Group()
    
    layer = EquResNet( EquBottleneck, [3, 4, 6, 3], 
                      num_classes=num_classes, 
                      group=rot90_group)
    

    # layer.eval()
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