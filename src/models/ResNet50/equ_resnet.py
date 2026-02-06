import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.ResNet50.base_resnet import ResNet, Bottleneck
from src.equ_lib.groups.flipping_group import FlipGroup
from src.equ_lib.layers.group_convolution import GroupConvolution
from src.equ_lib.layers.lifting_convolution import LiftingConvolution
from src.equ_lib.layers.shared_weight_linear import SharedWeightLinear
from src.equ_lib.layers.group_batch_norm import GroupBatchNorm
from src.equ_lib.equ_utils import BGCHW_to_BgCWH, BgCWH_to_BGCHW

class EquBottleneck(Bottleneck):
    def __init__(self, in_planes, planes, stride=1, downsample=None, group=FlipGroup()):
        super().__init__(in_planes, planes, stride, downsample)
        self.group = group
        
        
        self.conv1 = GroupConvolution(
            group=group,
            in_channels=in_planes,
            out_channels=planes,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False
        )
        
        self.bn1 = GroupBatchNorm(planes, num_groups=group.order)
        
        
        self.conv2 = GroupConvolution(
            group=group,
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            padding=1,
            stride=stride,
            bias=False
        )
        
        self.bn2 = GroupBatchNorm(planes, num_groups=group.order)
        
        
        
        self.conv3 = GroupConvolution(
            group=group,
            in_channels=planes,
            out_channels=planes * self.expansion,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False
        )
        
        self.bn3 = GroupBatchNorm(planes * self.expansion, num_groups=group.order)  
    def forward(self, x):
        # Implement the forward pass for the equivariant bottleneck
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class EquResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, group=FlipGroup()):
        super().__init__()
        
        self.group = group
        self.in_planes = 64
        
        self.lifting_layer = LiftingConvolution(
            group=self.group,
            in_channels=3,
            out_channels=3,
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )

        self.conv1 = GroupConvolution(
            group=self.group,
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            padding=3,
            stride=2,
            bias=False
        )
        
        self.bn1 = GroupBatchNorm(64, num_groups=group.order)
        
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1
        )
        
        
        # ResNet stages (C2, C3, C4, C5)
        self.layer1 = self._make_layer(block,  64, layers[0], stride=1, group=group)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, group=group)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, group=group)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, group=group)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.head = SharedWeightLinear(512 * block.expansion, num_classes, bias=True, group=group)
        self.head = nn.Linear(512 * block.expansion, num_classes)
        
        # Standard ResNet weight init
        for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            #     nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            # elif isinstance(m, nn.BatchNorm2d):
            
            if isinstance(m, GroupBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

        
    def _make_layer(self, block, planes, blocks, stride, group):
        downsample = None

        # Downsample if output channel count or stride mismatches
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                GroupConvolution(
                    group=group,
                    in_channels=self.in_planes,
                    out_channels=planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                GroupBatchNorm(planes * block.expansion, num_groups=group.order)
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, group=group))
        self.in_planes = planes * block.expansion

        # Additional blocks
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, group=group))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):

        B = x.shape[0]
        x = self.lifting_layer(x)
        
        x = self.conv1(x)
        
        
        # tmp = self.avgpool(x)
        # tmp = torch.flatten(tmp, 1)
        # tmp = tmp.view(B, self.group.order, -1).mean(dim=1)
        # return tmp

        x = self.bn1(x)
        
        x = BGCHW_to_BgCWH(x, self.group.order)
        
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = BgCWH_to_BGCHW(x, self.group.order)
        
    

       
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        x = self.avgpool(BGCHW_to_BgCWH(x, self.group.order))
        x = torch.flatten(x, 1)
        

        x = x.view(B, self.group.order, -1).mean(dim=1)

        x = self.head(x)

        return x