import torch
import math
from ..kernels.interpolative_group_kernel import InterpolativeGroupKernel
from ..utils import check_input_compatibility


class GroupConvolution(torch.nn.Module):

    def __init__(self, group, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=True):
        super().__init__()

        self.kernel = InterpolativeGroupKernel(
            group=group,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels
        )
        
        self.padding = padding
        self.stride = stride
        
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        

    def forward(self, x):
        """ Perform lifting convolution

        @param x: Input sample [batch_dim,  group_dim, in_channels, spatial_dim_1, 
            spatial_dim_2]
        @return: Function on a homogeneous space of the group
            [batch_dim, out_channels, num_group_elements, spatial_dim_1, 
            spatial_dim_2]
        """

        # We now fold the group dimensions of our input into the input channel
        # dimension.
        
        # H, W = x.shape[-2], x.shape[-1]
        # check_input_compatibility(
        #     height=H,
        #     width=W,
        #     padding=self.padding,
        #     kernel_size=self.kernel.kernel_size,
        #     stride=self.stride
        # )
        
        # Permute to [batch_dim, in_channels, group_dim, spatial_dim_1, spatial_dim_2]
        x = x.permute(0, 2, 1, 3, 4)

        ## YOUR CODE STARTS HERE ##
        x = x.reshape(
            -1,
            x.shape[1] * x.shape[2],
            x.shape[3],
            x.shape[4]
        )
        ## AND ENDS HERE ##

        # We obtain convolution kernels transformed under the group.

        ## YOUR CODE STARTS HERE ##
        conv_kernels = self.kernel.sample()
        ## AND ENDS HERE ##

        # Apply group convolution, note that the reshape folds the 'output' group 
        # dimension of the kernel into the output channel dimension, and the 
        # 'input' group dimension into the input channel dimension.

        # Question: Do you see why we (can) do this?

        ## YOUR CODE STARTS HERE ##
        x = torch.nn.functional.conv2d(
            input=x,
            weight=conv_kernels.reshape(
                self.kernel.out_channels * self.kernel.group.elements().numel(),
                self.kernel.in_channels * self.kernel.group.elements().numel(),
                self.kernel.kernel_size,
                self.kernel.kernel_size
            ),
            padding=self.padding,
            stride=self.stride
        )
        ## AND ENDS HERE ##
        
        
        # Reshape [batch_dim, in_channels * num_group_elements, spatial_dim_1, 
        # spatial_dim_2] into [batch_dim, in_channels, num_group_elements, 
        # spatial_dim_1, spatial_dim_2], separating channel and group 
        # dimensions.
        x = x.view(
            -1,
            self.kernel.out_channels,
            self.kernel.group.elements().numel(),
            x.shape[-1],
            x.shape[-2],
        )
        
        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1, 1).contiguous()


        # We permute to [batch_dim, num_group_elements, out_channels, spatial_dim_1, spatial_dim_2] before returning.
        return x.permute(0, 2, 1, 3, 4)

