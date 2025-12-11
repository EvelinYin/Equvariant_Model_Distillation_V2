import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..groups.flipping_group import FlipGroup


class SharedWeightLinear(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, group=FlipGroup):
        
        super().__init__()
        self.n_group_elements = group.order
        

        self.learnable_weights = nn.ParameterList(
            group.get_shared_weight_linear_weights(
            in_features=in_channel,
            out_features=out_channel
            )
        )
        

        self.bias = bias
        if bias:
            # self.learnable_bias = nn.ParameterList(
            #     group.get_shared_weight_linear_bias(
            #         out_features=out_channel
            #     )
            # )
            self.learnable_bias = nn.Parameter(torch.zeros(out_channel))

    def forward(self, x, debug=False):
        '''
        x: (batchsize, N, n_group_elements, C)
        output: (batchsize, N, n_group_elements, out_channel)
        
        '''
        
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (batchsize, N, n_group_elements*in_channel)
        
        # Manually construct weight matrix
        weight_rows = []
        n = len(self.learnable_weights)
        device = self.learnable_weights[0].device
        dtype = self.learnable_weights[0].dtype
        for i in range(n):
            # This formula (j - i) % n handles the cyclic shift automatically
            # Row 0: [0, 1, 2, 3] -> a, b, c, d
            # Row 1: [3, 0, 1, 2] -> d, a, b, c
            shifted_weights = [self.learnable_weights[(j - i) % n] for j in range(n)]
            weight_rows.append(torch.cat(shifted_weights, dim=-1))

        W = torch.cat(weight_rows, dim=0).to(device=device, dtype=dtype)
        
        # if debug:
        #     breakpoint()
        
        # Manually construct bias matrix
        if self.bias:
            stacked_bias = self.learnable_bias.repeat(self.n_group_elements).to(device=device, dtype=dtype)
        else:
            stacked_bias = None
        
        return F.linear(x, W, stacked_bias).reshape(
            x.shape[0],
            x.shape[1],
            self.n_group_elements,
            -1
        )  # (batchsize, N, n_group_elements, out_channel)


