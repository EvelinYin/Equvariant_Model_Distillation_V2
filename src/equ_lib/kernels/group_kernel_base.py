import torch

class GroupKernelBase(torch.nn.Module):

    def __init__(self, group, kernel_size, in_channels, out_channels):
        """ Implements base class for the group convolution kernel. Stores grid
        defined over the group R^2 \rtimes H and it's transformed copies under
        all elements of the group H.
        
        """
        super().__init__()
        self.group = group

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Create a spatial kernel grid
        self.register_buffer("grid_R2", torch.stack(torch.meshgrid(
            torch.linspace(-1., 1., self.kernel_size),
            torch.linspace(-1., 1., self.kernel_size),
            indexing='ij'
        )).to(self.group.identity.device))

        # The kernel grid now also extends over the group H, as our input 
        # feature maps contain an additional group dimension
        self.register_buffer("grid_H", self.group.elements())
        self.register_buffer("transformed_grid_R2xH", self.create_transformed_grid_R2xH())

    def create_transformed_grid_R2xH(self):
        """Transform the created grid over R^2 \rtimes H by the group action of 
        each group element in H.
        
        This yields a set of grids over the group. In other words, a list of 
        grids, each index of which is the original grid over G transformed by
        a corresponding group element in H.
        """
        # Sample the group H.
        
        ## YOUR CODE STARTS HERE ##
        group_elements = self.group.elements()
        ## AND ENDS HERE ##

        # Transform the grid defined over R2 with the sampled group elements.
        # We again would like to end up with a grid of shape [2, |H|, kernel_size, kernel_size].
        
        ## YOUR CODE STARTS HERE ##
        transformed_grid_R2 = []
        for g_inverse in self.group.inverse(group_elements):
            transformed_grid_R2.append(
                self.group.left_action_on_R2(g_inverse, self.grid_R2)
            )
        transformed_grid_R2 = torch.stack(transformed_grid_R2, dim=1)
        ## AND ENDS HERE ##

        # Transform the grid defined over H with the sampled group elements. We want a grid of 
        # shape [|H|, |H|]. Make sure to stack the transformed like above (over the 1st dim).

        ## YOUR CODE STARTS HERE ##
        transformed_grid_H = []
        for g_inverse in self.group.inverse(group_elements):
            transformed_grid_H.append(
                self.group.product(
                    g_inverse, self.grid_H
                )
            )
        transformed_grid_H = torch.stack(transformed_grid_H, dim=1)
        ## AND ENDS HERE ##

        # Rescale values to between -1 and 1, we do this to please the torch
        # grid_sample function.
        transformed_grid_H = self.group.normalize_group_elements(transformed_grid_H)

        # Create a combined grid as the product of the grids over R2 and H
        # repeat R2 along the group dimension, and repeat H along the spatial dimension
        # to create a [3, |H|, |H|, kernel_size, kernel_size] grid
        transformed_grid = torch.cat(
            (
                transformed_grid_R2.view(
                    2,
                    group_elements.numel(),
                    1,
                    self.kernel_size,
                    self.kernel_size,
                ).repeat(1, 1, group_elements.numel(), 1, 1),
                transformed_grid_H.view(
                    1,
                    group_elements.numel(),
                    group_elements.numel(),
                    1,
                    1,
                ).repeat(1, 1, 1, self.kernel_size, self.kernel_size)
            ),
            dim=0
        )
        return transformed_grid


    def sample(self, sampled_group_elements):
        """ Sample convolution kernels for a given number of group elements

        arguments should include:
        :param sampled_group_elements: the group elements over which to sample
            the convolution kernels

        should return:
        :return kernels: filter bank extending over all input channels, 
            containing kernels transformed for all output group elements.
        """
        raise NotImplementedError()