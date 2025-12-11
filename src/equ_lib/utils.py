import torch
import math

def bilinear_interpolation(signal, grid):
    """ Obtain signal values for a set of gridpoints through bilinear interpolation.
    
    @param signal: Tensor containing pixel values [C, H, W] or [N, C, H, W]
    @param grid: Tensor containing coordinate values [2, H, W] or [2, N, H, W]
    """
    # If signal or grid is a 3D array, add a dimension to support grid_sample.
    if len(signal.shape) == 3:
        signal = signal.unsqueeze(0)
    if len(grid.shape) == 3:
        grid = grid.unsqueeze(1)
    
    # Grid_sample expects [N, H, W, 2] instead of [2, N, H, W]
    grid = grid.permute(1, 2, 3, 0)
    
    # Grid sample expects YX instead of XY.
    grid = torch.roll(grid, shifts=1, dims=-1)
    
    return torch.nn.functional.grid_sample(
        signal,
        grid,
        padding_mode='zeros',
        align_corners=True,
        mode="bilinear"
    )

def trilinear_interpolation(signal, grid):
    """ 
    
    @param signal: Tensor containing pixel values [C, D, H, W] or [N, C, D, H, W]
    @param grid: Tensor containing coordinate values [3, D, H, W] or [3, N, D, H, W]
    """
    # If signal or grid is a 4D array, add a dimension to support grid_sample.
    if len(signal.shape) == 4:
        signal = signal.unsqueeze(0)
    if len(grid.shape) == 4:
        grid = grid.unsqueeze(1)

    # Grid_sample expects [N, D, H, W, 3] instead of [3, N, D, H, W]
    grid = grid.permute(1, 2, 3, 4, 0)
    
    # Grid sample expects YX instead of XY.
    grid = torch.roll(grid, shifts=1, dims=-1)
    
    return torch.nn.functional.grid_sample(
        signal, 
        grid,
        padding_mode='zeros',
        align_corners=True,
        mode="bilinear" # actually trilinear in this case...
    )


def gflip(x, dims=1):
    # Roll the input tensor along the group axis (axis=1) by k steps
    x_rolled = torch.roll(x, shifts=1, dims=dims)
    x_flip = torch.flip(x_rolled, dims=(-1,))
    return x_flip.to(x.device)


def BN2C_to_B2CHW(x):
    B, N, _, C = x.shape
    sqrt_N = int(math.sqrt(N))
    return x.reshape(B, sqrt_N, sqrt_N, 2, C).permute(0,3,4,1,2)

def B2CHW_to_BN2C(x):
    B, _, C, H, W = x.shape
    N = H * W
    return x.permute(0,3,4,1,2).reshape(B, N, 2, C)
    

def check_input_compatibility(height, width, padding, kernel_size, stride):
        # Calculate Input + Total Padding - Kernel
        h_numerator = height + (2 * padding) - kernel_size
        w_numerator = width + (2 * padding) - kernel_size

        # Check if the numerator is divisible by stride
        if h_numerator % stride != 0:
            raise ValueError(
                f"Input Height ({height}) results in discarded pixels with "
                f"kernel={kernel_size}, stride={stride}, padding={padding}."
            )
        
        if w_numerator % stride != 0:
            raise ValueError(
                f"Input Width ({width}) results in discarded pixels with "
                f"kernel={kernel_size}, stride={stride}, padding={padding}."
            )