import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_grid(kernel_size):
    """Generates a standard centered grid for the kernel."""
    k = kernel_size
    # Create a grid from -1 to 1
    ax = torch.linspace(-1, 1, k)
    xx, yy = torch.meshgrid(ax, ax, indexing='xy')
    grid = torch.stack([xx, yy], dim=-1) # (k, k, 2)
    return grid.reshape(-1, 2) # (k*k, 2)

class EquivariantKernel(nn.Module):
    """
    Manages weights and produces the expanded filter bank.
    """
    def __init__(self, group, in_channels, out_channels, kernel_size):
        super().__init__()
        self.group = group
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 1. Store the canonical weights
        # For Lifting: weights defined on R2 -> shape (Out, In, k, k)
        # For Group: weights defined on G x R2 -> handled by `group_dim` in subclass
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.init_weights()
        
        # 2. Precompute the transformed grids for the group elements
        # This speeds up the forward pass (only done once or cached)
        self.register_buffer('transformed_grids', self._precompute_grids())

    def init_weights(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def _precompute_grids(self):
        """
        Computes the grid coordinates transformed by every group element.
        Output: (Num_Group_Elements, K, K, 2)
        """
        group_elements = self.group.elements() # [0, 1] for Flip
        
        # 1. Get Canonical Grid
        canonical_grid = generate_grid(self.kernel_size).to(self.weight.device) # (K*K, 2)
        
        # 2. Apply Inverse Group Action
        # We use inverse because we want to sample the pixel from the 'source' location
        # K_g(x) = K(g^{-1} x)
        inv_elements = self.group.inverse(group_elements)
        
        # 3. Transform
        # (G, K*K, 2)
        transformed_coords = self.group.left_action_on_R2(inv_elements, canonical_grid)
        
        # 4. Reshape for grid_sample: (G, K, K, 2)
        transformed_coords = transformed_coords.view(
            len(group_elements), self.kernel_size, self.kernel_size, 2
        )
        return transformed_coords

    def get_filter_bank(self):
        """
        Transforms the canonical weight into the full filter bank.
        """
        # weight: (C_out, C_in, H, W)
        # We want to sample this weight at the transformed coordinates.
        
        # grid_sample expects input (N, C, H, W) and grid (N, H_out, W_out, 2)
        # We treat the weight channels as the "Batch" dimension for sampling efficiency,
        # or we repeat the grid.
        
        n_group = self.transformed_grids.shape[0]
        
        # Expand weights to apply group action
        # We repeat the weights for each group element to sample from them
        w_flat = self.weight.unsqueeze(0).expand(n_group, -1, -1, -1, -1) 
        # Shape: (G, C_out, C_in, K, K)
        
        # Collapse first dimensions for grid_sample
        # Input: (G * C_out * C_in, 1, K, K) - treating spatial dims as image
        w_in = w_flat.reshape(-1, 1, self.kernel_size, self.kernel_size)
        
        # Grid: (G, K, K, 2). Repeat for channels
        grid = self.transformed_grids.unsqueeze(1).unsqueeze(2) # (G, 1, 1, K, K, 2)
        grid = grid.expand(-1, self.out_channels, self.in_channels, -1, -1, -1)
        grid = grid.reshape(-1, self.kernel_size, self.kernel_size, 2)
        
        # Sample
        # align_corners=True matches the geometric definition better [-1, 1]
        sampled_w = F.grid_sample(w_in, grid, align_corners=True, mode='bilinear')
        
        # Reshape back: (G, C_out, C_in, K, K)
        sampled_w = sampled_w.view(n_group, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        
        # Transpose to (C_out, C_in, G, K, K) usually, or prepared for Conv2d
        return sampled_w