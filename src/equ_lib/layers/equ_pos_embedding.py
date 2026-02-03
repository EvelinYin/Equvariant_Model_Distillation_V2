import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FlippingSymmetricPosEmbed(nn.Module):
    def __init__(self, num_patches, embed_dim, group_attn_channel_pooling=False):
        super().__init__()
        
        assert embed_dim % 2 == 0, "Embedding dim must be divisible by group order for splitting"
        self.num_patches = num_patches
        self.H = int(num_patches ** 0.5)
        self.W = int(num_patches ** 0.5)
        self.C = embed_dim
        
        
        # 1. Define parameters for the LEFT half of the image
        # We calculate width of the learnable area
        self.W_half = (self.W + 1) // 2 
        
        # We learn the full 2C embedding for the left side
        # self.pos_embed_left = nn.Parameter(torch.randn(1, self.H, self.W_half, 2*self.C))
        self.pos_embed_left = nn.Parameter(torch.empty(1, self.H, self.W_half, 2*self.C).normal_(std=0.02))

        # 2. CLS token pos embed must be self-symmetric
        # We learn size C and repeat it
        self.cls_pos_half = nn.Parameter(torch.randn(1, 1, self.C))
        
        self.group_attn_channel_pooling = group_attn_channel_pooling
        if group_attn_channel_pooling:
            self.group_cls_pos_half = nn.Parameter(torch.randn(1, 1, self.C))

    def forward(self, x):
        # x shape: (B, N_patches + 1, 2C)
        # --- Construct Spatial Grid Embeddings ---
        
        # Get the left side
        left_grid = self.pos_embed_left # (1, H, W_half, 2C)
        
        # Create the right side by flipping spatially AND swapping channels
        # 1. Flip spatially (W dimension)
        right_grid_flipped = torch.flip(left_grid, dims=[2])
        
        # # 2. Swap channels: Split into (C, C), swap, concat back
        c1, c2 = right_grid_flipped.chunk(2, dim=-1)
        right_grid = torch.cat([c2, c1], dim=-1)
        # right_grid = right_grid_flipped
        
        # 3. Concatenate Left and Right
        grid = torch.cat([left_grid, right_grid], dim=2)
            
        # Flatten grid to (1, N_patches, 2C)
        pos_embed_patches = grid.flatten(1, 2)

        # --- Construct CLS Token Embedding ---
        cls_pos = torch.cat([self.cls_pos_half, self.cls_pos_half], dim=-1) # (1, 1, 2C)
        
        if self.group_attn_channel_pooling:
            group_cls_pos = torch.cat([self.group_cls_pos_half, self.group_cls_pos_half], dim=-1) # (1, 1, 2C)

        # --- Combine ---
        if self.group_attn_channel_pooling:
            full_pos_embed = torch.cat([cls_pos, group_cls_pos, pos_embed_patches], dim=1)
        else:
            full_pos_embed = torch.cat([cls_pos, pos_embed_patches], dim=1)
        
        
        return x + full_pos_embed



class Rotation90SymmetricPosEmbed(nn.Module):
    def __init__(self, num_patches, embed_dim, group_attn_channel_pooling=False):
        super().__init__()
        
        assert embed_dim % 4 == 0, "Embedding dim must be divisible by group order (4) for splitting"
        self.num_patches = num_patches
        self.H = int(num_patches ** 0.5)
        self.W = int(num_patches ** 0.5)
        self.C = embed_dim
        
        assert self.H == self.W, "Height and Width must be equal for rotation symmetry"
        
        # 1. Define parameters for one quadrant (top-right triangle including diagonal)
        # For a rotation-symmetric grid, we only need to learn ~1/4 of the positions
        # We'll learn the upper-right triangular region (including diagonal)
        # and generate the rest through rotations
        
        # Calculate the learnable region size
        # For an HxW grid, we learn positions where row <= col (upper triangle)
        self.learnable_positions = []
        for i in range(self.H):
            for j in range(self.W):
                if i <= j:  # Upper-right triangle including diagonal
                    self.learnable_positions.append((i, j))
        
        num_learnable = len(self.learnable_positions)
        
        # We learn the full 4C embedding for the learnable positions
        self.pos_embed_learnable = nn.Parameter(torch.empty(1, num_learnable, 4*self.C).normal_(std=0.02))

        # 2. CLS token pos embed must be self-symmetric under all rotations
        # We learn size C and repeat it 4 times
        self.cls_pos_quarter = nn.Parameter(torch.randn(1, 1, self.C))
        
        self.group_attn_channel_pooling = group_attn_channel_pooling
        if group_attn_channel_pooling:
            self.group_cls_pos_quarter = nn.Parameter(torch.randn(1, 1, self.C))

    def _create_rotation_grid(self):
        """
        Creates the full spatial grid by rotating the learnable positions.
        For each position (i,j) in the learnable region, we compute its 3 rotated versions
        and assign appropriate channel permutations.
        """
        device = self.pos_embed_learnable.device
        
        # Initialize full grid
        grid = torch.zeros(1, self.H, self.W, 4*self.C, device=device)
        
        # Fill in the grid using learnable positions and their rotations
        for idx, (i, j) in enumerate(self.learnable_positions):
            learned_embed = self.pos_embed_learnable[:, idx, :]  # (1, 4C)
            
            # Split into 4 channels
            c0, c1, c2, c3 = learned_embed.chunk(4, dim=-1)  # Each is (1, C)
            
            # Position 0°: (i, j) - Identity
            grid[:, i, j, :] = torch.cat([c0, c1, c2, c3], dim=-1)
            
            # Position 90° CCW: (i,j) -> (j, H-1-i)
            # Channels rotate: c0->c1->c2->c3->c0
            i_90 = j
            j_90 = self.H - 1 - i
            grid[:, i_90, j_90, :] = torch.cat([c1, c2, c3, c0], dim=-1)
            
            # Position 180°: (i,j) -> (H-1-i, W-1-j)
            # Channels rotate: c0->c2, c1->c3, c2->c0, c3->c1
            i_180 = self.H - 1 - i
            j_180 = self.W - 1 - j
            grid[:, i_180, j_180, :] = torch.cat([c2, c3, c0, c1], dim=-1)
            
            # Position 270° CCW: (i,j) -> (W-1-j, i)
            # Channels rotate: c0->c3->c2->c1->c0
            i_270 = self.W - 1 - j
            j_270 = i
            grid[:, i_270, j_270, :] = torch.cat([c3, c0, c1, c2], dim=-1)
        
        return grid

    def forward(self, x):
        # x shape: (B, N_patches + 1, 4C) or (B, N_patches + 2, 4C) with group pooling
        
        # --- Construct Spatial Grid Embeddings ---
        grid = self._create_rotation_grid()  # (1, H, W, 4C)
            
        # Flatten grid to (1, N_patches, 4C)
        pos_embed_patches = grid.flatten(1, 2)

        # --- Construct CLS Token Embedding ---
        # CLS token is invariant under rotations, so all 4 channels are the same
        cls_pos = torch.cat([self.cls_pos_quarter] * 4, dim=-1)  # (1, 1, 4C)
        
        if self.group_attn_channel_pooling:
            group_cls_pos = torch.cat([self.group_cls_pos_quarter] * 4, dim=-1)  # (1, 1, 4C)

        # --- Combine ---
        if self.group_attn_channel_pooling:
            full_pos_embed = torch.cat([cls_pos, group_cls_pos, pos_embed_patches], dim=1)
        else:
            full_pos_embed = torch.cat([cls_pos, pos_embed_patches], dim=1)
        
        return x + full_pos_embed

    
class Rotation45SymmetricPosEmbed(nn.Module):
    """
    Positional embedding that is invariant to 45-degree rotations.

    This is achieved by:
    1. Learning a base positional embedding of shape (H, W)
    2. Rotating it by all 8 multiples of 45 degrees (0°, 45°, 90°, ..., 315°)
    3. Averaging all rotated versions
    4. Using this average for all channels

    The resulting embedding is rotationally symmetric, so when the input is
    rotated by any multiple of 45 degrees, the positional embedding remains the same.
    """

    def __init__(self, num_patches, embed_dim, group_attn_channel_pooling=False):
        super().__init__()

        self.num_patches = num_patches
        self.H = int(num_patches ** 0.5)
        self.W = int(num_patches ** 0.5)
        self.C = embed_dim

        assert self.H == self.W, "Height and Width must be equal for rotation symmetry"

        # Learnable base positional embedding (single channel, will be averaged across rotations)
        # Shape: (1, 1, H, W) for use with grid_sample
        self.pos_embed_base = nn.Parameter(torch.empty(1, 1, self.H, self.W).normal_(std=0.02))

        # CLS token positional embedding (single value, expanded to all channels)
        self.cls_pos_base = nn.Parameter(torch.randn(1, 1, 1) * 0.02)

        self.group_attn_channel_pooling = group_attn_channel_pooling
        if group_attn_channel_pooling:
            self.group_cls_pos_base = nn.Parameter(torch.randn(1, 1, 1) * 0.02)

        # Pre-compute rotation angles (8 rotations at 45-degree increments)
        angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]
        self.angles_rad = [a * math.pi / 180 for a in angles_deg]

    def _create_rotation_symmetric_embedding(self):
        """
        Create rotation-symmetric embedding by averaging over all 8 rotations.
        Uses bilinear interpolation for non-90-degree rotations.
        """
        device = self.pos_embed_base.device
        dtype = self.pos_embed_base.dtype

        B, C, H, W = self.pos_embed_base.shape  # (1, 1, H, W)

        avg_embed = torch.zeros_like(self.pos_embed_base)

        for angle in self.angles_rad:
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            # Create rotation matrix for affine_grid
            # This rotates around the center of the image
            theta = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0]
            ], device=device, dtype=dtype).unsqueeze(0)  # (1, 2, 3)

            # Create sampling grid
            grid = F.affine_grid(theta, (B, C, H, W), align_corners=True)

            # Rotate the embedding using bilinear interpolation
            # padding_mode='border' extends edge values for positions outside the grid
            rotated = F.grid_sample(
                self.pos_embed_base,
                grid,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            )

            avg_embed = avg_embed + rotated

        # Average over all 8 rotations
        avg_embed = avg_embed / 8.0

        return avg_embed

    def forward(self, x):
        # x shape: (B, N_patches + 1, C) or (B, N_patches + 2, C) with group pooling
        # C here is the actual embedding dimension from x (e.g., 6144)
        actual_C = x.shape[-1]

        # Get rotation-symmetric embedding
        symmetric_embed = self._create_rotation_symmetric_embedding()  # (1, 1, H, W)

        # Expand to all channels: (1, 1, H, W) -> (1, actual_C, H, W)
        symmetric_embed = symmetric_embed.expand(-1, actual_C, -1, -1)

        # Reshape to (1, H*W, actual_C)
        pos_embed_patches = symmetric_embed.permute(0, 2, 3, 1).flatten(1, 2)  # (1, H*W, actual_C)

        # CLS token embedding (expanded to all channels)
        cls_pos = self.cls_pos_base.expand(-1, -1, actual_C)  # (1, 1, actual_C)

        if self.group_attn_channel_pooling:
            group_cls_pos = self.group_cls_pos_base.expand(-1, -1, actual_C)  # (1, 1, actual_C)
            full_pos_embed = torch.cat([cls_pos, group_cls_pos, pos_embed_patches], dim=1)
        else:
            full_pos_embed = torch.cat([cls_pos, pos_embed_patches], dim=1)

        return x + full_pos_embed



# class Rotation45SymmetricPosEmbed(nn.Module):
#     def __init__(self, num_patches, embed_dim, group_attn_channel_pooling=False):
#         super().__init__()
        
#         assert embed_dim % 8 == 0, "Embedding dim must be divisible by group order (8) for splitting"
#         self.num_patches = num_patches
#         self.H = int(num_patches ** 0.5)
#         self.W = int(num_patches ** 0.5)
#         self.C = embed_dim
        
#         assert self.H == self.W, "Height and Width must be equal for rotation symmetry"
        
#         # For 45-degree rotation symmetry, we use radial-angular decomposition
#         # Key insight: A position (i,j) can be described by:
#         # - Distance from center (radius): invariant under rotation
#         # - Angle from center: changes by 45° increments under rotation
        
#         # Precompute center coordinates
#         self.center = (self.H - 1) / 2.0
        
#         # Create coordinate grid
#         coords = []
#         for i in range(self.H):
#             for j in range(self.W):
#                 # Compute relative coordinates from center
#                 y = i - self.center
#                 x = j - self.center
                
#                 # Compute radius (distance from center)
#                 r = math.sqrt(x**2 + y**2)
                
#                 # Compute angle in [0, 2π), but quantize to 8 sectors (45° each)
#                 angle = math.atan2(y, x)  # Range: [-π, π]
#                 if angle < 0:
#                     angle += 2 * math.pi  # Range: [0, 2π)
                
#                 # Quantize angle to nearest 45° sector (0-7)
#                 sector = int((angle + math.pi/8) / (math.pi/4)) % 8
                
#                 coords.append((i, j, r, sector))
        
#         # Group positions by (radius, sector) to identify unique learnable positions
#         # Positions with same radius and sector (modulo 8) share embeddings
#         from collections import defaultdict
#         radius_sector_to_positions = defaultdict(list)
        
#         for i, j, r, sector in coords:
#             # Quantize radius to reduce parameters
#             r_quantized = round(r * 10) / 10  # Quantize to 0.1 precision
#             radius_sector_to_positions[(r_quantized, 0)].append((i, j, sector))
        
#         # We learn embeddings for unique radii only (sector 0)
#         # Each radius gets a C-dimensional embedding
#         # The full 8C embedding is created by rotating through sectors
#         self.unique_radii = sorted(set(r for r, s in radius_sector_to_positions.keys()))
#         self.num_unique_radii = len(self.unique_radii)
        
#         # Create mapping from grid position to (radius_idx, sector)
#         self.position_to_radius_sector = {}
#         for i in range(self.H):
#             for j in range(self.W):
#                 y = i - self.center
#                 x = j - self.center
#                 r = math.sqrt(x**2 + y**2)
#                 r_quantized = round(r * 10) / 10
                
#                 angle = math.atan2(y, x)
#                 if angle < 0:
#                     angle += 2 * math.pi
#                 sector = int((angle + math.pi/8) / (math.pi/4)) % 8
                
#                 # Find radius index
#                 radius_idx = self.unique_radii.index(r_quantized)
#                 self.position_to_radius_sector[(i, j)] = (radius_idx, sector)
        
#         # Learn C-dimensional embedding for each unique radius
#         self.radial_embed = nn.Parameter(torch.empty(self.num_unique_radii, self.C).normal_(std=0.02))
        
#         # CLS token pos embed must be self-symmetric under all rotations
#         # We learn size C and repeat it 8 times
#         self.cls_pos_quarter = nn.Parameter(torch.randn(1, 1, self.C))
        
#         self.group_attn_channel_pooling = group_attn_channel_pooling
#         if group_attn_channel_pooling:
#             self.group_cls_pos_quarter = nn.Parameter(torch.randn(1, 1, self.C))

#     def _create_rotation_grid(self):
#         """
#         Creates the full spatial grid using radial embeddings and angular rotations.
#         For each position (i,j):
#         - Find its radius -> get base C-dimensional embedding
#         - Find its sector (0-7) -> cyclically shift the 8 channels accordingly
#         """
#         device = self.radial_embed.device
        
#         # Initialize full grid
#         grid = torch.zeros(1, self.H, self.W, 8*self.C, device=device)
        
#         for i in range(self.H):
#             for j in range(self.W):
#                 radius_idx, sector = self.position_to_radius_sector[(i, j)]
                
#                 # Get base radial embedding (C-dimensional)
#                 base_embed = self.radial_embed[radius_idx]  # (C,)
                
#                 # Create 8 copies and cyclically rotate based on sector
#                 # Sector 0: [c, c, c, c, c, c, c, c]
#                 # Sector 1: [c, c, c, c, c, c, c, c] but rotated in channel groups
#                 # ... and so on
                
#                 # Expand to 8 groups
#                 full_embed = base_embed.unsqueeze(0).expand(8, -1)  # (8, C)
                
#                 # Apply cyclic shift based on sector
#                 # This ensures rotation equivariance
#                 full_embed = torch.roll(full_embed, shifts=sector, dims=0)  # (8, C)
                
#                 # Flatten to (8C,)
#                 grid[0, i, j, :] = full_embed.flatten()
        
#         return grid

#     def forward(self, x):
#         # x shape: (B, N_patches + 1, 8C) or (B, N_patches + 2, 8C) with group pooling
        
#         # --- Construct Spatial Grid Embeddings ---
#         grid = self._create_rotation_grid()  # (1, H, W, 8C)
            
#         # Flatten grid to (1, N_patches, 8C)
#         pos_embed_patches = grid.flatten(1, 2)

#         # --- Construct CLS Token Embedding ---
#         # CLS token is invariant under rotations, so all 8 channels are the same
#         cls_pos = torch.cat([self.cls_pos_quarter] * 8, dim=-1)  # (1, 1, 8C)
        
#         if self.group_attn_channel_pooling:
#             group_cls_pos = torch.cat([self.group_cls_pos_quarter] * 8, dim=-1)  # (1, 1, 8C)

#         # --- Combine ---
#         if self.group_attn_channel_pooling:
#             full_pos_embed = torch.cat([cls_pos, group_cls_pos, pos_embed_patches], dim=1)
#         else:
#             full_pos_embed = torch.cat([cls_pos, pos_embed_patches], dim=1)

#         return x + full_pos_embed

