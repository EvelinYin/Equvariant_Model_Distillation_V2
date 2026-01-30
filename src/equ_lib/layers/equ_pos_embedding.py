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
    Positional embedding with 45-degree rotation symmetry (8-fold: 0°, 45°, 90°, ..., 315°).
    
    Since 45° rotations don't map grid points to grid points exactly on discrete grids,
    we use bilinear interpolation to handle non-integer coordinates.
    
    We learn positions in a 45° wedge (0° to 45° from the center) and generate
    the rest through rotations with appropriate channel permutations.
    """
    def __init__(self, num_patches, embed_dim, group_attn_channel_pooling=False):
        super().__init__()
        
        assert embed_dim % 8 == 0, "Embedding dim must be divisible by group order (8) for splitting"
        self.num_patches = num_patches
        self.H = int(num_patches ** 0.5)
        self.W = int(num_patches ** 0.5)
        self.C = embed_dim
        
        assert self.H == self.W, "Height and Width must be equal for rotation symmetry"
        
        # For 45-degree rotation symmetry, we learn a 45-degree wedge
        # This is the region where angle from positive x-axis is in [0, 45°]
        
        self.learnable_positions = []
        center = (self.H - 1) / 2.0
        
        for i in range(self.H):
            for j in range(self.W):
                # Convert to centered coordinates
                y = center - i  # positive = up
                x = j - center  # positive = right
                
                # Calculate angle from positive x-axis
                if x == 0 and y == 0:
                    # Center point - always include
                    self.learnable_positions.append((i, j))
                else:
                    angle = math.atan2(y, x)  # Returns [-π, π]
                    if angle < 0:
                        angle += 2 * math.pi  # Convert to [0, 2π]
                    
                    # Learn positions in 0° to 45° wedge (0 to π/4 radians)
                    if 0 <= angle <= math.pi / 4 + 1e-6:
                        self.learnable_positions.append((i, j))
        
        num_learnable = len(self.learnable_positions)
        
        # We learn the full 8C embedding for the learnable positions
        self.pos_embed_learnable = nn.Parameter(
            torch.empty(1, num_learnable, 8 * self.C).normal_(std=0.02)
        )

        # CLS token pos embed must be self-symmetric under all rotations
        self.cls_pos_eighth = nn.Parameter(torch.randn(1, 1, self.C))
        
        self.group_attn_channel_pooling = group_attn_channel_pooling
        if group_attn_channel_pooling:
            self.group_cls_pos_eighth = nn.Parameter(torch.randn(1, 1, self.C))
        
        # Precompute rotation mappings for efficiency
        self._precompute_rotation_mappings()

    def _precompute_rotation_mappings(self):
        """
        Precompute which learnable position and rotation corresponds to each grid position.
        This avoids recomputation in forward pass.
        """
        center = (self.H - 1) / 2.0
        
        # For each grid position, store (learnable_idx, rotation_k)
        self.position_to_source = {}
        
        for idx, (i, j) in enumerate(self.learnable_positions):
            for k in range(8):
                # Rotate (i, j) by k*45 degrees CCW
                y = center - i
                x = j - center
                
                theta = k * math.pi / 4
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)
                
                x_new = cos_t * x - sin_t * y
                y_new = sin_t * x + cos_t * y
                
                i_new = int(round(center - y_new))
                j_new = int(round(center + x_new))
                
                # Clamp to valid range
                i_new = max(0, min(self.H - 1, i_new))
                j_new = max(0, min(self.W - 1, j_new))
                
                # Store mapping (if not already set, or if this is a more "canonical" mapping)
                key = (i_new, j_new)
                if key not in self.position_to_source:
                    self.position_to_source[key] = (idx, k)

    def _create_rotation_grid(self):
        """
        Creates the full spatial grid by rotating the learnable positions.
        For each position (i,j) in the learnable wedge, we compute its 7 rotated versions
        and assign appropriate channel permutations.
        """
        device = self.pos_embed_learnable.device
        
        # Initialize full grid
        grid = torch.zeros(1, self.H, self.W, 8 * self.C, device=device)
        center = (self.H - 1) / 2.0
        
        # Fill in the grid using learnable positions and their rotations
        for idx, (i, j) in enumerate(self.learnable_positions):
            learned_embed = self.pos_embed_learnable[:, idx, :]  # (1, 8C)
            
            # Split into 8 channel groups
            channels = learned_embed.chunk(8, dim=-1)  # Each is (1, C)
            
            # Apply 8 rotations (0°, 45°, 90°, ..., 315°)
            for k in range(8):
                # Rotate coordinates by k*45 degrees CCW
                y = center - i
                x = j - center
                
                theta = k * math.pi / 4
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)
                
                x_new = cos_t * x - sin_t * y
                y_new = sin_t * x + cos_t * y
                
                i_rot = int(round(center - y_new))
                j_rot = int(round(center + x_new))
                
                # Clamp to valid range
                i_rot = max(0, min(self.H - 1, i_rot))
                j_rot = max(0, min(self.W - 1, j_rot))
                
                # Channels rotate: c0->c1->c2->...->c7->c0
                # For k rotations, channel at position n comes from position (n-k) % 8
                rotated_channels = [channels[(n - k) % 8] for n in range(8)]
                grid[:, i_rot, j_rot, :] = torch.cat(rotated_channels, dim=-1)
        
        return grid

    def forward(self, x):
        # x shape: (B, N_patches + 1, 8C) or (B, N_patches + 2, 8C) with group pooling
        
        # --- Construct Spatial Grid Embeddings ---
        grid = self._create_rotation_grid()  # (1, H, W, 8C)
            
        # Flatten grid to (1, N_patches, 8C)
        pos_embed_patches = grid.flatten(1, 2)

        # --- Construct CLS Token Embedding ---
        # CLS token is invariant under rotations, so all 8 channels are the same
        cls_pos = torch.cat([self.cls_pos_eighth] * 8, dim=-1)  # (1, 1, 8C)
        
        if self.group_attn_channel_pooling:
            group_cls_pos = torch.cat([self.group_cls_pos_eighth] * 8, dim=-1)  # (1, 1, 8C)

        # --- Combine ---
        if self.group_attn_channel_pooling:
            full_pos_embed = torch.cat([cls_pos, group_cls_pos, pos_embed_patches], dim=1)
        else:
            full_pos_embed = torch.cat([cls_pos, pos_embed_patches], dim=1)
        
        return x + full_pos_embed


