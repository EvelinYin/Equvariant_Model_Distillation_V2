import torch
import torch.nn as nn
import math
import kornia as K
from .base_group import GroupBase
from src.equ_lib.layers.equ_pos_embedding import Rotation45SymmetricPosEmbed

class Rot45Group(GroupBase):
    """
    The C8 Group representing 45-degree rotations.
    Elements: {0: Identity, 1: 45°, 2: 90°, 3: 135°, 4: 180°, 5: 225°, 6: 270°, 7: 315°}
    """
    def __init__(self):
        super().__init__(dimension=1, identity=[0.])
        
        self.order = torch.tensor(8)  # Eight elements: {0, 1, 2, 3, 4, 5, 6, 7}

    def elements(self):
        # 0 = Identity, 1 = 45°, 2 = 90°, ..., 7 = 315°
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=self.identity.device)

    def product(self, h, h_prime):
        # Addition modulo 8: rotations compose by adding angles
        # e.g., 45° + 90° = 135° → (1 + 2) % 8 = 3
        return torch.remainder(h + h_prime, 8)

    def inverse(self, h):
        # Inverse of k*45° is (8-k)*45° for k > 0
        # 0 -> 0, 1 -> 7, 2 -> 6, 3 -> 5, 4 -> 4, 5 -> 3, 6 -> 2, 7 -> 1
        return torch.remainder(8 - h, 8)

    def left_action_on_R2(self, h, x):
        """
        Applies the rotation matrix to a vector x.
        """
        matrices = self.matrix_representation(h)  # (2, 2)
        
        if matrices.dtype != x.dtype:
            matrices = matrices.to(x.dtype)

        return torch.tensordot(matrices, x, dims=1)
    
    def matrix_representation(self, h):
        """
        Returns 2x2 rotation matrices for the group elements.
        h can be a batch of elements.
        
        Rotation by k*45° counterclockwise:
        [[cos(k*45°), -sin(k*45°)],
         [sin(k*45°),  cos(k*45°)]]
        """
        # Convert h to angle in radians: k * 45° = k * π/4
        angle = h * (math.pi / 4)
        
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        
        # Construct 2x2 rotation matrix
        representation = torch.tensor([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ], device=self.identity.device)
        
        return representation
    
    def get_shared_weight_linear_weights(self, in_features, out_features):
        # Eight separate weight matrices for the eight rotation elements
        weights = []
        for _ in range(8):
            w = torch.empty(out_features, in_features)
            w = nn.Parameter(w)
            nn.init.kaiming_uniform_(w, a=(5 ** 0.5))
            w.data /= math.sqrt(8)  # Divide by sqrt(8) since we have 8 elements
            weights.append(w)
        
        return weights
    
    def get_channel_attention(self, q, k, v, head_dim, temperature=1.0, attn_drop=None):
        # Split into 8 groups corresponding to the 8 rotation elements
        group_dim = head_dim
        
        attns = []
        xs = []
        
        for i in range(8):
            start_idx = i * group_dim
            end_idx = (i + 1) * group_dim
            
            attn = q[:, :, :, start_idx:end_idx] @ k[:, :, :, start_idx:end_idx].transpose(-2, -1)
            attn = attn / temperature
            attn = attn.softmax(dim=-1)
            
            if attn_drop is not None:
                attn = attn_drop(attn)
            
            x = attn @ v[:, :, :, start_idx:end_idx]
            xs.append(x)
        x = torch.cat(xs, dim=-1)
        return x
    
    def roll_group(self, x):
        B, N, C = x.shape
        x = x.reshape(B, N, 8, -1)
        x = torch.roll(x, shifts=1, dims=2)
        return x.reshape(B, N, C)
    
    def trans(self, x, g):
        """
        Apply rotation transformation to image x.
        g: rotation element (0, 1, 2, 3, 4, 5, 6, 7) - can be scalar or tensor
        Uses PyTorch's affine_grid and grid_sample for rotation.
        """
        import torch.nn.functional as F
        
        # Handle scalar case
        if g == 0:
            return x
        else:
            angle = float(g * 45) * (math.pi / 180)  # Convert to radians
            
            # Create rotation matrix
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            
            # Handle different input shapes
            if x.ndim == 3:  # Shape: [B, H, W] - add channel dimension
                batch_size = x.shape[0]
                x_with_channel = x.unsqueeze(1)  # [B, 1, H, W]
                
                # Create affine transformation matrix [B, 2, 3]
                theta = torch.tensor([
                    [cos_a, sin_a, 0],
                    [-sin_a, cos_a, 0]
                ], device=x.device, dtype=x.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
                
                # Generate sampling grid
                grid = F.affine_grid(theta, x_with_channel.size(), align_corners=False)
                
                # Sample from input
                rotated = F.grid_sample(x_with_channel, grid, mode='bilinear', 
                                        padding_mode='zeros', align_corners=False)
                
                return rotated.squeeze(1)  # Remove channel dimension: [B, H, W]
            else:  # Shape: [B, C, H, W]
                batch_size = x.shape[0]
                
                # Create affine transformation matrix [B, 2, 3]
                theta = torch.tensor([
                    [cos_a, sin_a, 0],
                    [-sin_a, cos_a, 0]
                ], device=x.device, dtype=x.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
                
                # Generate sampling grid
                grid = F.affine_grid(theta, x.size(), align_corners=False)
                
                # Sample from input
                rotated = F.grid_sample(x, grid, mode='bilinear', 
                                        padding_mode='zeros', align_corners=False)
                
                return rotated
        
        
    
    def roll(self, x, g):
        if g == 0:
            return x
        else:
            C, G, H, W = x.shape
            return torch.roll(x, shifts=int(g), dims=1)
    
    
        
    def get_pos_embd(self, num_patches, embed_dim, group_attn_channel_pooling):
        return Rotation45SymmetricPosEmbed(num_patches=num_patches, embed_dim=embed_dim, group_attn_channel_pooling=group_attn_channel_pooling)
    
    
    def get_canonicalization_ref(self, device, dtype):
        return torch.linspace(0.0, 360.0, self.order + 1)[:self.order].to(device=device, dtype=dtype)  # [0, 45, 90, 135, 180, 225, 270, 315]
    
    def get_canonicalized_images(self, images, indicator):
        """
        Canonicalize images by applying the inverse rotation.
        indicator: which rotation to undo (0, 1, 2, 3, 4, 5, 6, 7)
        """
        return K.geometry.rotate(images, -indicator), indicator.view(-1, 1, 1, 1)

    def normalize_group_elements(self, h):
        # Normalize to [-1, 1] range
        # 0 -> -1, 1 -> -0.714, ..., 7 -> 1
        return 2 * (h / 7) - 1