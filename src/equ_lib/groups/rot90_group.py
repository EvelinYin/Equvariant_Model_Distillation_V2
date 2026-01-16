import torch
import torch.nn as nn
import math
from .base_group import GroupBase
from src.equ_lib.layers.equ_pos_embedding import Rotation90SymmetricPosEmbed

class Rot90Group(GroupBase):
    """
    The C4 Group representing 90-degree rotations.
    Elements: {0: Identity, 1: 90°, 2: 180°, 3: 270°}
    """
    def __init__(self):
        super().__init__(dimension=1, identity=[0.])
        
        self.order = torch.tensor(4)  # Four elements: {0, 1, 2, 3}

    def elements(self):
        # 0 = Identity, 1 = 90°, 2 = 180°, 3 = 270°
        return torch.tensor([0, 1, 2, 3], device=self.identity.device)

    def product(self, h, h_prime):
        # Addition modulo 4: rotations compose by adding angles
        # e.g., 90° + 180° = 270° → (1 + 2) % 4 = 3
        return torch.remainder(h + h_prime, 4)

    def inverse(self, h):
        # Inverse of k*90° is (4-k)*90° for k > 0
        # 0 -> 0, 1 -> 3, 2 -> 2, 3 -> 1
        return torch.remainder(4 - h, 4)

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
        
        Rotation by k*90° counterclockwise:
        [[cos(k*90°), -sin(k*90°)],
         [sin(k*90°),  cos(k*90°)]]
        """
        # Convert h to angle in radians: k * 90° = k * π/2
        angle = h * (math.pi / 2)
        
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        
        # Construct 2x2 rotation matrix
        representation = torch.tensor([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ], device=self.identity.device)
        
        return representation
    
    def get_shared_weight_linear_weights(self, in_features, out_features):
        # Four separate weight matrices for the four rotation elements
        weights = []
        for _ in range(4):
            w = torch.empty(out_features, in_features)
            w = nn.Parameter(w)
            nn.init.kaiming_uniform_(w, a=(5 ** 0.5))
            w.data /= math.sqrt(4)  # Divide by sqrt(4) since we have 4 elements
            weights.append(w)
        
        return weights
    
    def get_channel_attention(self, q, k, v, head_dim, temperature=1.0, attn_drop=None):
        # Split into 4 groups corresponding to the 4 rotation elements
        group_dim = head_dim
        
        attns = []
        xs = []
        
        for i in range(4):
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
        x = x.reshape(B, N, 4, -1)
        x = torch.roll(x, shifts=1, dims=2)
        return x.reshape(B, N, C)
    
    def trans(self, x, g):
        """
        Apply rotation transformation to image x.
        g: rotation element (0, 1, 2, 3)
        """
        if g == 0:
            return x
        elif g == 1:
            # 90° counterclockwise
            return torch.rot90(x, k=1, dims=[-2, -1])
        elif g == 2:
            # 180°
            return torch.rot90(x, k=2, dims=[-2, -1])
        elif g == 3:
            # 270° counterclockwise (or 90° clockwise)
            return torch.rot90(x, k=3, dims=[-2, -1])
    
    def roll(self, x, g):
        if g == 0:
            return x
        else:
            C, G, H, W = x.shape
            return torch.roll(x, shifts=int(g), dims=1)
    
    
        
    def get_pos_embd(self, num_patches, embed_dim, group_attn_channel_pooling):
        return Rotation90SymmetricPosEmbed(num_patches=num_patches, embed_dim=embed_dim, group_attn_channel_pooling=group_attn_channel_pooling)
    
    

    def get_canonicalization_ref(self, device, dtype):
        # 0 -> Identity, 1 -> 90°, 2 -> 180°, 3 -> 270°
        return torch.tensor([0., 1., 2., 3.], device=device, dtype=dtype)
    
    def get_canonicalized_images(self, images, indicator):
        """
        Canonicalize images by applying the inverse rotation.
        indicator: which rotation to undo (0, 1, 2, 3)
        """
        B = images.shape[0]
        canonicalized_images = torch.zeros_like(images)
        
        for i in range(B):
            g = int(indicator[i].item())
            # Apply inverse rotation
            inv_g = int(self.inverse(torch.tensor(g)).item())
            canonicalized_images[i] = self.trans(images[i], inv_g)
        
        indicator = indicator.view(-1, 1, 1, 1)
        return canonicalized_images, indicator

    def normalize_group_elements(self, h):
        # Normalize to [-1, 1] range
        # 0 -> -1, 1 -> -0.33, 2 -> 0.33, 3 -> 1
        return 2 * (h / 3) - 1