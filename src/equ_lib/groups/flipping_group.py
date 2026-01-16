import torch
import torch.nn as nn
import math
from .base_group import GroupBase
from src.equ_lib.layers.equ_pos_embedding import FlippingSymmetricPosEmbed

class FlipGroup(GroupBase):
    """
    The C2 Group representing horizontal flips.
    Elements: {0: Identity, 1: Flip}
    """
    def __init__(self):
        super().__init__(dimension=1, identity=[0.])
        
        self.order = torch.tensor(2)  # Only two elements: {0, 1}

    def elements(self):
        # 0 = Identity, 1 = Flip
        return torch.tensor([0, 1], device=self.identity.device)

    def product(self, h, h_prime):
        # Addition modulo 2: 
        # e*e=e (0+0=0), e*f=f (0+1=1), f*e=f (1+0=1), f*f=e (1+1=0)
        return torch.remainder(h + h_prime, 2)

    def inverse(self, h):
        # The inverse of a flip is a flip. The inverse of identity is identity.
        return h

    def left_action_on_R2(self, h, x):
        """
        Applies the flip matrix to a vector x.
        """
        matrices = self.matrix_representation(h) # (2, 2)
        
        if matrices.dtype != x.dtype:
            matrices = matrices.to(x.dtype)

        return torch.tensordot(matrices, x, dims=1)
    

    def matrix_representation(self, h):
        """
        Returns 2x2 transformation matrices for the group elements.
        h can be a batch of elements.
        """
        # h = 0 → +1,  h = 1 → -1
        sx = 1 - 2 * h      # gives 1 or -1
        # sy = torch.ones_like(h)

        # Construct 2x2 matrix
        representation = torch.tensor([
            [1.0, 0.],
            [0., sx]
        ], device=self.identity.device)

        return representation

    
    def get_shared_weight_linear_weights(self, in_features, out_features):
        a = torch.empty(out_features, in_features)
        b = torch.empty(out_features, in_features)
        
        a = nn.Parameter(a)
        nn.init.kaiming_uniform_(a, a=(5 ** 0.5))
        a.data /= math.sqrt(2)
        
        b = nn.Parameter(b)
        nn.init.kaiming_uniform_(b, a=(5 ** 0.5))
        b.data /= math.sqrt(2)
        
        
        return [a, b]
    
    
    def get_channel_attention(self, q, k, v, head_dim, temperature=1.0, attn_drop=None):
        attn_1 = q[:, :, :, :head_dim] @ k[:, :, :, :head_dim].transpose(-2, -1)
        attn_2 = q[:, :, :, head_dim:] @ k[:, :, :, head_dim:].transpose(-2, -1)
        
        attn_1 = attn_1 / temperature
        attn_2 = attn_2 / temperature
        
        attn_1 = attn_1.softmax(dim=-1)
        attn_2 = attn_2.softmax(dim=-1)
        
        if attn_drop is not None:
            attn_1 = attn_drop(attn_1)
            attn_2 = attn_drop(attn_2)
        
        x1 = attn_1 @ v[:, :, :, :head_dim]
        x2 = attn_2 @ v[:, :, :, head_dim:]
        
        x = torch.cat([x1, x2], dim=-1)
        
        return x
        
    
    def roll_group(self, x):
        B, N, C = x.shape
        x = x.reshape(B, N, 2, -1)
        x = torch.roll(x, shifts=1, dims=2)
        return x.reshape(B, N, C)
    
    
    def trans(self, x, g):
        if g == 0:
            return x
        elif g == 1:
            return torch.flip(x, dims=[-1])
    
    
    def roll(self, x, g):
        if g == 0:
            return x
        elif g == 1:
            C, G, H, W = x.shape
            return torch.roll(x, shifts=1, dims=1)
    
    def get_pos_embd(self, num_patches, embed_dim, group_attn_channel_pooling):
        return FlippingSymmetricPosEmbed(num_patches=num_patches, embed_dim=embed_dim, group_attn_channel_pooling=group_attn_channel_pooling)
    

    def get_canonicalization_ref(self, device, dtype):
        # 0 -> Identity, 1 -> Flip
        return torch.tensor([0., 1.], device=device, dtype=dtype)
    
    
    def get_canonicalized_images(self, images, indicator):
        flipped = torch.flip(images, dims=[-1]) 
        indicator = indicator.view(-1, 1, 1, 1)
        canonicalized_images = (1 - indicator) * images + indicator * flipped
        return canonicalized_images, indicator
    
    # def determinant(self, h):
    #     h = torch.as_tensor(h)
    #     # det(Identity) = 1, det(Flip) = -1
    #     vals = torch.tensor([1., -1.], device=self.identity.device)
    #     return vals[h.long()]

    def normalize_group_elements(self, h):
        # 0 -> -1, 1 -> +1
        return 2 * h - 1