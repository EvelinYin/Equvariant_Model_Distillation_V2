import torch
import torch.nn as nn
import math
from .base_group import GroupBase

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
    
    
    # def get_shared_weight_linear_bias(self, out_features):
    #     bias = nn.Parameter(torch.zeros(out_features))
        
    #     return bias
    
    
    # def determinant(self, h):
    #     h = torch.as_tensor(h)
    #     # det(Identity) = 1, det(Flip) = -1
    #     vals = torch.tensor([1., -1.], device=self.identity.device)
    #     return vals[h.long()]

    def normalize_group_elements(self, h):
        # 0 -> -1, 1 -> +1
        return 2 * h - 1