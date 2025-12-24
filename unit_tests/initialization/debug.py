import torch
import torch.nn as nn
import torch.nn.functional as F

class BlockSharedLinear(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        
        # 1. Define Parameters
        # We define 'a' and 'b' with the shape of standard Linear weights: (Out, In)
        # Shape (C, C)
        self.a = nn.Parameter(torch.empty(C, C)) 
        self.b = nn.Parameter(torch.empty(C, C))
        
        # Define Biases
        # Shape (C)
        self.bias_a = nn.Parameter(torch.empty(C))
        self.bias_b = nn.Parameter(torch.empty(C))

    def init_from_l1(self, l1_layer):
        """
        Copies weights from a standard nn.Linear(C, C) layer
        to ensure exact behavior matching.
        """
        with torch.no_grad():
            # Copy l1 weights into 'a'
            self.a.copy_(l1_layer.weight)
            
            # Set 'b' to exact zeros
            self.b.zero_()
            
            # Copy l1 bias into 'bias_a' (controls first half of output)
            if l1_layer.bias is not None:
                self.bias_a.copy_(l1_layer.bias)
                # Initialize bias_b as well (optional: copy l1 or set to zeros)
                self.bias_b.copy_(l1_layer.bias) 
            else:
                self.bias_a.zero_()
                self.bias_b.zero_()

    def forward(self, x):
        # Your exact code snippet
        # dim=-1 concatenates columns (In_features)
        # dim=0 concatenates rows (Out_features)
        
        # Top row: [a, b] -> connects inputs to first half of outputs
        # Bottom row: [b, a] -> connects inputs to second half of outputs
        W = torch.cat([
                torch.cat([self.a, self.b], dim=-1),
                torch.cat([self.b, self.a], dim=-1)
            ], dim=0) # Result shape: (2C, 2C)
            
        stacked_bias = torch.cat([self.bias_a, self.bias_b], dim=0)

        return F.linear(x, W, stacked_bias)



# --- SETUP ---
B, N, C = 2, 5, 4
l1 = nn.Linear(C, C)
custom_layer = BlockSharedLinear(C)

# --- INITIALIZE ---
# Copy weights from l1 to custom_layer
custom_layer.init_from_l1(l1)

# --- DATA ---
# x is the original input
x = torch.randn(B, N, C)

# x2 is the expanded input. 
# CRITICAL: The first C channels must match x exactly.
# The second C channels can be anything (since b is 0, they get multiplied by 0).
x_dummy = torch.randn(B, N, C)
x2 = torch.cat([x, x_dummy], dim=-1) # Shape (B, N, 2C)

# --- RUN ---
out_l1 = l1(x)                # Shape (B, N, C)
out_custom = custom_layer(x2) # Shape (B, N, 2C)

# Slice the custom output to check the first half
out_custom_first_half = out_custom[:, :, :C]

# --- CHECK ---
print(f"L1 Output shape: {out_l1.shape}")
print(f"Custom Output shape: {out_custom.shape}")
print("Values Match:", torch.allclose(out_l1, out_custom_first_half, atol=1e-6))