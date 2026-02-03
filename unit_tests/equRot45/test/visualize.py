"""
Visualization script for Rotation45SymmetricPosEmbed.

This script visualizes the positional embeddings and demonstrates
that they are equivariant under 45-degree rotations.
"""
import sys
import os
import argparse

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import rotate as scipy_rotate

from src.equ_lib.layers.equ_pos_embedding import Rotation45SymmetricPosEmbed, Rotation90SymmetricPosEmbed



def rotate_tensor(tensor, angle_deg):
    """Rotate a 2D tensor by the given angle using bilinear interpolation."""
    angle_rad = angle_deg * math.pi / 180
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Add batch and channel dims if needed: (H, W) -> (1, 1, H, W)
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)

    B, C, H, W = tensor.shape

    theta = torch.tensor([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0]
    ], device=tensor.device, dtype=tensor.dtype).unsqueeze(0)

    grid = F.affine_grid(theta, (B, C, H, W), align_corners=True)
    rotated = F.grid_sample(tensor, grid, mode='bilinear', padding_mode='border', align_corners=True)

    return rotated.squeeze()


# =============================================================================
# Visualization for Rotation45SymmetricPosEmbed (averaging approach)
# =============================================================================

def visualize_45_rotation_symmetry(model, save_dir='pos_embed_vis_45'):
    """
    Visualize the rotation-symmetric positional embedding for Rotation45SymmetricPosEmbed.

    Creates figures showing:
    1. The original symmetric embedding
    2. The embedding rotated by each multiple of 45 degrees
    3. Difference maps to confirm symmetry
    """
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        # Get the symmetric embedding
        symmetric_embed = model._create_rotation_symmetric_embedding()  # (1, 1, H, W)
        embed_2d = symmetric_embed.squeeze().cpu().numpy()  # (H, W)

    H, W = embed_2d.shape
    angles = [0, 45, 90, 135, 180, 225, 270, 315]

    # Figure 1: Show rotated embeddings vs original
    fig, axes = plt.subplots(3, 8, figsize=(20, 8))

    vmin, vmax = embed_2d.min(), embed_2d.max()

    for idx, angle in enumerate(angles):
        # Rotate the symmetric embedding
        embed_tensor = torch.tensor(embed_2d).unsqueeze(0).unsqueeze(0)
        rotated = rotate_tensor(embed_tensor, angle).numpy()

        # Row 1: Rotated embedding
        im1 = axes[0, idx].imshow(rotated, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, idx].set_title(f'Rotated {angle}deg')
        axes[0, idx].axis('off')

        # Row 2: Original embedding
        im2 = axes[1, idx].imshow(embed_2d, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1, idx].set_title('Original')
        axes[1, idx].axis('off')

        # Row 3: Difference
        diff = rotated - embed_2d
        max_diff = max(abs(diff.min()), abs(diff.max()))
        if max_diff < 1e-6:
            max_diff = 1e-6
        im3 = axes[2, idx].imshow(diff, cmap='RdBu', vmin=-max_diff, vmax=max_diff)
        axes[2, idx].set_title(f'Diff (max={np.abs(diff).max():.4f})')
        axes[2, idx].axis('off')

    axes[0, 0].set_ylabel('Rotated\nEmbedding', fontsize=12)
    axes[1, 0].set_ylabel('Original\nEmbedding', fontsize=12)
    axes[2, 0].set_ylabel('Difference', fontsize=12)

    fig.colorbar(im1, ax=axes[0, :], shrink=0.8, label='Embedding Value')
    fig.colorbar(im3, ax=axes[2, :], shrink=0.8, label='Difference')

    plt.suptitle('Rotation45SymmetricPosEmbed: Verifying 45-degree Rotation Symmetry\n'
                 '(If symmetric, rotated embeddings should match original, differences should be ~0)',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/01_rotation_symmetry.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Figure 2: Base vs symmetric embedding
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    with torch.no_grad():
        base_embed = model.pos_embed_base.squeeze().cpu().numpy()  # (H, W)

    # Base embedding
    im0 = axes[0].imshow(base_embed, cmap='viridis')
    axes[0].set_title('Learned Base Embedding\n(Before Symmetrization)')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # Symmetric embedding
    im1 = axes[1].imshow(embed_2d, cmap='viridis')
    axes[1].set_title('Symmetric Embedding\n(After Averaging 8 Rotations)')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    # Difference
    diff = embed_2d - base_embed
    max_diff = max(abs(diff.min()), abs(diff.max()))
    im2 = axes[2].imshow(diff, cmap='RdBu', vmin=-max_diff, vmax=max_diff)
    axes[2].set_title('Difference\n(Symmetric - Base)')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    plt.suptitle('Base vs Symmetrized Positional Embedding', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/02_base_vs_symmetric.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Figure 3: All 8 rotations of base and their average
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    with torch.no_grad():
        base_embed_tensor = model.pos_embed_base  # (1, 1, H, W)

        all_rotations = []
        vmin_rot, vmax_rot = float('inf'), float('-inf')

        for angle_deg in angles:
            angle_rad = angle_deg * math.pi / 180
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)

            theta = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0]
            ], dtype=base_embed_tensor.dtype).unsqueeze(0)

            grid = F.affine_grid(theta, base_embed_tensor.shape, align_corners=True)
            rotated = F.grid_sample(base_embed_tensor, grid, mode='bilinear',
                                   padding_mode='border', align_corners=True)
            rotated_np = rotated.squeeze().cpu().numpy()
            all_rotations.append(rotated_np)
            vmin_rot = min(vmin_rot, rotated_np.min())
            vmax_rot = max(vmax_rot, rotated_np.max())

        # Plot each rotation
        for idx, (angle, rotated) in enumerate(zip(angles, all_rotations)):
            im = axes[idx].imshow(rotated, cmap='viridis', vmin=vmin_rot, vmax=vmax_rot)
            axes[idx].set_title(f'Base rotated {angle}deg')
            axes[idx].axis('off')

        # Plot the average (symmetric embedding)
        avg_embed = np.mean(all_rotations, axis=0)
        im = axes[8].imshow(avg_embed, cmap='viridis', vmin=vmin_rot, vmax=vmax_rot)
        axes[8].set_title('Average\n(Symmetric Embedding)', fontweight='bold')
        axes[8].axis('off')

        # Hide the last subplot
        axes[9].axis('off')

    fig.colorbar(im, ax=axes, shrink=0.6, label='Embedding Value')
    plt.suptitle('All 8 Rotations of Base Embedding and Their Average', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/03_rotations_and_average.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print max differences for each rotation
    print("\nMax absolute differences after rotating symmetric embedding:")
    for angle in angles:
        embed_tensor = torch.tensor(embed_2d).unsqueeze(0).unsqueeze(0)
        rotated = rotate_tensor(embed_tensor, angle).numpy()
        diff = np.abs(rotated - embed_2d).max()
        print(f"  {angle:3d} deg: {diff:.6f}")

    print(f"\nFigures saved to {save_dir}/")


# =============================================================================
# Visualization for Rotation90SymmetricPosEmbed (channel permutation approach)
# =============================================================================

def visualize_90_rotation_symmetry(model, save_dir='pos_embed_vis_90'):
    """
    Visualize the rotation-symmetric positional embedding for Rotation90SymmetricPosEmbed.
    """
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        grid = model._create_rotation_grid()  # (1, H, W, 4C)
        grid = grid.squeeze(0).cpu().numpy()  # (H, W, 4C)

    H, W, total_C = grid.shape
    C = total_C // 4

    # Figure 1: All 4 channel groups
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for k in range(4):
        channel_group = grid[:, :, k * C:(k + 1) * C]
        channel_mean = np.mean(channel_group, axis=-1)

        im = axes[k].imshow(channel_mean, cmap='RdBu_r')
        axes[k].set_title(f'Channel Group {k}\n(mean over {C} dims)')
        axes[k].axis('off')
        plt.colorbar(im, ax=axes[k], shrink=0.8)

    plt.suptitle('Rotation90SymmetricPosEmbed: 4 Channel Groups\n'
                 '(Channel k should look like Channel 0 rotated by k*90 deg)',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/01_channel_groups.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Figure 2: Verify channel rotation relationship
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    channel_0 = np.mean(grid[:, :, 0:C], axis=-1)
    vmin, vmax = channel_0.min(), channel_0.max()

    for k in range(4):
        # Row 1: Actual channel k
        channel_k = np.mean(grid[:, :, k * C:(k + 1) * C], axis=-1)
        im1 = axes[0, k].imshow(channel_k, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[0, k].set_title(f'Channel {k}')
        axes[0, k].axis('off')

        # Row 2: Channel 0 rotated by k * 90 degrees
        rotated_ch0 = np.rot90(channel_0, k=-k)  # Negative for CCW rotation
        im2 = axes[1, k].imshow(rotated_ch0, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[1, k].set_title(f'Ch0 rotated {k*90}deg')
        axes[1, k].axis('off')

    axes[0, 0].set_ylabel('Actual\nChannels', fontsize=11)
    axes[1, 0].set_ylabel('Channel 0\nRotated', fontsize=11)

    plt.suptitle('Rotation90SymmetricPosEmbed: Channel Rotation Relationship\n'
                 '(Top row should match bottom row)',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/02_channel_rotation.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print differences
    print("\nMax absolute differences (channel k vs channel 0 rotated by k*90 deg):")
    for k in range(4):
        channel_k = np.mean(grid[:, :, k * C:(k + 1) * C], axis=-1)
        rotated_ch0 = np.rot90(channel_0, k=-k)
        diff = np.abs(channel_k - rotated_ch0).max()
        print(f"  Channel {k}: {diff:.6f}")

    print(f"\nFigures saved to {save_dir}/")


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Visualize rotation-symmetric positional embeddings'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        choices=['45', '90', 'both'],
        default='45',
        help='Model type: 45, 90, or both (default: 45)'
    )
    parser.add_argument(
        '--grid-size', '-g',
        type=int,
        default=14,
        help='Grid size (default: 14, giving 196 patches)'
    )
    parser.add_argument(
        '--embed-dim', '-e',
        type=int,
        default=64,
        help='Embedding dimension (default: 64)'
    )
    parser.add_argument(
        '--save-dir', '-s',
        type=str,
        default='visualizations',
        help='Directory to save the figures (default: visualizations)'
    )

    args = parser.parse_args()

    num_patches = args.grid_size * args.grid_size

    if args.model in ['45', 'both']:
        print("=" * 60)
        print("Visualizing Rotation45SymmetricPosEmbed")
        print("=" * 60)
        model_45 = Rotation45SymmetricPosEmbed(num_patches, args.embed_dim)
        model_45.eval()
        visualize_45_rotation_symmetry(model_45, save_dir=args.save_dir)

    if args.model in ['90', 'both']:
        print("\n" + "=" * 60)
        print("Visualizing Rotation90SymmetricPosEmbed")
        print("=" * 60)
        # embed_dim must be divisible by 4 for Rotation90
        embed_dim_90 = (args.embed_dim // 4) * 4
        if embed_dim_90 == 0:
            embed_dim_90 = 4
        model_90 = Rotation90SymmetricPosEmbed(num_patches, embed_dim_90)
        model_90.eval()
        visualize_90_rotation_symmetry(model_90, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
