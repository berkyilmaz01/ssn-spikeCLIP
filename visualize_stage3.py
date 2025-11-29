"""
SpikeCLIP Stage 3 Visualization
Visualize reconstructions from the fine-tuned SNN

Your PSNR improved from 9.80 dB to 12.24 dB - congrats!
"""

import os
import sys

# ==========================================
# CONFIGURATION - UPDATE THESE PATHS
# ==========================================
PROJECT_ROOT = "C:/Users/berky/PycharmProjects/ssn-spikeCLIP"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "spikeclip_snn"))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

# Import your modules
from data.ncaltech101_dataset import NCaltech101Dataset
from models.snn_model import SNNReconstruction

# ==========================================
# SETTINGS (same as your configs)
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_BINS = 5
NUM_STEPS = 50
BETA = 0.95
BATCH_SIZE = 8

# Dataset Paths
EVENT_PATH = "C:/Users/berky/PycharmProjects/ssn-spikeCLIP/datasets/N-Caltech101/Caltech101/Caltech101"
IMAGE_PATH = "C:/Users/berky/PycharmProjects/ssn-spikeCLIP/datasets/101_ObjectCategories/101_ObjectCategories"

# Checkpoint Paths
STAGE1_CHECKPOINT = "C:/Users/berky/PycharmProjects/ssn-spikeCLIP/spikeclip_snn/checkpoints/spikeclip_best.pth"
STAGE3_CHECKPOINT_DIR = "C:/Users/berky/PycharmProjects/ssn-spikeCLIP/spikeclip_snn/checkpoints/stage3"

# Output directory for saved figures
OUTPUT_DIR = "C:/Users/berky/PycharmProjects/ssn-spikeCLIP/spikeclip_snn/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def calculate_psnr(pred, target):
    """Calculate PSNR between prediction and target."""
    # Resize if needed
    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)

    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()


def find_stage3_checkpoint():
    """Find the best Stage 3 checkpoint file."""
    possible_names = [
        "snn_stage3_best.pth",
        "snn_stage3_final.pth",
        "stage3_best.pth",
        "best.pth"
    ]

    for name in possible_names:
        path = os.path.join(STAGE3_CHECKPOINT_DIR, name)
        if os.path.exists(path):
            return path

    # List what's actually there
    if os.path.exists(STAGE3_CHECKPOINT_DIR):
        files = os.listdir(STAGE3_CHECKPOINT_DIR)
        pth_files = [f for f in files if f.endswith('.pth')]
        if pth_files:
            return os.path.join(STAGE3_CHECKPOINT_DIR, pth_files[0])

    return None


def load_models():
    """Load Stage 1 and Stage 3 models for comparison."""
    print("=" * 60)
    print("Loading Models...")
    print("=" * 60)

    # Stage 1 model
    print("\n[1] Loading Stage 1 SNN...")
    snn_stage1 = SNNReconstruction(num_bins=NUM_BINS, beta=BETA).to(DEVICE)
    stage1_ckpt = torch.load(STAGE1_CHECKPOINT, map_location=DEVICE, weights_only=False)
    snn_stage1.load_state_dict(stage1_ckpt['model_state_dict'])
    snn_stage1.eval()
    print(f"    ✓ Stage 1 loaded (Epoch {stage1_ckpt.get('epoch', '?')})")
    print(f"    ✓ Stage 1 PSNR: {stage1_ckpt.get('val_psnr', 'N/A'):.2f} dB")

    # Stage 3 model
    print("\n[2] Loading Stage 3 SNN...")
    stage3_path = find_stage3_checkpoint()
    if stage3_path is None:
        print(f"    ✗ Could not find Stage 3 checkpoint in {STAGE3_CHECKPOINT_DIR}")
        print(
            f"    Available files: {os.listdir(STAGE3_CHECKPOINT_DIR) if os.path.exists(STAGE3_CHECKPOINT_DIR) else 'Directory not found'}")
        return snn_stage1, None, stage1_ckpt, None

    print(f"    Found checkpoint: {stage3_path}")
    snn_stage3 = SNNReconstruction(num_bins=NUM_BINS, beta=BETA).to(DEVICE)
    stage3_ckpt = torch.load(stage3_path, map_location=DEVICE, weights_only=False)
    snn_stage3.load_state_dict(stage3_ckpt['model_state_dict'])
    snn_stage3.eval()
    print(f"    ✓ Stage 3 loaded (Epoch {stage3_ckpt.get('epoch', '?')})")
    print(f"    ✓ Stage 3 PSNR: {stage3_ckpt.get('val_psnr', 'N/A'):.2f} dB")

    improvement = stage3_ckpt.get('val_psnr', 0) - stage1_ckpt.get('val_psnr', 0)
    print(f"\n    🎯 PSNR Improvement: +{improvement:.2f} dB")

    return snn_stage1, snn_stage3, stage1_ckpt, stage3_ckpt


def load_dataset():
    """Load validation dataset using your existing NCaltech101Dataset."""
    print("\n[3] Loading Dataset...")

    # Use your existing dataset class
    dataset = NCaltech101Dataset(
        root_dir=EVENT_PATH,
        num_bins=NUM_BINS,
        image_dir=IMAGE_PATH
    )

    # Use same split as training
    torch.manual_seed(12)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    _, val_dataset = random_split(dataset, [train_size, val_size])

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # Shuffle to get diverse samples
        num_workers=0
    )

    print(f"    ✓ Validation samples: {len(val_dataset)}")

    # Get class names
    class_names = dataset.classes if hasattr(dataset, 'classes') else None

    return val_loader, class_names


def visualize_comparison(snn_stage1, snn_stage3, val_loader, class_names, num_samples=4):
    """
    Visualize Stage 1 vs Stage 3 reconstructions.

    Shows: Event Voxel | Ground Truth | Stage 1 Output | Stage 3 Output
    """
    print("\n" + "=" * 60)
    print("Generating Visualizations...")
    print("=" * 60)

    # Get a batch
    batch = next(iter(val_loader))

    # Handle different return formats
    if len(batch) == 3:
        voxels, images, labels = batch
    elif len(batch) == 2:
        voxels, labels = batch
        images = None
    else:
        voxels = batch[0]
        labels = batch[1] if len(batch) > 1 else None
        images = batch[2] if len(batch) > 2 else None

    voxels = voxels.to(DEVICE)
    if images is not None:
        images = images.to(DEVICE)

    with torch.no_grad():
        # Generate reconstructions
        outputs_stage1 = snn_stage1(voxels, num_steps=NUM_STEPS)
        if snn_stage3 is not None:
            outputs_stage3 = snn_stage3(voxels, num_steps=NUM_STEPS)
        else:
            outputs_stage3 = outputs_stage1  # Fallback

    # Determine number of columns based on available data
    if images is not None and snn_stage3 is not None:
        num_cols = 4
        col_titles = ['Event Voxel', 'Ground Truth', 'Stage 1', 'Stage 3']
    elif images is not None:
        num_cols = 3
        col_titles = ['Event Voxel', 'Ground Truth', 'Stage 1']
    elif snn_stage3 is not None:
        num_cols = 3
        col_titles = ['Event Voxel', 'Stage 1', 'Stage 3']
    else:
        num_cols = 2
        col_titles = ['Event Voxel', 'Stage 1']

    # Create figure
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(4 * num_cols, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(min(num_samples, len(voxels))):
        col_idx = 0

        # Get class name
        if class_names is not None and labels is not None:
            class_name = class_names[labels[i].item()]
        else:
            class_name = f"Sample {i + 1}"

        # Event voxel (sum across time bins for visualization)
        voxel_vis = voxels[i].cpu().sum(dim=0).numpy()
        axes[i, col_idx].imshow(voxel_vis, cmap='viridis')
        axes[i, col_idx].set_title(f'{col_titles[col_idx]}\n{class_name}')
        axes[i, col_idx].axis('off')
        col_idx += 1

        # Ground truth (if available)
        if images is not None:
            gt_img = images[i, 0].cpu().numpy()
            axes[i, col_idx].imshow(gt_img, cmap='gray', vmin=0, vmax=1)
            axes[i, col_idx].set_title(col_titles[col_idx])
            axes[i, col_idx].axis('off')
            col_idx += 1

        # Stage 1 output
        stage1_img = outputs_stage1[i, 0].cpu().numpy()
        stage1_img = np.clip(stage1_img, 0, 1)

        if images is not None:
            psnr_stage1 = calculate_psnr(outputs_stage1[i:i + 1], images[i:i + 1])
            title_s1 = f'{col_titles[col_idx]}\nPSNR: {psnr_stage1:.2f} dB'
        else:
            title_s1 = col_titles[col_idx]

        axes[i, col_idx].imshow(stage1_img, cmap='gray', vmin=0, vmax=1)
        axes[i, col_idx].set_title(title_s1)
        axes[i, col_idx].axis('off')
        col_idx += 1

        # Stage 3 output (if available)
        if snn_stage3 is not None and col_idx < num_cols:
            stage3_img = outputs_stage3[i, 0].cpu().numpy()
            stage3_img = np.clip(stage3_img, 0, 1)

            if images is not None:
                psnr_stage3 = calculate_psnr(outputs_stage3[i:i + 1], images[i:i + 1])
                title_s3 = f'{col_titles[col_idx]}\nPSNR: {psnr_stage3:.2f} dB'
                print(f"  Sample {i + 1}: {class_name}")
                print(f"    Stage 1 PSNR: {psnr_stage1:.2f} dB")
                print(f"    Stage 3 PSNR: {psnr_stage3:.2f} dB")
                print(f"    Improvement: +{psnr_stage3 - psnr_stage1:.2f} dB")
            else:
                title_s3 = col_titles[col_idx]

            axes[i, col_idx].imshow(stage3_img, cmap='gray', vmin=0, vmax=1)
            axes[i, col_idx].set_title(title_s3)
            axes[i, col_idx].axis('off')

    plt.tight_layout()

    # Save figure
    save_path = os.path.join(OUTPUT_DIR, "stage3_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved to: {save_path}")

    plt.show()

    return fig


def visualize_gallery(snn_stage3, val_loader, class_names, num_samples=8):
    """Show Stage 3 results in a grid."""
    if snn_stage3 is None:
        print("No Stage 3 model available for gallery.")
        return

    print("\n" + "=" * 60)
    print("Stage 3 Gallery...")
    print("=" * 60)

    all_outputs = []
    all_images = []
    all_labels = []
    all_psnrs = []

    # Collect samples
    for batch in val_loader:
        if len(batch) == 3:
            voxels, images, labels = batch
        elif len(batch) == 2:
            voxels, labels = batch
            images = None
        else:
            break

        voxels = voxels.to(DEVICE)
        if images is not None:
            images = images.to(DEVICE)

        with torch.no_grad():
            outputs = snn_stage3(voxels, num_steps=NUM_STEPS)

        all_outputs.append(outputs.cpu())
        if images is not None:
            all_images.append(images.cpu())
        all_labels.extend(labels.tolist())

        if images is not None:
            for j in range(len(outputs)):
                psnr = calculate_psnr(outputs[j:j + 1], images[j:j + 1])
                all_psnrs.append(psnr)

        if len(all_labels) >= num_samples:
            break

    # Concatenate
    all_outputs = torch.cat(all_outputs, dim=0)[:num_samples]
    if all_images:
        all_images = torch.cat(all_images, dim=0)[:num_samples]
    all_labels = all_labels[:num_samples]
    all_psnrs = all_psnrs[:num_samples] if all_psnrs else None

    # Create grid
    cols = 4
    rows = (num_samples + cols - 1) // cols

    if all_images:
        fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 6, rows * 3))
    else:
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    for i in range(num_samples):
        # Get class name
        if class_names is not None:
            class_name = class_names[all_labels[i]]
        else:
            class_name = f"Class {all_labels[i]}"

        if all_images:
            # Ground truth
            ax_gt = axes[i * 2]
            gt_img = all_images[i, 0].numpy()
            ax_gt.imshow(gt_img, cmap='gray', vmin=0, vmax=1)
            ax_gt.set_title(f'GT: {class_name}', fontsize=10)
            ax_gt.axis('off')

            # Reconstruction
            ax_recon = axes[i * 2 + 1]
            recon_img = np.clip(all_outputs[i, 0].numpy(), 0, 1)
            ax_recon.imshow(recon_img, cmap='gray', vmin=0, vmax=1)
            psnr_str = f'{all_psnrs[i]:.1f} dB' if all_psnrs else ''
            ax_recon.set_title(f'Recon: {psnr_str}', fontsize=10)
            ax_recon.axis('off')
        else:
            ax = axes[i]
            recon_img = np.clip(all_outputs[i, 0].numpy(), 0, 1)
            ax.imshow(recon_img, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'{class_name}', fontsize=10)
            ax.axis('off')

    # Hide unused axes
    total_used = num_samples * 2 if all_images else num_samples
    for i in range(total_used, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Stage 3 Reconstructions', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save
    save_path = os.path.join(OUTPUT_DIR, "stage3_gallery.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to: {save_path}")

    # Print stats
    if all_psnrs:
        avg_psnr = np.mean(all_psnrs)
        print(f"\nAverage PSNR: {avg_psnr:.2f} dB")
        print(f"Min PSNR: {min(all_psnrs):.2f} dB")
        print(f"Max PSNR: {max(all_psnrs):.2f} dB")

    plt.show()


def main():
    print("=" * 60)
    print("SpikeCLIP Stage 3 Visualization")
    print(f"PSNR: 9.80 dB → 12.24 dB (+2.44 dB improvement!)")
    print("=" * 60)

    # Load models
    snn_stage1, snn_stage3, stage1_ckpt, stage3_ckpt = load_models()

    # Load dataset
    val_loader, class_names = load_dataset()

    # Visualize comparison (Stage 1 vs Stage 3)
    visualize_comparison(snn_stage1, snn_stage3, val_loader, class_names, num_samples=4)

    # Gallery of Stage 3 results
    if snn_stage3 is not None:
        visualize_gallery(snn_stage3, val_loader, class_names, num_samples=8)

    print("\n" + "=" * 60)
    print("Visualization Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

