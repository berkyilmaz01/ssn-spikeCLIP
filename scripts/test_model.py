"""
Test script to demonstrate SNN model inference and outputs

This script loads a trained SNN model, runs inference on sample data,
and displays the results with metrics (PSNR, latency, etc.).

Usage:
    python scripts/test_model.py
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "spikeclip_snn"))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from models.snn_model import SNNReconstruction
from data.ncaltech101_dataset import NCaltech101Dataset

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_BINS = 5
NUM_STEPS = 50
BETA = 0.95
NUM_SAMPLES = 8

# Paths from environment
EVENT_PATH = os.getenv("EVENT_PATH")
IMAGE_PATH = os.getenv("IMAGE_PATH")
STAGE1_CHECKPOINT = os.getenv("STAGE1_CHECKPOINT")
STAGE3_CHECKPOINT = os.getenv("STAGE3_CHECKPOINT_DIR")

if not all([EVENT_PATH, IMAGE_PATH, STAGE1_CHECKPOINT]):
    print("Error: Missing required environment variables in .env file")
    print("Required: EVENT_PATH, IMAGE_PATH, STAGE1_CHECKPOINT")
    sys.exit(1)


def calculate_psnr(pred, target):
    """Calculate PSNR between prediction and target."""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return -10 * torch.log10(mse).item()


def load_model(checkpoint_path, stage_name="Stage 1"):
    """Load SNN model from checkpoint."""
    print(f"\n[{stage_name}] Loading model...")
    model = SNNReconstruction(num_bins=NUM_BINS, beta=BETA).to(DEVICE)
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    epoch = checkpoint.get('epoch', '?')
    val_psnr = checkpoint.get('val_psnr', 'N/A')
    
    print(f"  [OK] Loaded checkpoint from epoch {epoch}")
    if isinstance(val_psnr, (int, float)):
        print(f"  [OK] Validation PSNR: {val_psnr:.2f} dB")
    
    return model, checkpoint


def load_test_data():
    """Load test dataset."""
    print("\n[Loading Dataset] Loading test data...")
    
    dataset = NCaltech101Dataset(
        root_dir=EVENT_PATH,
        num_bins=NUM_BINS,
        image_dir=IMAGE_PATH
    )
    
    # Use validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=NUM_SAMPLES,
        shuffle=True,
        num_workers=0
    )
    
    print(f"  [OK] Loaded {len(val_dataset)} validation samples")
    
    return val_loader, dataset.classes


def run_inference(model, voxels, measure_latency=True):
    """Run inference and measure latency."""
    model.eval()
    
    if measure_latency:
        # Warmup
        with torch.no_grad():
            _ = model(voxels[:1], num_steps=NUM_STEPS)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Measure latency
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model(voxels, num_steps=NUM_STEPS)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        latency_per_sample = latency_ms / len(voxels)
        
        return outputs, latency_ms, latency_per_sample
    else:
        with torch.no_grad():
            outputs = model(voxels, num_steps=NUM_STEPS)
        return outputs, None, None


def visualize_results(voxels, images, outputs, labels, class_names, stage_name, psnrs, save_path):
    """Visualize input, ground truth, and output."""
    fig, axes = plt.subplots(3, NUM_SAMPLES, figsize=(NUM_SAMPLES * 2.5, 7.5))
    
    for i in range(NUM_SAMPLES):
        # Row 0: Event voxel (sum across time bins)
        voxel_vis = voxels[i].cpu().sum(dim=0).numpy()
        axes[0, i].imshow(voxel_vis, cmap='hot')
        axes[0, i].set_title(f"Events\n{class_names[labels[i]]}", fontsize=8)
        axes[0, i].axis('off')
        
        # Row 1: Ground truth
        axes[1, i].imshow(images[i, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title("Ground Truth", fontsize=8)
        axes[1, i].axis('off')
        
        # Row 2: SNN output
        output_img = outputs[i, 0].cpu().numpy()
        axes[2, i].imshow(output_img, cmap='gray', vmin=0, vmax=1)
        psnr_str = f"{psnrs[i]:.1f} dB" if psnrs else ""
        axes[2, i].set_title(f"SNN Output\n{psnr_str}", fontsize=8)
        axes[2, i].axis('off')
    
    # Add row labels
    fig.text(0.02, 0.83, 'INPUT\n(Events)', ha='left', va='center', fontsize=10, fontweight='bold')
    fig.text(0.02, 0.50, 'TARGET\n(GT Image)', ha='left', va='center', fontsize=10, fontweight='bold')
    fig.text(0.02, 0.17, 'OUTPUT\n(SNN)', ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.suptitle(f"{stage_name} Test Results", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(left=0.08)
    
    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  [OK] Saved visualization to: {save_path}")
    
    return fig


def main():
    """Main test function."""
    print("=" * 80)
    print("SpikeCLIP Model Test")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Load model
    model, checkpoint = load_model(STAGE1_CHECKPOINT, "Stage 1")
    
    # Load test data
    val_loader, class_names = load_test_data()
    
    # Get a batch
    print("\n[Running Inference] Processing test batch...")
    voxels, images, labels = next(iter(val_loader))
    voxels = voxels.to(DEVICE)
    images = images.to(DEVICE)
    
    # Run inference
    outputs, latency_ms, latency_per_sample = run_inference(model, voxels, measure_latency=True)
    
    # Calculate metrics
    print("\n[Metrics] Calculating performance metrics...")
    psnrs = []
    for i in range(len(outputs)):
        psnr = calculate_psnr(outputs[i:i+1], images[i:i+1])
        psnrs.append(psnr)
    
    avg_psnr = np.mean(psnrs)
    min_psnr = np.min(psnrs)
    max_psnr = np.max(psnrs)
    
    print(f"  PSNR: {avg_psnr:.2f} dB (min: {min_psnr:.2f}, max: {max_psnr:.2f})")
    if latency_ms:
        print(f"  Latency: {latency_ms:.2f} ms total ({latency_per_sample:.2f} ms per sample)")
        print(f"  Throughput: {1000/latency_per_sample:.2f} samples/second")
    
    # Print per-sample results
    print("\n[Per-Sample Results]")
    print("-" * 80)
    for i in range(len(outputs)):
        class_name = class_names[labels[i]]
        print(f"  Sample {i+1} ({class_name:20s}): PSNR = {psnrs[i]:.2f} dB")
    
    # Visualize
    print("\n[Visualization] Creating output visualization...")
    output_dir = PROJECT_ROOT / "spikeclip_snn" / "checkpoints"
    save_path = output_dir / "test_output.png"
    visualize_results(voxels, images, outputs, labels, class_names, "Stage 1", psnrs, save_path)
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Model: Stage 1 SNN (Epoch {checkpoint.get('epoch', '?')})")
    print(f"Test Samples: {len(outputs)}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    if latency_ms:
        print(f"Average Latency: {latency_per_sample:.2f} ms per sample")
        print(f"Throughput: {1000/latency_per_sample:.2f} samples/second")
    print(f"Visualization saved to: {save_path}")
    print("=" * 80)
    
    # Show plot
    plt.show()


if __name__ == "__main__":
    main()

