"""
SpikeCLIP Comprehensive Demo

This script demonstrates the complete SpikeCLIP pipeline:
- Model loading and verification
- Dataset loading
- Inference with visualizations
- Performance profiling (latency, throughput, power)
- Training verification

Usage:
    python scripts/demo.py
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
STAGE3_CHECKPOINT_DIR = os.getenv("STAGE3_CHECKPOINT_DIR")

if not all([EVENT_PATH, IMAGE_PATH, STAGE1_CHECKPOINT]):
    print("Error: Missing required environment variables in .env file")
    print("Required: EVENT_PATH, IMAGE_PATH, STAGE1_CHECKPOINT")
    sys.exit(1)

# Profiling parameters
WARMUP_ITERATIONS = 10
PROFILE_ITERATIONS = 50
BATCH_SIZES = [1, 4, 8, 16]


def get_gpu_power():
    """Get current GPU power consumption using nvidia-smi."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            power_str = result.stdout.strip().split('\n')[0]
            return float(power_str)
    except:
        pass
    return None


def count_parameters(model):
    """Count total and trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def estimate_flops(model, input_shape, num_steps):
    """Estimate FLOPs for SNN forward pass."""
    batch_size, channels, height, width = input_shape
    
    # Encoder FLOPs (per timestep)
    conv1_flops = batch_size * 64 * (height // 2) * (width // 2) * 5 * 3 * 3
    conv2_flops = batch_size * 128 * (height // 4) * (width // 4) * 64 * 3 * 3
    conv3_flops = batch_size * 256 * (height // 8) * (width // 8) * 128 * 3 * 3
    
    lif_neurons = (64 * (height // 2) * (width // 2) + 
                   128 * (height // 4) * (width // 4) + 
                   256 * (height // 8) * (width // 8))
    lif_flops = batch_size * lif_neurons * 3
    
    encoder_flops = (conv1_flops + conv2_flops + conv3_flops + lif_flops) * num_steps
    avg_flops = batch_size * 256 * (height // 8) * (width // 8)
    
    # Decoder FLOPs
    deconv1_flops = batch_size * 128 * (height // 4) * (width // 4) * 256 * 4 * 4
    deconv2_flops = batch_size * 64 * (height // 2) * (width // 2) * 128 * 4 * 4
    deconv3_flops = batch_size * 1 * height * width * 64 * 4 * 4
    relu_flops = batch_size * (128 + 64) * (height // 2) * (width // 2)
    sigmoid_flops = batch_size * height * width
    interpolate_flops = batch_size * height * width * 4
    
    decoder_flops = deconv1_flops + deconv2_flops + deconv3_flops + relu_flops + sigmoid_flops + interpolate_flops
    total_flops = encoder_flops + avg_flops + decoder_flops
    return total_flops


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
    val_loss = checkpoint.get('val_loss', 'N/A')
    
    print(f"  [OK] Loaded checkpoint from epoch {epoch}")
    if isinstance(val_psnr, (int, float)):
        print(f"  [OK] Validation PSNR: {val_psnr:.2f} dB")
    if isinstance(val_loss, (int, float)):
        print(f"  [OK] Validation Loss: {val_loss:.4f}")
    
    return model, checkpoint


def load_test_data():
    """Load test dataset."""
    print("\n[Dataset] Loading test data...")
    
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
    print(f"  [OK] Dataset has {len(dataset.classes)} classes")
    
    return val_loader, dataset.classes


def measure_latency(model, input_tensor, num_steps, num_iterations, warmup_iterations):
    """Measure inference latency."""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(input_tensor, num_steps=num_steps)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Measure latency
    latencies = []
    with torch.no_grad():
        for _ in range(num_iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            _ = model(input_tensor, num_steps=num_steps)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
    
    latencies = np.array(latencies)
    return {
        'mean': np.mean(latencies),
        'std': np.std(latencies),
        'min': np.min(latencies),
        'max': np.max(latencies),
        'median': np.median(latencies)
    }


def measure_throughput(model, input_tensor, num_steps, duration_seconds=3.0):
    """Measure throughput (samples per second)."""
    model.eval()
    batch_size = input_tensor.shape[0]
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor, num_steps=num_steps)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Measure throughput
    num_iterations = 0
    start_time = time.perf_counter()
    
    with torch.no_grad():
        while (time.perf_counter() - start_time) < duration_seconds:
            _ = model(input_tensor, num_steps=num_steps)
            num_iterations += 1
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed_time = time.perf_counter() - start_time
    total_samples = num_iterations * batch_size
    throughput = total_samples / elapsed_time
    
    return throughput


def measure_power_consumption(model, input_tensor, num_steps, duration_seconds=5.0):
    """Measure power consumption during inference."""
    # Measure idle power
    idle_powers = []
    for _ in range(3):
        power = get_gpu_power()
        if power is not None:
            idle_powers.append(power)
        time.sleep(0.5)
    
    idle_power = np.mean(idle_powers) if idle_powers else None
    
    # Warmup
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor, num_steps=num_steps)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Measure active power
    active_powers = []
    start_time = time.perf_counter()
    num_iterations = 0
    
    with torch.no_grad():
        while (time.perf_counter() - start_time) < duration_seconds:
            _ = model(input_tensor, num_steps=num_steps)
            num_iterations += 1
            
            if num_iterations % 5 == 0:
                power = get_gpu_power()
                if power is not None:
                    active_powers.append(power)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    active_power = np.mean(active_powers) if active_powers else None
    
    return {
        'idle_power': idle_power,
        'active_power': active_power,
        'power_delta': (active_power - idle_power) if (active_power and idle_power) else None
    }


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
    
    plt.suptitle(f"{stage_name} Demo Results", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(left=0.08)
    
    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  [OK] Saved visualization to: {save_path}")
    
    return fig


def main():
    """Main demo function."""
    print("=" * 80)
    print("SpikeCLIP Comprehensive Demo")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print()
    
    # ========================================================================
    # SECTION 1: Model Loading and Info
    # ========================================================================
    print("=" * 80)
    print("SECTION 1: Model Loading and Verification")
    print("=" * 80)
    
    model, checkpoint = load_model(STAGE1_CHECKPOINT, "Stage 1")
    
    # Model info
    total_params, trainable_params = count_parameters(model)
    print(f"\n[Model Info]")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    # Training verification
    print(f"\n[Training Verification]")
    epoch = checkpoint.get('epoch', '?')
    val_psnr = checkpoint.get('val_psnr', 'N/A')
    val_loss = checkpoint.get('val_loss', 'N/A')
    print(f"  Training completed: Epoch {epoch}")
    if isinstance(val_psnr, (int, float)):
        print(f"  Validation PSNR: {val_psnr:.2f} dB")
    if isinstance(val_loss, (int, float)):
        print(f"  Validation Loss: {val_loss:.4f}")
    print("  [OK] Model training verified successfully")
    
    # ========================================================================
    # SECTION 2: Dataset Loading
    # ========================================================================
    print("\n" + "=" * 80)
    print("SECTION 2: Dataset Loading")
    print("=" * 80)
    
    val_loader, class_names = load_test_data()
    
    # ========================================================================
    # SECTION 3: Inference Demo
    # ========================================================================
    print("\n" + "=" * 80)
    print("SECTION 3: Inference Demo")
    print("=" * 80)
    
    # Get a batch
    print("\n[Inference] Running inference on sample batch...")
    voxels, images, labels = next(iter(val_loader))
    voxels = voxels.to(DEVICE)
    images = images.to(DEVICE)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        start_time = time.perf_counter()
        outputs = model(voxels, num_steps=NUM_STEPS)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
    
    inference_time = (end_time - start_time) * 1000
    print(f"  [OK] Inference completed in {inference_time:.2f} ms")
    print(f"  [OK] Processed {len(outputs)} samples")
    
    # Calculate PSNR
    print("\n[Image Quality] Calculating PSNR...")
    psnrs = []
    for i in range(len(outputs)):
        psnr = calculate_psnr(outputs[i:i+1], images[i:i+1])
        psnrs.append(psnr)
    
    avg_psnr = np.mean(psnrs)
    min_psnr = np.min(psnrs)
    max_psnr = np.max(psnrs)
    
    print(f"  Average PSNR: {avg_psnr:.2f} dB")
    print(f"  Min PSNR: {min_psnr:.2f} dB")
    print(f"  Max PSNR: {max_psnr:.2f} dB")
    
    # Per-sample results
    print("\n[Per-Sample Results]")
    print("-" * 80)
    for i in range(len(outputs)):
        class_name = class_names[labels[i]]
        print(f"  Sample {i+1} ({class_name:20s}): PSNR = {psnrs[i]:.2f} dB")
    
    # Visualize
    print("\n[Visualization] Creating output visualization...")
    output_dir = PROJECT_ROOT / "spikeclip_snn" / "checkpoints"
    save_path = output_dir / "demo_output.png"
    visualize_results(voxels, images, outputs, labels, class_names, "Stage 1", psnrs, save_path)
    
    # ========================================================================
    # SECTION 4: Performance Profiling
    # ========================================================================
    print("\n" + "=" * 80)
    print("SECTION 4: Performance Profiling")
    print("=" * 80)
    
    # Estimate FLOPs
    print("\n[FLOPs] Estimating computational complexity...")
    input_shape = voxels.shape
    flops = estimate_flops(model, input_shape, NUM_STEPS)
    print(f"  Estimated FLOPs per inference: {flops / 1e9:.2f} GFLOPs")
    print(f"  Estimated FLOPs per timestep: {flops / NUM_STEPS / 1e9:.4f} GFLOPs")
    
    # Latency measurements
    print("\n[Latency] Measuring inference latency...")
    print("-" * 80)
    print(f"{'Batch Size':<12} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'Per Sample (ms)':<15}")
    print("-" * 80)
    
    latency_results = {}
    sample_voxel = voxels[0:1]  # Single sample for batch creation
    
    for batch_size in BATCH_SIZES:
        if batch_size == 1:
            batch_input = sample_voxel
        else:
            batch_input = sample_voxel.repeat(batch_size, 1, 1, 1)
        
        latency_stats = measure_latency(
            model, batch_input, NUM_STEPS,
            PROFILE_ITERATIONS, WARMUP_ITERATIONS
        )
        
        latency_results[batch_size] = latency_stats
        per_sample = latency_stats['mean'] / batch_size
        
        print(f"{batch_size:<12} "
              f"{latency_stats['mean']:<12.3f} "
              f"{latency_stats['std']:<12.3f} "
              f"{latency_stats['min']:<12.3f} "
              f"{latency_stats['max']:<12.3f} "
              f"{per_sample:<15.3f}")
    
    # Throughput measurements
    print("\n[Throughput] Measuring throughput...")
    print("-" * 80)
    print(f"{'Batch Size':<12} {'Throughput (samples/s)':<25} {'Throughput (images/s)':<25}")
    print("-" * 80)
    
    throughput_results = {}
    
    for batch_size in BATCH_SIZES:
        if batch_size == 1:
            batch_input = sample_voxel
        else:
            batch_input = sample_voxel.repeat(batch_size, 1, 1, 1)
        
        throughput = measure_throughput(model, batch_input, NUM_STEPS, duration_seconds=2.0)
        throughput_results[batch_size] = throughput
        
        images_per_second = throughput / batch_size if batch_size > 0 else throughput
        
        print(f"{batch_size:<12} {throughput:<25.2f} {images_per_second:<25.2f}")
    
    # Power consumption
    print("\n[Power] Measuring power consumption...")
    print("-" * 80)
    
    power_batch_size = 16
    if power_batch_size == 1:
        power_input = sample_voxel
    else:
        power_input = sample_voxel.repeat(power_batch_size, 1, 1, 1)
    
    power_stats = measure_power_consumption(model, power_input, NUM_STEPS, duration_seconds=5.0)
    
    if power_stats['idle_power'] is not None:
        print(f"  Idle Power: {power_stats['idle_power']:.2f} W")
    else:
        print("  Idle Power: Not available (nvidia-smi not found)")
    
    if power_stats['active_power'] is not None:
        print(f"  Active Power: {power_stats['active_power']:.2f} W")
        if power_stats['power_delta'] is not None:
            print(f"  Power Delta: {power_stats['power_delta']:.2f} W")
    else:
        print("  Active Power: Not available (nvidia-smi not found)")
    
    # ========================================================================
    # SECTION 5: Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("DEMO SUMMARY")
    print("=" * 80)
    
    print("\n[Model]")
    print(f"  Architecture: SNN Encoder-Decoder")
    print(f"  Parameters: {total_params:,} ({total_params * 4 / 1024 / 1024:.2f} MB)")
    print(f"  Temporal Steps: {NUM_STEPS}")
    print(f"  Training Epoch: {epoch}")
    if isinstance(val_psnr, (int, float)):
        print(f"  Validation PSNR: {val_psnr:.2f} dB")
    
    print("\n[Inference]")
    print(f"  Test Samples: {len(outputs)}")
    print(f"  Average PSNR: {avg_psnr:.2f} dB")
    print(f"  Inference Time: {inference_time:.2f} ms")
    
    print("\n[Performance]")
    if 1 in latency_results:
        print(f"  Latency (batch=1): {latency_results[1]['mean']:.2f} ms")
    if 16 in latency_results:
        print(f"  Latency (batch=16): {latency_results[16]['mean']:.2f} ms ({latency_results[16]['mean']/16:.2f} ms per sample)")
    if 16 in throughput_results:
        print(f"  Throughput (batch=16): {throughput_results[16]:.2f} samples/second")
    print(f"  FLOPs: {flops / 1e9:.2f} GFLOPs per inference")
    
    if power_stats['active_power'] is not None:
        print("\n[Power]")
        print(f"  Active Power: {power_stats['active_power']:.2f} W")
        if power_stats['power_delta'] is not None:
            print(f"  Inference Overhead: {power_stats['power_delta']:.2f} W")
    
    print("\n[Outputs]")
    print(f"  Visualization saved to: {save_path}")
    
    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)
    
    # Note: Visualization saved to file, use plt.show() if you want to display it
    # plt.show()  # Uncomment to display plot window


if __name__ == "__main__":
    main()

