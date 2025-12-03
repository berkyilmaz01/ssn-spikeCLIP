"""
SNN Performance Profiling Script
Measures latency, throughput, and power consumption for SNN model

Metrics:
- Latency: Time per inference (single sample and batch)
- Throughput: Samples processed per second
- Power: GPU power consumption and theoretical estimates
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "spikeclip_snn"))

from models.snn_model import SNNReconstruction

# Try to import dataset, but make it optional
try:
    from data.ncaltech101_dataset import NCaltech101Dataset
    DATASET_AVAILABLE = True
except ImportError:
    DATASET_AVAILABLE = False
    print("Warning: Dataset module not found. Will use random input tensors for profiling.")

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_BINS = 5
BETA = 0.95
NUM_STEPS = 50

# Paths from environment variables
STAGE1_CHECKPOINT = os.getenv("STAGE1_CHECKPOINT")
EVENT_PATH = os.getenv("EVENT_PATH")
IMAGE_PATH = os.getenv("IMAGE_PATH")

# Profiling parameters
WARMUP_ITERATIONS = 10
PROFILE_ITERATIONS = 100
BATCH_SIZES = [1, 4, 8, 16, 32]


def get_gpu_power():
    """
    Get current GPU power consumption using nvidia-smi
    Returns power in Watts, or None if not available
    """
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
    """
    Count total and trainable parameters in the model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def estimate_flops(model, input_shape, num_steps):
    """
    Estimate FLOPs (Floating Point Operations) for SNN forward pass
    This is an approximation based on layer operations
    
    Args:
        model: SNN model
        input_shape: Input tensor shape (B, C, H, W)
        num_steps: Number of temporal steps
    
    Returns:
        Estimated FLOPs count
    """
    batch_size, channels, height, width = input_shape
    
    # Encoder FLOPs (per timestep)
    # Conv1: 5 -> 64, kernel 3x3, stride 2
    conv1_flops = batch_size * 64 * (height // 2) * (width // 2) * 5 * 3 * 3
    
    # Conv2: 64 -> 128, kernel 3x3, stride 2
    conv2_flops = batch_size * 128 * (height // 4) * (width // 4) * 64 * 3 * 3
    
    # Conv3: 128 -> 256, kernel 3x3, stride 2
    conv3_flops = batch_size * 256 * (height // 8) * (width // 8) * 128 * 3 * 3
    
    # LIF neuron operations (simplified: 3 operations per neuron per timestep)
    # Membrane update, threshold check, reset
    lif_neurons = (64 * (height // 2) * (width // 2) + 
                   128 * (height // 4) * (width // 4) + 
                   256 * (height // 8) * (width // 8))
    lif_flops = batch_size * lif_neurons * 3
    
    # Encoder total (multiply by num_steps for temporal processing)
    encoder_flops = (conv1_flops + conv2_flops + conv3_flops + lif_flops) * num_steps
    
    # Temporal averaging
    avg_flops = batch_size * 256 * (height // 8) * (width // 8)
    
    # Decoder FLOPs
    # DeConv1: 256 -> 128, kernel 4x4, stride 2
    deconv1_flops = batch_size * 128 * (height // 4) * (width // 4) * 256 * 4 * 4
    
    # DeConv2: 128 -> 64, kernel 4x4, stride 2
    deconv2_flops = batch_size * 64 * (height // 2) * (width // 2) * 128 * 4 * 4
    
    # DeConv3: 64 -> 1, kernel 4x4, stride 2
    deconv3_flops = batch_size * 1 * height * width * 64 * 4 * 4
    
    # ReLU operations (2 layers)
    relu_flops = batch_size * (128 + 64) * (height // 2) * (width // 2)
    
    # Sigmoid
    sigmoid_flops = batch_size * height * width
    
    # Interpolation
    interpolate_flops = batch_size * height * width * 4  # Bilinear interpolation
    
    decoder_flops = deconv1_flops + deconv2_flops + deconv3_flops + relu_flops + sigmoid_flops + interpolate_flops
    
    total_flops = encoder_flops + avg_flops + decoder_flops
    return total_flops


def measure_latency(model, input_tensor, num_steps, num_iterations, warmup_iterations):
    """
    Measure inference latency
    
    Args:
        model: SNN model
        input_tensor: Input tensor
        num_steps: Number of temporal steps
        num_iterations: Number of iterations to average
        warmup_iterations: Number of warmup iterations
    
    Returns:
        Mean latency (ms), std latency (ms), min latency (ms), max latency (ms)
    """
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(input_tensor, num_steps=num_steps)
    
    # Synchronize GPU before timing
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
        'median': np.median(latencies),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99)
    }


def measure_throughput(model, input_tensor, num_steps, duration_seconds=5.0):
    """
    Measure throughput (samples per second) over a duration
    
    Args:
        model: SNN model
        input_tensor: Input tensor
        num_steps: Number of temporal steps
        duration_seconds: Duration to measure throughput
    
    Returns:
        Throughput (samples/second)
    """
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


def measure_power_consumption(model, input_tensor, num_steps, duration_seconds=10.0):
    """
    Measure power consumption during inference
    
    Args:
        model: SNN model
        input_tensor: Input tensor
        num_steps: Number of temporal steps
        duration_seconds: Duration to measure power
    
    Returns:
        Average power (W), idle power (W), active power (W)
    """
    # Measure idle power
    idle_powers = []
    for _ in range(5):
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
            
            # Sample power every 0.5 seconds
            if num_iterations % 10 == 0:
                power = get_gpu_power()
                if power is not None:
                    active_powers.append(power)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    active_power = np.mean(active_powers) if active_powers else None
    avg_power = active_power
    
    return {
        'idle_power': idle_power,
        'active_power': active_power,
        'avg_power': avg_power,
        'power_delta': (active_power - idle_power) if (active_power and idle_power) else None
    }


def main():
    print("=" * 80)
    print("SNN Performance Profiling")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print()
    
    # Load model
    print("[1/5] Loading SNN model...")
    model = SNNReconstruction(num_bins=NUM_BINS, beta=BETA).to(DEVICE)
    
    if os.path.exists(STAGE1_CHECKPOINT):
        checkpoint = torch.load(STAGE1_CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    else:
        print("  Warning: Checkpoint not found, using random weights")
    
    model.eval()
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    print()
    
    # Load dataset for real input data
    print("[2/5] Loading dataset...")
    if DATASET_AVAILABLE:
        try:
            dataset = NCaltech101Dataset(
                root_dir=EVENT_PATH,
                num_bins=NUM_BINS,
                image_dir=IMAGE_PATH
            )
            print(f"  Loaded {len(dataset)} samples")
            
            # Get a sample for single inference
            sample_voxel, _, _ = dataset[0]
            sample_voxel = sample_voxel.unsqueeze(0).to(DEVICE)
            print(f"  Input shape: {sample_voxel.shape}")
        except Exception as e:
            print(f"  Warning: Could not load dataset: {e}")
            print("  Using random input tensor")
            sample_voxel = torch.randn(1, NUM_BINS, 180, 240).to(DEVICE)
    else:
        print("  Dataset module not available")
        print("  Using random input tensor")
        sample_voxel = torch.randn(1, NUM_BINS, 180, 240).to(DEVICE)
    print()
    
    # Estimate FLOPs
    print("[3/5] Estimating FLOPs...")
    input_shape = sample_voxel.shape
    flops = estimate_flops(model, input_shape, NUM_STEPS)
    print(f"  Estimated FLOPs per inference: {flops / 1e9:.2f} GFLOPs")
    print(f"  Estimated FLOPs per timestep: {flops / NUM_STEPS / 1e9:.4f} GFLOPs")
    print()
    
    # Measure latency for different batch sizes
    print("[4/5] Measuring Latency...")
    print("-" * 80)
    print(f"{'Batch Size':<12} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12}")
    print("-" * 80)
    
    latency_results = {}
    
    for batch_size in BATCH_SIZES:
        # Create batch input
        if batch_size == 1:
            batch_input = sample_voxel
        else:
            batch_input = sample_voxel.repeat(batch_size, 1, 1, 1)
        
        # Measure latency
        latency_stats = measure_latency(
            model, batch_input, NUM_STEPS, 
            PROFILE_ITERATIONS, WARMUP_ITERATIONS
        )
        
        latency_results[batch_size] = latency_stats
        
        print(f"{batch_size:<12} "
              f"{latency_stats['mean']:<12.3f} "
              f"{latency_stats['std']:<12.3f} "
              f"{latency_stats['min']:<12.3f} "
              f"{latency_stats['max']:<12.3f} "
              f"{latency_stats['p95']:<12.3f} "
              f"{latency_stats['p99']:<12.3f}")
    
    print()
    
    # Measure throughput
    print("[5/5] Measuring Throughput...")
    print("-" * 80)
    print(f"{'Batch Size':<12} {'Throughput (samples/s)':<25} {'Throughput (images/s)':<25}")
    print("-" * 80)
    
    throughput_results = {}
    
    for batch_size in BATCH_SIZES:
        if batch_size == 1:
            batch_input = sample_voxel
        else:
            batch_input = sample_voxel.repeat(batch_size, 1, 1, 1)
        
        throughput = measure_throughput(model, batch_input, NUM_STEPS, duration_seconds=3.0)
        throughput_results[batch_size] = throughput
        
        # Throughput in images per second (accounting for batch)
        images_per_second = throughput / batch_size if batch_size > 0 else throughput
        
        print(f"{batch_size:<12} {throughput:<25.2f} {images_per_second:<25.2f}")
    
    print()
    
    # Measure power consumption
    print("[6/6] Measuring Power Consumption...")
    print("-" * 80)
    
    # Use batch size 16 for power measurement (typical training batch)
    power_batch_size = 16
    if power_batch_size == 1:
        power_input = sample_voxel
    else:
        power_input = sample_voxel.repeat(power_batch_size, 1, 1, 1)
    
    power_stats = measure_power_consumption(model, power_input, NUM_STEPS, duration_seconds=10.0)
    
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
    
    # Theoretical power estimate (if GPU power not available)
    if power_stats['active_power'] is None:
        print("\n  Theoretical Power Estimate:")
        print("  Note: This is a rough estimate based on FLOPs")
        # Typical GPU: ~100-200 GFLOPS per Watt
        # Assume 150 GFLOPS/W for modern GPU
        gflops_per_watt = 150.0
        estimated_power_watts = (flops * throughput_results.get(1, 1.0)) / 1e9 / gflops_per_watt
        print(f"  Estimated Power (single inference): {estimated_power_watts:.2f} W")
    
    print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Model Architecture:")
    print(f"  - Encoder: 3 Conv2d + LIF layers")
    print(f"  - Decoder: 3 ConvTranspose2d + ReLU layers")
    print(f"  - Temporal Steps: {NUM_STEPS}")
    print(f"  - Input Shape: {input_shape}")
    print()
    print(f"Performance Metrics (Batch Size = 1):")
    print(f"  - Latency: {latency_results[1]['mean']:.3f} ms (mean)")
    print(f"  - Latency: {latency_results[1]['median']:.3f} ms (median)")
    print(f"  - Throughput: {throughput_results[1]:.2f} samples/second")
    print(f"  - FLOPs: {flops / 1e9:.2f} GFLOPs per inference")
    print()
    print(f"Performance Metrics (Batch Size = 16):")
    if 16 in latency_results:
        print(f"  - Latency: {latency_results[16]['mean']:.3f} ms (mean)")
        print(f"  - Latency per sample: {latency_results[16]['mean'] / 16:.3f} ms")
    if 16 in throughput_results:
        print(f"  - Throughput: {throughput_results[16]:.2f} samples/second")
        print(f"  - Images per second: {throughput_results[16] / 16:.2f}")
    print()
    if power_stats['active_power'] is not None:
        print(f"Power Consumption:")
        print(f"  - Active Power: {power_stats['active_power']:.2f} W")
        if power_stats['power_delta'] is not None:
            print(f"  - Inference Power: {power_stats['power_delta']:.2f} W")
    print("=" * 80)


if __name__ == "__main__":
    main()

