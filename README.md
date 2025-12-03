# SNN-CLIP: Event-Based Image Reconstruction with CLIP Guidance

Implementation of **SNN-CLIP: Event-Based Image Reconstruction via Spiking Neural Networks with CLIP Guidance** (arXiv:2501.04477).

<img width="2985" height="1431" alt="stage1_vs_stage3_comparison" src="https://github.com/user-attachments/assets/dad96311-1639-46e3-b1f4-c086c3a2ac39" />

This repository contains a three-stage training pipeline for high-quality image reconstruction from event camera data using Spiking Neural Networks (SNNs) and CLIP semantic guidance.

## Weight Path
Google Drive : https://drive.google.com/drive/folders/1bl45-LRSiWhcC7WyLEiQWHtf9rxWpUXG?usp=sharing

## Overview

SpikeCLIP combines:
- **Spiking Neural Networks (SNNs)** for efficient event-to-image reconstruction
- **CLIP (Contrastive Language-Image Pre-training)** for semantic alignment and quality guidance
- **Three-stage training pipeline**:
  1. **Stage 1**: SNN reconstruction with InfoNCE loss for semantic alignment
  2. **Stage 2**: Prompt learning to distinguish high-quality vs low-quality images
  3. **Stage 3**: Quality-guided SNN fine-tuning using learned prompts

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- PyTorch 2.0+

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/ssn-spikeCLIP.git
cd ssn-spikeCLIP
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

4. **Configure environment:**
```bash
# Copy example environment file
cp .env.example .env

# Edit .env and update paths to match your system
# Required variables: EVENT_PATH, IMAGE_PATH, STAGE1_CHECKPOINT, etc.
```

## Dataset Preparation

### N-Caltech101 Dataset

**Option 1: Automatic Setup (Recommended)**

Run the data download script:

```bash
python scripts/download_data.py
```

This script will:
- Check for existing datasets
- Attempt to download Caltech101 images automatically
- Provide instructions for downloading N-Caltech101 event data (may require manual download due to Baidu restrictions)
- Verify dataset structure

**Option 2: Manual Setup**

1. **Download N-Caltech101:**
   - Event data: Visit https://www.garrickorchard.com/datasets/n-caltech101
   - Extract to `datasets/N-Caltech101/Caltech101/Caltech101/`
   - Image data: Visit http://www.vision.caltech.edu/Image_Datasets/Caltech101/
   - Extract to `datasets/101_ObjectCategories/101_ObjectCategories/`

2. **Configure Environment:**
   - Copy `.env.example` to `.env`
   - Update paths in `.env` file:
     ```
     EVENT_PATH=path/to/datasets/N-Caltech101/Caltech101/Caltech101
     IMAGE_PATH=path/to/datasets/101_ObjectCategories/101_ObjectCategories
     ```

## Training Pipeline

### Stage 1: SNN Reconstruction with Semantic Alignment

Train the SNN to reconstruct images from event voxels using CLIP-based InfoNCE loss:

```bash
python spikeclip_snn/train_spikeclip.py
```

**Output:**
- Best model saved to `checkpoints/spikeclip_best.pth`
- Training curves saved to `checkpoints/spikeclip_training_curves.png`

**Key hyperparameters:**
- `NUM_BINS`: Number of temporal bins (default: 5)
- `BETA`: Leaky integrate-and-fire threshold (default: 0.95)
- `NUM_STEPS`: SNN simulation steps (default: 50)
- `TEMPERATURE`: InfoNCE temperature (default: 0.07)

### Stage 2: Prompt Learning

Learn HQ/LQ prompts to distinguish high-quality from low-quality images:

```bash
python scripts/train_stage2.py
```

**Output:**
- Learned prompts saved to `checkpoints/stage2/prompt_best.pth`
- Training curves saved to `checkpoints/stage2/stage2_training_curves.png`

**Key hyperparameters:**
- `N_CTX`: Number of learnable context tokens (default: 4)
- `BATCH_SIZE`: Training batch size (default: 32)
- `LEARNING_RATE`: Prompt learning rate (default: 0.001)

### Stage 3: Quality-Guided Fine-tuning

Fine-tune SNN using learned prompts for quality improvement:

```bash
python scripts/train_stage3.py
```

**Output:**
- Fine-tuned model saved to `checkpoints/stage3/snn_stage3_best.pth`
- Training curves saved to `checkpoints/stage3/stage3_training_curves.png`

**Key hyperparameters:**
- `LAMBDA_PROMPT`: Weight for prompt loss (default: 100.0)
- `BATCH_SIZE`: Fine-tuning batch size (default: 16)
- `LEARNING_RATE`: Fine-tuning learning rate (default: 0.0001)

## Project Structure

```
ssn-spikeCLIP/
├── configs/                 # Configuration files for each stage
│   ├── stage2_config.py
│   └── stage3_config.py
├── scripts/                 # Training and utility scripts
│   ├── train_stage2.py
│   ├── train_stage3.py
│   ├── test_model.py        # Test script for inference
│   ├── profile_snn.py       # Performance profiling
│   └── download_data.py     # Dataset download helper
├── spikeclip_snn/          # Core SNN implementation
│   ├── models/
│   │   ├── snn_model.py    # SNN reconstruction model
│   │   └── prompt_learner.py # CoOp-style prompt learner
│   ├── utils/
│   │   ├── clip_utils.py   # CLIP utilities and InfoNCE loss
│   │   └── event_utils.py  # Event processing utilities
│   └── train_spikeclip.py  # Stage 1 training script
├── losses/                  # Loss functions
│   ├── prompt_loss.py     # Stage 2 prompt loss
│   └── quality_loss.py    # Stage 3 quality loss
├── configs/                 # Configuration files
│   ├── stage2_config.py
│   └── stage3_config.py
├── data/                    # Dataset loaders
│   └── ncaltech101_dataset.py
├── .env.example            # Environment variable template
├── tests/                   # Testing and verification scripts
│   ├── verify_stage2_prompts.py
│   └── visualize_stage2.py
├── utils/                   # Utility modules
│   └── logger.py           # Professional logging utility
└── requirements.txt         # Python dependencies
```

## Logging

The project includes a professional logging utility. To use it alongside your existing print statements:

```python
from utils.logger import setup_logger

# Setup logger (optional)
logger = setup_logger(log_dir="logs")

# Use logger
logger.info("Training started")
logger.warning("Low PSNR detected")
logger.error("Error occurred")

# Your existing print statements still work
print("This still works!")
```

Logs are saved to `logs/spikeclip_YYYYMMDD_HHMMSS.log` and also printed to console.

## Testing

### Quick Test

Run the test script to verify model inference and see outputs:

```bash
python scripts/test_model.py
```

This will:
- Load trained Stage 1 model
- Run inference on sample data
- Display input events, ground truth, and reconstructed images
- Show metrics (PSNR, latency, throughput)
- Save visualization to `checkpoints/test_output.png`

### Performance Profiling

Profile model performance (latency, throughput, power):

```bash
python scripts/profile_snn.py
```

This measures:
- **Latency**: Inference time per sample and batch
- **Throughput**: Samples processed per second
- **Power**: GPU power consumption during inference
- **FLOPs**: Estimated floating point operations

## Evaluation

After training, evaluate your models:

```bash
# Verify Stage 2 prompts
python tests/verify_stage2_prompts.py

# Visualize Stage 2 results
python tests/visualize_stage2.py

# Compare Stage 1 vs Stage 3 outputs
python tests/visualize_stage3.py
```

## Results

### Model Performance

**Model Architecture:**
- Parameters: 1,028,545 (3.92 MB)
- FLOPs: 24.30 GFLOPs per inference
- Temporal Steps: 50

**Inference Performance (RTX 4080 SUPER):**

| Batch Size | Latency (ms) | Latency/Sample (ms) | Throughput (samples/s) |
|------------|--------------|---------------------|------------------------|
| 1          | 102.00       | 102.00              | 9.84                   |
| 4          | 104.64       | 26.16               | 39.76                  |
| 8          | 103.91       | 12.99               | 80.07                  |
| 16         | 148.48       | 9.28                | 109.05                 |
| 32         | 375.71       | 11.74               | 84.89                  |

**Power Consumption:**
- Idle Power: 110.14 W
- Active Power: 204.63 W
- Inference Overhead: 94.49 W

**Image Quality:**
- Stage 1 PSNR: ~9.80 dB
- Stage 3 PSNR: ~12.24 dB
- Improvement: +2.44 dB

**Recommendations:**
- Use batch size 16 for optimal throughput (109 samples/s)
- Single-sample latency: ~102 ms
- Batch processing reduces per-sample latency significantly

## Citation

If you use this code in your research, please cite:

```bibtex
@article{spikeclip2025,
  title={SpikeCLIP: Event-Based Image Reconstruction via Spiking Neural Networks with CLIP Guidance},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:2501.04477},
  year={2025}
}
```

## License

This project is licensed under the MIT License.

## Visualization
<img width="2000" height="1000" alt="stage_2" src="https://github.com/user-attachments/assets/77c46d1d-01fb-4a04-b64d-dd163bd10653" />

<img width="1500" height="1000" alt="stage_2_1" src="https://github.com/user-attachments/assets/3aa9419e-2dae-4889-a4b0-69c765e54e50" />

<img width="525" height="920" alt="image" src="https://github.com/user-attachments/assets/8c4d1f08-ac75-4975-88c4-391da73db100" />

<img width="3535" height="1262" alt="stage1_visualization" src="https://github.com/user-attachments/assets/01a22c02-c4fa-4bd1-bf1a-5aa1b8fca6b5" />

<img width="3535" height="1262" alt="stage1_visualization_2" src="https://github.com/user-attachments/assets/43d04049-3d24-4415-a611-af4a20054789" />

<img width="3535" height="1262" alt="stage1_visualization_3" src="https://github.com/user-attachments/assets/3f276b55-d1f9-4258-a3b6-f96efce31148" />

<img width="3535" height="1262" alt="stage1_visualization_4" src="https://github.com/user-attachments/assets/892ac3f2-a2d3-4676-ad6b-1cdfb61bccfe" />

## Acknowledgments

- OpenAI CLIP: https://github.com/openai/CLIP
- CoOp: https://github.com/KaiyangZhou/CoOp
- SpikeCLIP Paper: https://arxiv.org/abs/2501.04477

## Summary
implemented SpikeCLIP, a three-stage pipeline for event camera image reconstruction:
Stage 1: Trained a Spiking Neural Network using contrastive learning with CLIP, achieving 9.80 dB PSNR on N-Caltech101.
Stage 2: Learned quality-aware prompts using CoOp-style prompt learning, achieving 100% accuracy in distinguishing high-quality from low-quality reconstructions.
Stage 3: Applied quality-guided fine-tuning, improving PSNR by +2.14 dB. However, we observed that directly optimizing for CLIP similarity can introduce artifacts, requiring regularization techniques like Total Variation loss.

