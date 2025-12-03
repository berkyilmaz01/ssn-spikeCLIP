"""
Stage 3 Configuration: Quality-Guided SNN Fine-tuning
Following SpikeCLIP paper Section 3.3
https://arxiv.org/abs/2501.04477
"""

# Import libraries
import os
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Paths from environment variables
EVENT_PATH = os.getenv("EVENT_PATH")
IMAGE_PATH = os.getenv("IMAGE_PATH")
STAGE1_CHECKPOINT = os.getenv("STAGE1_CHECKPOINT")
STAGE2_CHECKPOINT = os.getenv("STAGE2_CHECKPOINT")
STAGE3_CHECKPOINT_DIR = os.getenv("STAGE3_CHECKPOINT_DIR")

# Model Settings
NUM_BINS = 5
BETA = 0.95
NUM_STEPS = 50
N_CTX = 4

# Stage 3 Loss Settings (following paper Eq. 11)
# L_total = L_class + λ * L_prompt
# Weight for prompt loss (quality guidance) - paper uses λ = 100
LAMBDA_PROMPT = 100.0

# Training Settings
# Smaller batch for fine-tuning
# Lower LR for fine-tuning
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
NUM_EPOCHS = 30
TEMPERATURE = 0.07

# Check device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")