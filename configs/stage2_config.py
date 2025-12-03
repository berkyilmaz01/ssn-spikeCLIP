"""
Stage 2 Configuration: Prompt Learning
Following SpikeCLIP paper Section 3.2
https://arxiv.org/abs/2501.04477
"""
import os
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Paths from environment variables
EVENT_PATH = os.getenv("EVENT_PATH")
IMAGE_PATH = os.getenv("IMAGE_PATH")
STAGE1_CHECKPOINT = os.getenv("STAGE1_CHECKPOINT")
STAGE2_CHECKPOINT_DIR = os.getenv("STAGE2_CHECKPOINT_DIR")


# Model Settings
NUM_BINS = 5
BETA = 0.95
NUM_STEPS = 50

# We need to define the number of learnable
# context tokens and CLIP embedding dimensions
# Also, token poisiton to identify where to
# put the class token
N_CTX = 4
CTX_DIM = 512
CLASS_TOKEN_POSITION = "end"

# Training settings
BATCH_SIZE = 32
LEARNING_RATE = 0.002  # Increased for better prompt learning
NUM_EPOCHS = 100  # Increased to allow more learning
TEMPERATURE = 0.07
WEIGHT_DECAY = 1e-4  # Regularization to prevent prompt convergence

# LQ Image Degradation Settings (to create stronger quality gap)
DEGRADE_LQ = True  # Enable degradation of LQ images
NOISE_STD = 0.15  # Gaussian noise standard deviation 
BLUR_KERNEL_SIZE = 5  # Blur kernel size 

# DEGRADATION_LEVEL
DEGRADATION_LEVEL = "match_snn"

# Check device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HQ_DATASET_DIR = os.getenv("HQ_DATASET_DIR")
