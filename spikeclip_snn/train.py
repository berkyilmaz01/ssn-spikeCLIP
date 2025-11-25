"""
Training script for SNN Image Reconstruction
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from data.ncaltech101_dataset import NCaltech101Dataset
from models.snn_model import SNNReconstruction

# Setting up the training config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
# If GPU is available use GPU, else CPU
# Process 8 samples at a time

BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
NUM_BINS = 5
BETA = 0.9
NUM_STEPS = 5

# Storage for plotting
train_losses = []
epoch_losses = []

# Dataset Paths
EVENT_PATH = "C:/Users/berky/PycharmProjects/ssn-spikeCLIP/datasets/N-Caltech101/Caltech101/Caltech101"
IMAGE_PATH = "C:/Users/berky/PycharmProjects/ssn-spikeCLIP/datasets/101_ObjectCategories/101_ObjectCategories"

# Create a dataset and dataloader
print("\nLoading dataset...")
dataset = NCaltech101Dataset(
    root_dir=EVENT_PATH,
    num_bins=NUM_BINS,
    image_dir=IMAGE_PATH
)

# Create DataLoader for batching
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
print(f"Created DataLoader with batch size {BATCH_SIZE}")

# Create the model
print("\nCreating SNN model...")
model = SNNReconstruction(num_bins=NUM_BINS, beta=BETA).to(device)
print("Model created and moved to device")

# Loss Function and Optimizer
# Mean Squared Error for image reconstruction
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print(f"Using MSE loss and Adam optimizer (lr={LEARNING_RATE})")

# Training loop
print("\n" + "="*50)
print("Starting Training...")
print("="*50)

# Loop through all epochs
# Train the model
for epoch in range(NUM_EPOCHS):
    model.train()  # Set model to training mode
    epoch_loss = 0.0

    for batch_idx, (voxels, images, labels) in enumerate(dataloader):
        # Move data to device
        voxels = voxels.to(device)
        images = images.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(voxels, num_steps=NUM_STEPS)
        loss = criterion(outputs, images)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track loss
        epoch_loss += loss.item()
        train_losses.append(loss.item())

        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    # Print epoch summary
    avg_loss = epoch_loss / len(dataloader)
    epoch_losses.append(avg_loss)
    print(f"\n>>> Epoch [{epoch + 1}/{NUM_EPOCHS}] Complete - Average Loss: {avg_loss:.4f}\n")

# Save the model
print("\n" + "="*50)
print("Training Complete. Saving model...")
torch.save(model.state_dict(), "snn_reconstruction.pth")
print("Model saved as 'snn_reconstruction.pth'")
print("="*50)


