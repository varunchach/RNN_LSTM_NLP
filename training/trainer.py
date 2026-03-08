# ============================================================
# training/trainer.py
# The training and validation loops.
#
# The training loop:
#   For each batch:
#     1. Forward pass: make predictions
#     2. Compute loss: compare predictions to true labels
#     3. Backward pass: compute gradients
#     4. Update weights: optimizer step
# ============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Train the model for one full pass over the training data.

    Args:
        model:     PyTorch model
        loader:    Training DataLoader
        optimizer: Optimizer (e.g., Adam)
        criterion: Loss function (e.g., BCEWithLogitsLoss)
        device:    'cuda' or 'cpu'

    Returns:
        avg_loss (float), avg_accuracy (float)
    """
    model.train()  # Set model to training mode (enables dropout)

    total_loss     = 0
    total_correct  = 0
    total_samples  = 0

    # tqdm shows a nice progress bar
    for texts, labels in tqdm(loader, desc="Training", leave=False):
        # Move data to device (GPU if available)
        texts  = texts.to(device)
        labels = labels.to(device)

        # Step 1: Clear old gradients
        optimizer.zero_grad()

        # Step 2: Forward pass -- get predictions
        predictions = model(texts)

        # Step 3: Compute loss
        loss = criterion(predictions, labels)

        # Step 4: Backward pass -- compute gradients
        loss.backward()

        # Step 4b: Clip gradients to prevent exploding gradients (critical for RNNs)
        # If gradients grow too large, they cause NaN weights and random predictions.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Step 5: Update weights
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * len(labels)

        # Convert raw scores to binary predictions
        predicted_labels = (torch.sigmoid(predictions) >= 0.5).float()
        total_correct  += (predicted_labels == labels).sum().item()
        total_samples  += len(labels)

    avg_loss     = total_loss / total_samples
    avg_accuracy = total_correct / total_samples
    return avg_loss, avg_accuracy


def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on validation or test data.

    No gradient computation -- faster and uses less memory.

    Args:
        model:     PyTorch model
        loader:    DataLoader
        criterion: Loss function
        device:    'cuda' or 'cpu'

    Returns:
        avg_loss (float), avg_accuracy (float)
    """
    model.eval()  # Evaluation mode (disables dropout)

    total_loss    = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # No gradient tracking needed
        for texts, labels in tqdm(loader, desc="Evaluating", leave=False):
            texts  = texts.to(device)
            labels = labels.to(device)

            predictions = model(texts)
            loss = criterion(predictions, labels)

            total_loss += loss.item() * len(labels)
            predicted_labels = (torch.sigmoid(predictions) >= 0.5).float()
            total_correct  += (predicted_labels == labels).sum().item()
            total_samples  += len(labels)

    avg_loss     = total_loss / total_samples
    avg_accuracy = total_correct / total_samples
    return avg_loss, avg_accuracy
