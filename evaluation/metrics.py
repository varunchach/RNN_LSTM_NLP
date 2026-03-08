# ============================================================
# evaluation/metrics.py
# Compute evaluation metrics: accuracy, precision, recall, F1.
# ============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)


def get_predictions(model, loader, device):
    """
    Run model on all batches and collect true labels and predictions.

    Args:
        model:  Trained PyTorch model
        loader: DataLoader (val or test)
        device: 'cuda' or 'cpu'

    Returns:
        all_labels (np.array): True labels
        all_preds  (np.array): Predicted labels (0 or 1)
    """
    model.eval()
    all_labels = []
    all_preds  = []

    with torch.no_grad():
        for texts, labels in loader:
            texts  = texts.to(device)
            labels = labels.to(device)

            outputs = model(texts)
            preds   = (torch.sigmoid(outputs) >= 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def compute_metrics(true_labels, pred_labels):
    """
    Compute a full set of classification metrics.

    Args:
        true_labels (np.array): Ground truth labels
        pred_labels (np.array): Model predictions

    Returns:
        dict: Dictionary with accuracy, precision, recall, f1, confusion_matrix
    """
    return {
        "accuracy":         accuracy_score(true_labels, pred_labels),
        "precision":        precision_score(true_labels, pred_labels),
        "recall":           recall_score(true_labels, pred_labels),
        "f1":               f1_score(true_labels, pred_labels),
        "confusion_matrix": confusion_matrix(true_labels, pred_labels)
    }


def print_metrics(metrics, model_name="Model"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'='*40}")
    print(f"  {model_name} Evaluation Results")
    print(f"{'='*40}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"{'='*40}\n")
