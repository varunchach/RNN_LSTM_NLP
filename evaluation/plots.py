# ============================================================
# evaluation/plots.py
# Visualizations for training history and evaluation results.
# ============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_training_history(history, model_name="Model", save_path=None):
    """
    Plot training and validation loss and accuracy over epochs.

    Args:
        history (dict): Contains 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        model_name (str): Name to display in title
        save_path (str): If provided, save plot to this path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history['train_loss']) + 1)

    # --- Loss plot ---
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Training Loss')
    ax1.plot(epochs, history['val_loss'],   'r-o', label='Validation Loss')
    ax1.set_title(f'{model_name} -- Loss per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # --- Accuracy plot ---
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'],   'r-o', label='Validation Accuracy')
    ax2.set_title(f'{model_name} -- Accuracy per Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(cm, model_name="Model", save_path=None):
    """
    Plot a confusion matrix as a heatmap.

    Args:
        cm (np.array): 2x2 confusion matrix [[TN, FP], [FN, TP]]
        model_name (str): Model name for title
        save_path (str): Save path (optional)
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Predicted Negative', 'Predicted Positive'],
        yticklabels=['Actual Negative',    'Actual Positive']
    )
    plt.title(f'{model_name} -- Confusion Matrix')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_model_comparison(results_dict, save_path=None):
    """
    Bar chart comparing multiple models on key metrics.

    Args:
        results_dict (dict): {model_name: {accuracy, f1, ...}, ...}
        save_path (str): Save path (optional)
    """
    model_names = list(results_dict.keys())
    metrics     = ['accuracy', 'precision', 'recall', 'f1']

    x = np.arange(len(model_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, metric in enumerate(metrics):
        values = [results_dict[m][metric] for m in model_names]
        ax.bar(x + i * width, values, width, label=metric.capitalize())

    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
