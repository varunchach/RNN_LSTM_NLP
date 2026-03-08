# ============================================================
# main.py
# Training entry point.
#
# Run this file to train RNN, LSTM, and GRU models on IMDb data.
# Usage:
#   python main.py
#
# After training, start the API with:
#   uvicorn api.app:app --reload
# ============================================================

import os
import sys
import pickle

import torch
import torch.nn as nn

# Import all project modules
from ingestion.loader import load_imdb_as_dataframe
from processing.pipeline import build_vocab_from_df, create_dataloaders
from models.rnn_model import RNNModel
from models.lstm_model import LSTMModel
from models.gru_model import GRUModel
from training.trainer import train_one_epoch, evaluate
from training.early_stopping import EarlyStopping
from evaluation.metrics import get_predictions, compute_metrics, print_metrics
from evaluation.plots import plot_training_history, plot_confusion_matrix, plot_model_comparison
from monitoring.tensorboard_logger import TensorBoardLogger
from config.settings import (
    NUM_EPOCHS, LEARNING_RATE, MODEL_SAVE_DIR, PATIENCE
)


def train_model(model, model_name, train_loader, val_loader, device):
    """
    Full training loop for one model.
    Handles optimizer, loss, early stopping, and TensorBoard logging.

    Returns:
        model: Trained model with best weights restored
        history (dict): Training metrics per epoch
    """
    print(f"\n{'='*50}")
    print(f"  Training {model_name}")
    print(f"{'='*50}")

    # Move model to device (GPU or CPU)
    model = model.to(device)

    # Optimizer: Adam works well for most NLP tasks
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Loss: BCEWithLogitsLoss = sigmoid + binary cross entropy (numerically stable)
    criterion = nn.BCEWithLogitsLoss()

    # Early stopping and TensorBoard
    early_stopper = EarlyStopping(patience=PATIENCE)
    tb_logger     = TensorBoardLogger(model_name)

    # Store metrics for plotting
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc':  [], 'val_acc':  []
    }

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        # Train and evaluate
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}   | Val Acc:   {val_acc:.4f}")

        # Log to TensorBoard
        tb_logger.log_epoch(epoch, train_loss, val_loss, train_acc, val_acc)

        # Store in history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Early stopping check
        early_stopper(val_loss, model)
        if early_stopper.should_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    # Restore best weights
    early_stopper.restore_best_weights(model)
    tb_logger.close()

    return model, history


def main():
    # --- Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # --- Load Data ---
    train_df, val_df, test_df = load_imdb_as_dataframe()

    # --- Build Vocabulary ---
    vocab = build_vocab_from_df(train_df)

    # --- Save Vocabulary (needed for inference) ---
    vocab_path = os.path.join(MODEL_SAVE_DIR, "vocab.pkl")
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Vocabulary saved to {vocab_path}")

    # --- Create DataLoaders ---
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, vocab
    )

    # --- Define Models ---
    models_to_train = {
        "RNN":  RNNModel(vocab.vocab_size),
        "LSTM": LSTMModel(vocab.vocab_size),
        "GRU":  GRUModel(vocab.vocab_size),
    }

    # --- Train Each Model ---
    all_results = {}

    for model_name, model in models_to_train.items():
        # Train
        trained_model, history = train_model(
            model, model_name, train_loader, val_loader, device
        )

        # Save model
        model_path = os.path.join(MODEL_SAVE_DIR, f"{model_name.lower()}_model.pt")
        torch.save(trained_model, model_path)
        print(f"Model saved: {model_path}")

        # Evaluate on test set
        true_labels, pred_labels = get_predictions(trained_model, test_loader, device)
        metrics = compute_metrics(true_labels, pred_labels)
        all_results[model_name] = metrics
        print_metrics(metrics, model_name)

        # Plot training history
        plot_training_history(history, model_name)
        plot_confusion_matrix(metrics['confusion_matrix'], model_name)

    # --- Compare All Models ---
    print("\nModel Comparison:")
    plot_model_comparison(all_results)

    # --- Save the best model for API use ---
    best_model_name = max(all_results, key=lambda m: all_results[m]['accuracy'])
    print(f"\nBest model: {best_model_name}")

    best_model_path  = os.path.join(MODEL_SAVE_DIR, f"{best_model_name.lower()}_model.pt")
    final_model_path = os.path.join(MODEL_SAVE_DIR, "best_model.pt")

    import shutil
    shutil.copy(best_model_path, final_model_path)
    print(f"Best model saved as: {final_model_path}")
    print("\nTraining complete!")
    print("Start the API with: uvicorn api.app:app --reload")
    print("View TensorBoard with: tensorboard --logdir=runs")


if __name__ == "__main__":
    main()
