# ============================================================
# training/early_stopping.py
# Early Stopping: stop training when validation loss stops improving.
#
# Why early stopping?
#   Without it, models can "overfit" -- they memorize the training
#   data but perform poorly on new data.
#
#   Early stopping monitors validation loss and stops training
#   when the model stops improving on unseen data.
# ============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import PATIENCE


class EarlyStopping:
    """
    Monitors validation loss and stops training if no improvement
    is seen for 'patience' epochs.

    Also saves the best model weights automatically.

    Usage:
        early_stop = EarlyStopping(patience=3)
        for epoch in ...:
            val_loss = validate(...)
            early_stop(val_loss, model)
            if early_stop.should_stop:
                break
    """

    def __init__(self, patience=PATIENCE, min_delta=0.001):
        """
        Args:
            patience (int):    How many epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as improvement
        """
        self.patience    = patience
        self.min_delta   = min_delta
        self.counter     = 0           # How many epochs without improvement
        self.best_loss   = None        # Best validation loss seen so far
        self.best_weights = None       # Best model weights
        self.should_stop = False       # Flag to signal training should stop

    def __call__(self, val_loss, model):
        """
        Check if validation loss improved. Save model if yes. Stop if patience exceeded.

        Args:
            val_loss (float): Current epoch validation loss
            model: PyTorch model
        """
        if self.best_loss is None:
            # First epoch -- save everything
            self.best_loss = val_loss
            self._save_weights(model)

        elif val_loss < self.best_loss - self.min_delta:
            # Improvement found!
            print(f"  Validation loss improved: {self.best_loss:.4f} -> {val_loss:.4f}")
            self.best_loss = val_loss
            self.counter = 0
            self._save_weights(model)

        else:
            # No improvement this epoch
            self.counter += 1
            print(f"  No improvement. Patience: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                print(f"Early stopping triggered! Best val loss: {self.best_loss:.4f}")
                self.should_stop = True

    def _save_weights(self, model):
        """Save a copy of the model's current weights."""
        import copy
        self.best_weights = copy.deepcopy(model.state_dict())

    def restore_best_weights(self, model):
        """Restore the model to its best weights after training."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            print("Best model weights restored.")
