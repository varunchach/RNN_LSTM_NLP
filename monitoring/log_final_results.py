# ============================================================
# monitoring/log_final_results.py
# Logs the final training results from all 3 models to TensorBoard.
#
# Run this after training to generate clean TensorBoard visualizations:
#   python monitoring/log_final_results.py
#   tensorboard --logdir=runs
# ============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.tensorboard import SummaryWriter

# ----------------------------------------------------------------
# Final training results from the last training run
# (RNN: 5 epochs, LSTM: 5 epochs, GRU: 5 epochs)
# ----------------------------------------------------------------

results = {
    "RNN": {
        "train_loss": [0.6999, 0.6977, 0.6987, 0.6864, 0.6794],
        "train_acc":  [0.4997, 0.5118, 0.5118, 0.5412, 0.5552],
        "val_loss":   [0.7038, 0.6960, 0.6932, 0.6935, 0.6974],
        "val_acc":    [0.4832, 0.4832, 0.5024, 0.5276, 0.5260],
        "test_accuracy":  0.5023,
        "test_precision": 0.5028,
        "test_recall":    0.4155,
        "test_f1":        0.4550,
    },
    "LSTM": {
        "train_loss": [0.6938, 0.6907, 0.6663, 0.6028, 0.4566],
        "train_acc":  [0.5014, 0.5286, 0.5698, 0.6804, 0.8053],
        "val_loss":   [0.6958, 0.6927, 0.7017, 0.5871, 0.4549],
        "val_acc":    [0.4972, 0.5152, 0.4932, 0.7140, 0.7944],
        "test_accuracy":  0.7851,
        "test_precision": 0.7915,
        "test_recall":    0.7742,
        "test_f1":        0.7827,
    },
    "GRU": {
        "train_loss": [0.6946, 0.4833, 0.2604, 0.1575, 0.0866],
        "train_acc":  [0.5181, 0.7513, 0.8944, 0.9429, 0.9729],
        "val_loss":   [0.6980, 0.3462, 0.3181, 0.3591, 0.4198],
        "val_acc":    [0.4876, 0.8500, 0.8732, 0.8692, 0.8568],
        "test_accuracy":  0.8604,
        "test_precision": 0.8599,
        "test_recall":    0.8610,
        "test_f1":        0.8605,
    },
}


def log_model(model_name, data):
    """Log training curves and final test metrics for one model."""
    writer = SummaryWriter(log_dir=f"runs/{model_name}")

    # --- Per-epoch training curves ---
    for epoch in range(1, len(data["train_loss"]) + 1):
        writer.add_scalars("Loss", {
            "train": data["train_loss"][epoch - 1],
            "val":   data["val_loss"][epoch - 1],
        }, epoch)
        writer.add_scalars("Accuracy", {
            "train": data["train_acc"][epoch - 1],
            "val":   data["val_acc"][epoch - 1],
        }, epoch)

    # --- Final test metrics (logged as scalars at step 0) ---
    writer.add_scalar("Test/Accuracy",  data["test_accuracy"],  0)
    writer.add_scalar("Test/Precision", data["test_precision"], 0)
    writer.add_scalar("Test/Recall",    data["test_recall"],    0)
    writer.add_scalar("Test/F1",        data["test_f1"],        0)

    writer.close()
    print(f"  {model_name}: logged {len(data['train_loss'])} epochs "
          f"| Test Acc: {data['test_accuracy']:.4f} | F1: {data['test_f1']:.4f}")


if __name__ == "__main__":
    print("Logging final training results to TensorBoard...")
    for model_name, data in results.items():
        log_model(model_name, data)
    print("\nDone! Run: tensorboard --logdir=runs")
    print("Then open: http://localhost:6006")
