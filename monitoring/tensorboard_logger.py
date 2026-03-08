# ============================================================
# monitoring/tensorboard_logger.py
# TensorBoard logging for training metrics.
#
# TensorBoard is a visual dashboard to monitor training.
# Run it with: tensorboard --logdir=runs
# Then open your browser at: http://localhost:6006
# ============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.tensorboard import SummaryWriter
from config.settings import TENSORBOARD_LOG_DIR
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
import torch


class TensorBoardLogger:
    """
    Simple wrapper around TensorBoard's SummaryWriter.

    Logs: loss, accuracy, confusion matrix images, sample predictions.
    """

    def __init__(self, model_name):
        """
        Args:
            model_name (str): Name used for the log subdirectory (e.g. 'RNN', 'LSTM')
        """
        log_dir = os.path.join(TENSORBOARD_LOG_DIR, model_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.model_name = model_name
        print(f"TensorBoard logging to: {log_dir}")

    def log_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc):
        """
        Log metrics for one epoch.

        Args:
            epoch (int):        Current epoch number
            train_loss (float): Training loss
            val_loss (float):   Validation loss
            train_acc (float):  Training accuracy
            val_acc (float):    Validation accuracy
        """
        self.writer.add_scalars('Loss', {
            'train': train_loss,
            'val':   val_loss
        }, epoch)

        self.writer.add_scalars('Accuracy', {
            'train': train_acc,
            'val':   val_acc
        }, epoch)

    def log_confusion_matrix(self, cm, epoch):
        """
        Log confusion matrix as an image to TensorBoard.

        Args:
            cm (np.array): Confusion matrix
            epoch (int):   Current epoch
        """
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{self.model_name} Confusion Matrix (Epoch {epoch})')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

        # Convert plot to image tensor
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)
        image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1)[:3]

        self.writer.add_image('Confusion_Matrix', image_tensor, epoch)
        plt.close()

    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()
