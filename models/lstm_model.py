# ============================================================
# models/lstm_model.py
# LSTM (Long Short-Term Memory) for sentiment analysis.
#
# How LSTMs improve on RNNs:
#   - Add a "cell state" (long-term memory)
#   - Three gates control information flow:
#       Forget gate:  what to remove from memory
#       Input gate:   what new info to store
#       Output gate:  what to output from memory
#   - Solves the vanishing gradient problem!
#   - Better at learning long-range dependencies
# ============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
from config.settings import EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT, PAD_IDX


class LSTMModel(nn.Module):
    """
    LSTM for binary sentiment classification.

    Architecture:
      Embedding -> LSTM -> Dropout -> Linear -> Sigmoid
    """

    def __init__(self, vocab_size):
        super(LSTMModel, self).__init__()

        # Embedding layer: word ID -> dense vector
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=PAD_IDX)

        # LSTM layer: processes sequence with both hidden AND cell state
        self.lstm = nn.LSTM(
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0  # only active when num_layers > 1
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(DROPOUT)

        # Final classifier layer
        self.fc = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, text):
        """
        Forward pass through the LSTM.

        Args:
            text: Tensor of shape (batch_size, seq_len)

        Returns:
            output: Tensor of shape (batch_size,)
        """
        # Embedding lookup
        # No dropout here — apply dropout only before the final classifier
        embedded = self.embedding(text)

        # LSTM returns: output, (hidden_state, cell_state)
        # hidden: (num_layers, batch_size, hidden_dim)
        # cell:   (num_layers, batch_size, hidden_dim)
        output, (hidden, cell) = self.lstm(embedded)

        # Use top-layer final hidden state
        last_hidden = self.dropout(hidden[-1])

        # Classify
        prediction = self.fc(last_hidden)
        return prediction.squeeze(1)
