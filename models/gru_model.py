# ============================================================
# models/gru_model.py
# GRU (Gated Recurrent Unit) for sentiment analysis.
#
# GRU is a simpler alternative to LSTM:
#   - Only two gates: Reset gate and Update gate
#   - No separate cell state (simpler than LSTM)
#   - Often performs similarly to LSTM with fewer parameters
#   - Faster to train than LSTM
#
# GRU is often the best balance between RNN (simple) and LSTM (complex).
# ============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
from config.settings import EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT, PAD_IDX


class GRUModel(nn.Module):
    """
    GRU for binary sentiment classification.

    Architecture:
      Embedding -> GRU -> Dropout -> Linear -> Sigmoid
    """

    def __init__(self, vocab_size):
        super(GRUModel, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=PAD_IDX)

        # GRU layer: simpler than LSTM, only hidden state (no cell state)
        self.gru = nn.GRU(
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0  # only active when num_layers > 1
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(DROPOUT)

        # Final classifier
        self.fc = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, text):
        """
        Forward pass through the GRU.

        Args:
            text: Tensor of shape (batch_size, seq_len)

        Returns:
            output: Tensor of shape (batch_size,)
        """
        # Embedding lookup
        # No dropout here — apply dropout only before the final classifier
        embedded = self.embedding(text)

        # GRU: returns output and final hidden state
        output, hidden = self.gru(embedded)

        # Use top-layer final hidden state for classification
        last_hidden = self.dropout(hidden[-1])

        # Linear classifier
        prediction = self.fc(last_hidden)
        return prediction.squeeze(1)
