# ============================================================
# models/rnn_model.py
# Simple RNN (Recurrent Neural Network) for sentiment analysis.
#
# How RNNs work:
#   - Process the sequence word by word
#   - Maintain a "hidden state" that carries information forward
#   - The final hidden state represents the whole sequence
#   - That state is passed to a classifier
#
# Limitation: RNNs struggle with long sequences (vanishing gradient)
# ============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from config.settings import EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT, PAD_IDX


class RNNModel(nn.Module):
    """
    Vanilla RNN for binary sentiment classification.

    Architecture:
      Embedding -> RNN -> Dropout -> Linear -> Sigmoid
    """

    def __init__(self, vocab_size):
        """
        Args:
            vocab_size (int): Total number of words in vocabulary
        """
        super(RNNModel, self).__init__()

        # Embedding: converts word IDs to dense vectors
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=PAD_IDX)

        # RNN: processes the sequence step by step
        # batch_first=True means input shape is (batch, seq_len, embed_dim)
        self.rnn = nn.RNN(
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0  # only active when num_layers > 1
        )

        # Dropout: randomly zeros some neurons to prevent overfitting
        self.dropout = nn.Dropout(DROPOUT)

        # Linear: maps hidden state to a single output score
        self.fc = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, text):
        """
        Forward pass through the network.

        Args:
            text: Tensor of shape (batch_size, seq_len) -- token IDs

        Returns:
            output: Tensor of shape (batch_size,) -- sentiment scores
        """
        # Step 1: text -> embeddings
        # Shape: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        # Note: DO NOT apply dropout here — zeroing embedding values prevents
        # the model from learning stable word representations.
        embedded = self.embedding(text)

        # Step 2: embeddings -> RNN hidden states
        # output shape: (batch_size, seq_len, hidden_dim)
        # hidden shape: (num_layers, batch_size, hidden_dim)
        output, hidden = self.rnn(embedded)

        # Step 3: Use the LAST hidden state for classification
        # hidden[-1] is the top-layer final hidden state
        # Shape: (batch_size, hidden_dim)
        last_hidden = self.dropout(hidden[-1])

        # Step 4: Linear layer -> single score per review
        # Shape: (batch_size, hidden_dim) -> (batch_size, 1)
        prediction = self.fc(last_hidden)

        # Squeeze to remove last dimension: (batch_size, 1) -> (batch_size,)
        return prediction.squeeze(1)
