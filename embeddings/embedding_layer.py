# ============================================================
# embeddings/embedding_layer.py
# Word Embeddings: converting word IDs to dense vectors.
#
# Instead of using one-hot vectors (sparse, high-dimensional),
# embeddings map each word to a small dense vector.
#
# Similar words get similar vectors!
# Example: embedding("king") ≈ embedding("queen")
#
# This is the same idea as Word2Vec and GloVe.
# ============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
from config.settings import PAD_IDX


def create_embedding_layer(vocab_size, embedding_dim):
    """
    Create an embedding layer for the model.

    The embedding layer is a lookup table:
      - Input:  word index (integer)
      - Output: embedding vector (float vector of size embedding_dim)

    padding_idx=PAD_IDX means the PAD token always maps to a zero vector.
    This prevents padding from affecting gradient updates.

    Args:
        vocab_size (int):    Number of words in vocabulary
        embedding_dim (int): Size of each embedding vector

    Returns:
        nn.Embedding: PyTorch embedding layer
    """
    embedding = nn.Embedding(
        num_embeddings=vocab_size,    # One vector per word
        embedding_dim=embedding_dim,  # Size of each vector
        padding_idx=PAD_IDX           # PAD always gets zero vector
    )

    print(f"Embedding layer created:")
    print(f"  Vocabulary size:  {vocab_size}")
    print(f"  Embedding dim:    {embedding_dim}")
    print(f"  Total parameters: {vocab_size * embedding_dim:,}")

    return embedding
