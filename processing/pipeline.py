# ============================================================
# processing/pipeline.py
# The complete text-to-tensor pipeline.
#
# This module ties together: tokenization -> encoding -> padding
# and creates PyTorch DataLoaders for training.
#
# Pipeline:
#   raw text -> tokens -> token IDs -> padded sequences -> DataLoader
# ============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset, DataLoader
from processing.tokenizer import simple_tokenize
from processing.vocabulary import Vocabulary
from config.settings import MAX_SEQ_LEN, PAD_IDX, BATCH_SIZE


class IMDbDataset(Dataset):
    """
    Custom PyTorch Dataset for IMDb reviews.

    PyTorch needs data in a specific format.
    This class wraps our DataFrame so PyTorch can read it.
    """

    def __init__(self, texts, labels, vocab):
        """
        Args:
            texts (list[str]):  Raw review texts
            labels (list[int]): Sentiment labels (0=negative, 1=positive)
            vocab (Vocabulary): Trained vocabulary object
        """
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        # Returns total number of samples in dataset
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Returns one (encoded_text, label) pair.
        The text is tokenized -> encoded -> padded/truncated here.
        """
        raw_text = self.texts[idx]
        label    = self.labels[idx]

        # Step 1: Tokenize
        tokens = simple_tokenize(raw_text)

        # Step 2: Encode tokens to IDs
        encoded = self.vocab.encode(tokens)

        # Step 3: Truncate if too long
        encoded = encoded[:MAX_SEQ_LEN]

        # Step 4: Pad if too short
        padding_needed = MAX_SEQ_LEN - len(encoded)
        encoded = encoded + [PAD_IDX] * padding_needed

        # Convert to PyTorch tensors
        text_tensor  = torch.tensor(encoded, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.float)

        return text_tensor, label_tensor


def build_vocab_from_df(train_df):
    """
    Build vocabulary using only the training data.
    (Never fit vocabulary on validation or test data!)

    Args:
        train_df: Training DataFrame with 'text' column

    Returns:
        Vocabulary: Trained vocabulary object
    """
    print("Building vocabulary from training data...")
    token_lists = [simple_tokenize(text) for text in train_df['text']]
    vocab = Vocabulary()
    vocab.build(token_lists)
    return vocab


def create_dataloaders(train_df, val_df, test_df, vocab):
    """
    Create PyTorch DataLoaders for train, validation, and test sets.

    DataLoaders handle:
    - Batching (group reviews into batches)
    - Shuffling (randomize order each epoch)
    - Parallel loading (load data in background)

    Args:
        train_df, val_df, test_df: DataFrames with 'text' and 'label' columns
        vocab: Trained Vocabulary object

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create Dataset objects
    train_dataset = IMDbDataset(
        train_df['text'].tolist(), train_df['label'].tolist(), vocab
    )
    val_dataset = IMDbDataset(
        val_df['text'].tolist(), val_df['label'].tolist(), vocab
    )
    test_dataset = IMDbDataset(
        test_df['text'].tolist(), test_df['label'].tolist(), vocab
    )

    # Create DataLoaders
    # shuffle=True for training (important for SGD)
    # shuffle=False for val/test (consistent evaluation)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    print(f"DataLoaders created:")
    print(f"  Train batches:      {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches:       {len(test_loader)}")

    return train_loader, val_loader, test_loader
