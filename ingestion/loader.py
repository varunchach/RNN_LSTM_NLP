# ============================================================
# ingestion/loader.py
# Loads the dataset and splits it into train/validation/test.
# Keeps data loading logic in one clean place.
# ============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datasets import load_dataset
from config.settings import (
    DATASET_NAME, DATA_CACHE_DIR,
    TRAIN_SPLIT, VAL_SPLIT
)


def load_imdb_as_dataframe():
    """
    Load IMDb dataset and return as pandas DataFrames.

    Splits the data into:
      - train:      80% of original train set
      - validation: 10% of original train set
      - test:       the original test set (held out)

    Returns:
        train_df, val_df, test_df: three DataFrames with columns 'text' and 'label'
    """
    print("Loading IMDb dataset...")
    dataset = load_dataset(DATASET_NAME, cache_dir=DATA_CACHE_DIR)

    # Convert HuggingFace dataset to pandas
    train_full = dataset['train'].to_pandas()
    test_df    = dataset['test'].to_pandas()

    # Shuffle training data (important for unbiased training)
    train_full = train_full.sample(frac=1, random_state=42).reset_index(drop=True)

    # Calculate split sizes
    n = len(train_full)
    n_train = int(n * TRAIN_SPLIT)  # 80% → training
    n_val   = int(n * VAL_SPLIT)    # 10% → validation

    # Split into train and validation
    train_df = train_full.iloc[:n_train].reset_index(drop=True)
    val_df   = train_full.iloc[n_train:n_train + n_val].reset_index(drop=True)

    print(f"Dataset loaded successfully:")
    print(f"  Train:      {len(train_df)} examples")
    print(f"  Validation: {len(val_df)} examples")
    print(f"  Test:       {len(test_df)} examples")

    return train_df, val_df, test_df
