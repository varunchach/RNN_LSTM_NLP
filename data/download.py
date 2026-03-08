# ============================================================
# data/download.py
# Downloads and caches the IMDb dataset using HuggingFace.
# This keeps data loading separate from model code.
# ============================================================

from datasets import load_dataset
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DATASET_NAME, DATA_CACHE_DIR


def download_imdb():
    """
    Download the IMDb dataset from HuggingFace.

    The IMDb dataset contains 50,000 movie reviews:
    - 25,000 for training
    - 25,000 for testing
    Each review is labeled as positive (1) or negative (0).

    Returns:
        dataset: HuggingFace DatasetDict with 'train' and 'test' splits
    """
    print(f"Downloading {DATASET_NAME} dataset...")
    print("This may take a moment on first run (dataset will be cached).")

    # Load dataset - HuggingFace caches it automatically
    dataset = load_dataset(DATASET_NAME, cache_dir=DATA_CACHE_DIR)

    print(f"Download complete!")
    print(f"  Training examples: {len(dataset['train'])}")
    print(f"  Test examples:     {len(dataset['test'])}")

    return dataset


if __name__ == "__main__":
    # Run this file directly to test the download
    dataset = download_imdb()
    print("\nSample review:")
    print(dataset['train'][0])
