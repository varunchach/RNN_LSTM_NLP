# ============================================================
# processing/vocabulary.py
# Vocabulary: mapping words to integer IDs.
#
# Models cannot understand words — they need numbers.
# The vocabulary is a dictionary: word -> integer index.
#
# Special tokens:
#   <PAD> = 0  -> used to fill short sequences to equal length
#   <UNK> = 1  -> used for words not seen during training
# ============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter
from config.settings import MAX_VOCAB_SIZE, MIN_FREQ, PAD_TOKEN, UNK_TOKEN, PAD_IDX, UNK_IDX


class Vocabulary:
    """
    Builds and manages a word-to-index mapping.

    Usage:
        vocab = Vocabulary()
        vocab.build(all_token_lists)
        idx = vocab.word_to_idx["good"]
        word = vocab.idx_to_word[idx]
    """

    def __init__(self):
        # word -> index dictionary
        self.word_to_idx = {}
        # index -> word dictionary (for reverse lookup)
        self.idx_to_word = {}
        # Track vocabulary size
        self.vocab_size = 0

    def build(self, token_lists):
        """
        Build vocabulary from a list of token lists.

        Only keeps the top MAX_VOCAB_SIZE most frequent words
        that appear at least MIN_FREQ times.

        Args:
            token_lists (list[list[str]]): All tokenized training documents
        """
        # Count word frequencies across all documents
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)

        # Start with special tokens at fixed indices
        self.word_to_idx = {
            PAD_TOKEN: PAD_IDX,   # 0 -> padding
            UNK_TOKEN: UNK_IDX,   # 1 -> unknown words
        }

        # Add most common words (filtered by minimum frequency)
        common_words = [
            word for word, freq in counter.most_common(MAX_VOCAB_SIZE)
            if freq >= MIN_FREQ
        ]

        for word in common_words:
            if word not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx

        # Build reverse mapping (index -> word)
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)

        print(f"Vocabulary built: {self.vocab_size} words")

    def encode(self, tokens):
        """
        Convert a list of tokens to a list of integer IDs.

        Words not in vocabulary are replaced with UNK_IDX.

        Args:
            tokens (list[str]): List of word tokens

        Returns:
            list[int]: List of integer IDs
        """
        return [
            self.word_to_idx.get(token, UNK_IDX)
            for token in tokens
        ]

    def decode(self, indices):
        """
        Convert a list of integer IDs back to words.

        Args:
            indices (list[int]): List of integer IDs

        Returns:
            list[str]: List of words
        """
        return [self.idx_to_word.get(idx, UNK_TOKEN) for idx in indices]
