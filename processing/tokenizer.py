# ============================================================
# processing/tokenizer.py
# Tokenization: converting raw text into a list of words (tokens).
#
# Tokenization is the FIRST step in any NLP pipeline.
# It answers the question: "How do we split text into units?"
# ============================================================

import re


def simple_tokenize(text):
    """
    Tokenize text into a list of lowercase word tokens.

    Steps:
      1. Convert to lowercase  (so "Movie" == "movie")
      2. Remove HTML tags      (IMDb reviews sometimes contain <br /> tags)
      3. Keep only letters and spaces
      4. Split on whitespace

    Example:
      Input:  "The movie was GREAT! <br/> Loved it."
      Output: ['the', 'movie', 'was', 'great', 'loved', 'it']

    Args:
        text (str): Raw review text

    Returns:
        list[str]: List of word tokens
    """
    # Step 1: Lowercase the text
    text = text.lower()

    # Step 2: Remove HTML tags like <br />, <b>, etc.
    text = re.sub(r'<[^>]+>', ' ', text)

    # Step 3: Remove punctuation — keep only letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)

    # Step 4: Split on whitespace and remove empty tokens
    tokens = text.split()

    return tokens


def tokenize_batch(texts):
    """
    Tokenize a list of texts.

    Args:
        texts (list[str]): List of raw review strings

    Returns:
        list[list[str]]: List of token lists
    """
    return [simple_tokenize(text) for text in texts]
