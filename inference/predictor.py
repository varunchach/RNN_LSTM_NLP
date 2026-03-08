# ============================================================
# inference/predictor.py
# Inference pipeline: predict sentiment for new text.
#
# This module takes a raw text string and returns a prediction.
# It applies the same preprocessing used during training.
# ============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from processing.tokenizer import simple_tokenize
from config.settings import MAX_SEQ_LEN, PAD_IDX


def predict_sentiment(text, model, vocab, device):
    """
    Predict sentiment of a single text string.

    Pipeline:
      raw text -> tokenize -> encode -> pad -> tensor -> model -> prediction

    Args:
        text (str):    Raw review text
        model:         Trained PyTorch model
        vocab:         Vocabulary object (from training)
        device (str):  'cuda' or 'cpu'

    Returns:
        dict: {
            'sentiment': 'positive' or 'negative',
            'confidence': float (0.0 to 1.0),
            'score': raw probability
        }
    """
    model.eval()

    # Step 1: Tokenize
    tokens = simple_tokenize(text)

    # Step 2: Encode to IDs
    encoded = vocab.encode(tokens)

    # Step 3: Truncate to max length
    encoded = encoded[:MAX_SEQ_LEN]

    # Step 4: Pad to max length
    padding = [PAD_IDX] * (MAX_SEQ_LEN - len(encoded))
    encoded = encoded + padding

    # Step 5: Convert to tensor with batch dimension
    text_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)

    # Step 6: Run model inference
    with torch.no_grad():
        output = model(text_tensor)
        score  = torch.sigmoid(output).item()

    # Step 7: Interpret the score
    sentiment  = "positive" if score >= 0.5 else "negative"
    confidence = score if score >= 0.5 else 1 - score

    return {
        "sentiment":  sentiment,
        "confidence": round(confidence, 4),
        "score":      round(score, 4)
    }
