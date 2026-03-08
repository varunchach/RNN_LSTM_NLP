# ============================================================
# api/app.py
# FastAPI application for sentiment inference.
#
# This exposes a REST API endpoint:
#   POST /predict
#   Body: {"text": "This movie was amazing!"}
#   Response: {"sentiment": "positive", "confidence": 0.95}
#
# Run with: uvicorn api.app:app --reload
# ============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from inference.predictor import predict_sentiment
from config.settings import MODEL_SAVE_DIR

# --- Global variables to hold the loaded model ---
model  = None
vocab  = None
device = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and vocabulary when the API starts up."""
    global model, vocab, device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = os.path.join(MODEL_SAVE_DIR, "best_model.pt")
    vocab_path = os.path.join(MODEL_SAVE_DIR, "vocab.pkl")

    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found at {model_path}. Train first!")
    if not os.path.exists(vocab_path):
        raise RuntimeError(f"Vocab not found at {vocab_path}. Train first!")

    print("Loading model and vocabulary...")
    # weights_only=False needed because we saved the full model object (not just state_dict)
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    print(f"Model loaded. Using device: {device}")
    yield
    # Cleanup on shutdown (optional)
    print("API shutting down.")


# Create the FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="Predict sentiment of movie reviews using RNN/LSTM/GRU",
    version="1.0.0",
    lifespan=lifespan
)


# --- Request and Response schemas ---

class PredictRequest(BaseModel):
    """Input schema: requires a 'text' field."""
    text: str


class PredictResponse(BaseModel):
    """Output schema: sentiment, confidence, raw score."""
    sentiment:  str
    confidence: float
    score:      float
    text:       str


# --- Endpoints ---

@app.get("/")
def root():
    """Health check endpoint."""
    return {"message": "Sentiment Analysis API is running!", "status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Predict sentiment of the given text.

    Example:
        POST /predict
        {"text": "This movie was absolutely fantastic!"}

    Returns:
        {"sentiment": "positive", "confidence": 0.95, "score": 0.95, "text": "..."}
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    result = predict_sentiment(request.text, model, vocab, device)
    result['text'] = request.text

    return PredictResponse(**result)
