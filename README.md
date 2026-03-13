# NLP Sentiment Analysis: RNN, LSTM & GRU with FastAPI

An end-to-end educational NLP project that teaches students how text becomes
a model prediction — from raw IMDb reviews to a deployed REST API.

---

## What You Will Learn

- How raw text is transformed into model-ready tensors
- How Word Embeddings (Word2Vec style) work
- How RNN, LSTM, and GRU sequence models differ
- How models are trained, monitored, and evaluated
- How to deploy a trained model as a REST API using FastAPI

---

## Time to Complete

| Phase | Estimated Time |
|---|---|
| Setup & Install | 5–10 minutes |
| Training (all 3 models) | 20–40 minutes (CPU) |
| API deployment | 2 minutes |
| Full exploration / notebook | 2–3 hours |

---

## Quick Start

```bash
# Step 1: Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Train all three models (RNN, LSTM, GRU)
python main.py

# Step 4: Start the API
uvicorn api.app:app --reload

# Step 5: Open the interactive docs
# http://localhost:8000/docs
```

---

## Project Structure & What Each File Does

```
sentiment_project/              <- Create this root folder
|
+-- config/
|   +-- settings.py            <- Central config: all hyperparameters in one place
|                                  (vocab size, embedding dim, hidden dim, epochs, etc.)
|
+-- data/
|   +-- download.py            <- Utility to download IMDb dataset from HuggingFace
|
+-- ingestion/
|   +-- loader.py              <- Loads raw dataset into train/val/test DataFrames
|
+-- processing/
|   +-- tokenizer.py           <- Converts raw text to a list of word tokens
|   |                              e.g. "Great movie!" -> ['great', 'movie']
|   +-- vocabulary.py          <- Builds word->index dictionary from training data
|   |                              e.g. {'great': 2, 'movie': 3, '<PAD>': 0, '<UNK>': 1}
|   +-- pipeline.py            <- Combines tokenize + encode + pad into a PyTorch Dataset
|                                  and wraps it in efficient DataLoaders
|
+-- embeddings/
|   +-- embedding_layer.py     <- Defines the embedding lookup table (word vectors)
|                                  Each word ID maps to a dense vector (dim=64)
|
+-- models/
|   +-- rnn_model.py           <- Vanilla RNN: Embedding -> RNN -> Linear
|   |                              Simple but suffers from vanishing gradients
|   +-- lstm_model.py          <- LSTM: Embedding -> LSTM -> Linear
|   |                              Uses 3 gates to control memory, handles long sequences
|   +-- gru_model.py           <- GRU: Embedding -> GRU -> Linear
|                                  Simpler than LSTM (2 gates), often matches LSTM accuracy
|
+-- training/
|   +-- trainer.py             <- train_one_epoch() and evaluate() functions
|   |                              Handles forward pass, loss, backward, optimizer step
|   +-- early_stopping.py      <- Stops training if val loss doesn't improve for N epochs
|                                  Saves best weights automatically
|
+-- evaluation/
|   +-- metrics.py             <- Computes accuracy, precision, recall, F1, confusion matrix
|   +-- plots.py               <- Plots training curves, confusion matrix, model comparison
|
+-- monitoring/
|   +-- tensorboard_logger.py  <- Logs metrics (loss, accuracy) to TensorBoard after each epoch
|                                  Run: tensorboard --logdir=runs  ->  http://localhost:6006
|
+-- inference/
|   +-- predictor.py           <- predict_sentiment(text, model, vocab, device)
|                                  Takes raw text -> returns {"sentiment": "positive", "confidence": 0.95}
|
+-- api/
|   +-- app.py                 <- FastAPI REST API
|                                  GET  /         -> health check
|                                  POST /predict  -> sentiment prediction
|
+-- notebooks/
|   +-- sentiment_analysis_tutorial.ipynb  <- Step-by-step teaching notebook
|                                              Covers all 16 stages with markdown explanations
|
+-- saved_models/              <- Auto-created during training
|   +-- rnn_model.pt           <- Trained RNN weights
|   +-- lstm_model.pt          <- Trained LSTM weights
|   +-- gru_model.pt           <- Trained GRU weights
|   +-- best_model.pt          <- Best model (used by API)
|   +-- vocab.pkl              <- Serialised vocabulary (used by API)
|
+-- runs/                      <- Auto-created during training (TensorBoard logs)
|   +-- RNN/
|   +-- LSTM/
|   +-- GRU/
|
+-- assets/                    <- Result images (plots saved after training)
|
+-- main.py                    <- TRAINING ENTRY POINT — run this to train everything
+-- requirements.txt           <- All Python dependencies
```

---

## Step-by-Step Flow

### Step 1 — Raw Data

```
Source: IMDb Movie Reviews (HuggingFace `datasets` library)
Size:   50,000 labeled reviews (positive / negative)
Split:  80% train | 10% validation | 10% test
```

`ingestion/loader.py` downloads and returns three DataFrames.

---

### Step 2 — Text Preprocessing Pipeline

Raw text goes through a 4-stage transformation before hitting the model:

```
"The movie was GREAT! <br/> Loved it."
         |
         v  tokenizer.py (lowercase, remove HTML/punctuation, split)
['the', 'movie', 'was', 'great', 'loved', 'it']
         |
         v  vocabulary.py (word -> index lookup)
[4, 102, 38, 57, 890, 9]
         |
         v  pipeline.py (truncate or pad to length 200)
[4, 102, 38, 57, 890, 9, 0, 0, 0, ... 0]   <- tensor of 200 integers
         |
         v  embeddings/embedding_layer.py (index -> dense vector)
[[0.12, -0.5, ...], [0.9, 0.1, ...], ...]   <- shape: (200, 64)
```

**Key concept:** Each word is an integer. The embedding layer turns each integer
into a 64-dimensional float vector. The model never sees raw text.

---

### Step 3 — Models

All three models share the same skeleton:

```
Input (batch of padded sequences)  ->  shape: (batch_size, 200)
         |
Embedding layer                    ->  shape: (batch_size, 200, 64)
         |
Recurrent layer (RNN / LSTM / GRU) ->  processes 200 timesteps, outputs hidden state
         |
Dropout (0.3)                      ->  regularization to prevent overfitting
         |
Linear(128 -> 1)                   ->  single score per review
         |
Sigmoid                            ->  probability 0-1
         |
> 0.5 = Positive,  <= 0.5 = Negative
```

#### RNN — `models/rnn_model.py`
- Simplest recurrent model
- Hidden state passed forward at each timestep
- **Problem**: gradients vanish over 200 steps — forgets early words
- **Result**: ~50% accuracy (barely above random)

#### LSTM — `models/lstm_model.py`
- Adds a **cell state** (long-term memory) alongside hidden state
- Three gates (forget / input / output) control what to remember and what to discard
- Solves the vanishing gradient problem
- **Result**: ~78% accuracy

#### GRU — `models/gru_model.py`
- Simpler version of LSTM with only 2 gates (reset / update)
- No separate cell state — hidden state carries everything
- Fewer parameters — trains faster, less prone to overfitting
- **Result**: ~86% accuracy (best in this project)

---

### Step 4 — Training

`main.py` orchestrates the full training loop for all three models:

```
For each model in [RNN, LSTM, GRU]:
    For each epoch (up to 5):
        1. Forward pass on training batches
        2. Compute BCEWithLogitsLoss
        3. Backward pass (compute gradients)
        4. Clip gradients (max norm = 1.0)   <- prevents RNN explosion
        5. Adam optimizer step
        6. Evaluate on validation set
        7. Log loss & accuracy to TensorBoard
        8. Check early stopping (patience = 3)
    Evaluate final performance on test set
    Save model weights to saved_models/
```

---

### Step 5 — Monitoring with TensorBoard

After (or during) training, run:

```bash
tensorboard --logdir=runs
```

Open `http://localhost:6006` to see:
- Training loss curve for RNN, LSTM, GRU side by side
- Validation loss and accuracy per epoch
- Confusion matrix images

---

### Step 6 — Evaluation Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| RNN | 50.2% | 0.47 | 0.46 | 0.46 |
| LSTM | 78.5% | 0.79 | 0.78 | 0.78 |
| **GRU** | **86.0%** | **0.86** | **0.86** | **0.86** |

**Why GRU won:** With only 5 training epochs on CPU, GRU's simpler architecture
(fewer parameters) converged faster and overfit less than LSTM.

---

### Step 7 — API Deployment

The best model (GRU) is loaded by the FastAPI app at startup.

```bash
uvicorn api.app:app --reload
```

#### Endpoint: `GET /`

Health check.

```json
{"message": "Sentiment Analysis API is running!", "status": "ok"}
```

#### Endpoint: `POST /predict`

**Request:**
```json
{"text": "This movie was absolutely fantastic!"}
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.97,
  "score": 0.97,
  "text": "This movie was absolutely fantastic!"
}
```

**Try it interactively:** `http://localhost:8000/docs`

**curl example:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"This movie was absolutely fantastic!\"}"
```

**Python example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Terrible acting, I fell asleep halfway through."}
)
print(response.json())
# {"sentiment": "negative", "confidence": 0.91, "score": 0.09, "text": "..."}
```

---

## Configuration Reference

All settings are in `config/settings.py`. Change them to experiment:

| Setting | Default | What It Controls |
|---|---|---|
| MAX_VOCAB_SIZE | 10,000 | How many unique words to keep |
| MAX_SEQ_LEN | 200 | Pad/truncate reviews to this many tokens |
| MIN_FREQ | 2 | Ignore words appearing fewer times |
| EMBEDDING_DIM | 64 | Size of each word vector |
| HIDDEN_DIM | 128 | Size of RNN/LSTM/GRU hidden state |
| BATCH_SIZE | 64 | Reviews per training step |
| LEARNING_RATE | 0.003 | Adam optimizer learning rate |
| NUM_EPOCHS | 5 | Maximum training epochs |
| PATIENCE | 3 | Early stopping — epochs without improvement |

---

## Technology Stack

| Tool | Purpose |
|---|---|
| Python 3.8+ | Programming language |
| PyTorch | Deep learning framework |
| HuggingFace datasets | IMDb dataset download |
| FastAPI | REST API framework |
| Uvicorn | ASGI server for FastAPI |
| TensorBoard | Training monitoring |
| pandas / numpy | Data manipulation |
| matplotlib / seaborn | Visualization |
| scikit-learn | Evaluation metrics |

---

## Key Concepts Explained

**Tokenization** — Breaking text into individual words (tokens).
"Hello world" becomes ["hello", "world"].

**Vocabulary** — A lookup table mapping each unique word to an integer ID.
The model only understands numbers, not words.

**Padding** — Reviews have different lengths. We fix all sequences to 200 tokens
by adding zeros at the end (shorter reviews) or cutting off (longer reviews).
This allows batching multiple reviews together.

**Embedding** — A learned table where each word ID maps to a vector of floats.
Similar words end up with similar vectors. The model learns these during training.
This is similar to Word2Vec.

**Hidden State** — The "memory" of a recurrent model. After processing each word,
the model updates its hidden state — carrying forward what it has learned so far.

**Early Stopping** — Automatically stop training when the model stops improving
on validation data. Prevents overfitting (memorizing training data instead of generalizing).

**Vanishing Gradient** — In plain RNNs, gradients shrink to near-zero as they
flow backward through 200 timesteps. LSTM and GRU solve this with gating mechanisms.

---

## Folder Creation Checklist (Build From Scratch)

```bash
mkdir sentiment_project
cd sentiment_project
mkdir config data ingestion processing embeddings
mkdir models training evaluation monitoring inference api notebooks
mkdir saved_models runs assets
```

Then create files in this order:

1. `requirements.txt`
2. `config/settings.py`
3. `ingestion/loader.py`
4. `processing/tokenizer.py`
5. `processing/vocabulary.py`
6. `processing/pipeline.py`
7. `embeddings/embedding_layer.py`
8. `models/rnn_model.py`, `models/lstm_model.py`, `models/gru_model.py`
9. `training/trainer.py`, `training/early_stopping.py`
10. `monitoring/tensorboard_logger.py`
11. `evaluation/metrics.py`, `evaluation/plots.py`
12. `main.py`
13. `inference/predictor.py`
14. `api/app.py`
15. `notebooks/sentiment_analysis_tutorial.ipynb`

---

## Visual Results

![Training Curves](assets/training_curves.png)
*Loss and accuracy curves for RNN, LSTM, and GRU across training epochs*

![GRU Confusion Matrix](assets/gru_confusion_matrix.png)
*Confusion matrix for the best model (GRU) on the test set*

![Model Comparison](assets/model_comparison.png)
*Side-by-side bar chart comparing all three models across evaluation metrics*

---

## License

Educational use. Dataset: IMDb via HuggingFace (see their license).
