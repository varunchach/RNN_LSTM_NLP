# ============================================================
# config/settings.py
# Central configuration for the entire project.
# All hyperparameters and paths are defined here so students
# can easily change settings without modifying model code.
# ============================================================

# --- Dataset Settings ---
DATASET_NAME = "imdb"          # HuggingFace dataset name
MAX_VOCAB_SIZE = 10000         # Keep only top 10,000 most frequent words
MAX_SEQ_LEN = 200              # Truncate/pad all reviews to this length
MIN_FREQ = 2                   # Ignore words appearing less than 2 times

# --- Split Ratios ---
TRAIN_SPLIT = 0.8              # 80% for training
VAL_SPLIT = 0.1                # 10% for validation
TEST_SPLIT = 0.1               # 10% for testing

# --- Model Hyperparameters ---
EMBEDDING_DIM = 64             # Size of word embedding vectors
HIDDEN_DIM = 128               # Hidden state size of RNN/LSTM/GRU
OUTPUT_DIM = 1                 # Binary classification: positive or negative
NUM_LAYERS = 1                 # Single layer: faster on CPU, converges better for 5 epochs
DROPOUT = 0.3                  # Light dropout — enough to regularize, not so much it kills learning

# --- Training Settings ---
BATCH_SIZE = 64                # Number of reviews per batch
LEARNING_RATE = 0.003          # Slightly higher LR helps escape the 0.5 plateau faster
NUM_EPOCHS = 5                 # Maximum training epochs (kept low for CPU speed)
PATIENCE = 3                   # Stop after 3 epochs with no improvement

# --- Special Tokens ---
PAD_TOKEN = "<PAD>"            # Padding token (fills short sequences)
UNK_TOKEN = "<UNK>"            # Unknown token (for words not in vocabulary)
PAD_IDX = 0                    # Index for PAD token
UNK_IDX = 1                    # Index for UNK token

# --- Paths ---
MODEL_SAVE_DIR = "saved_models"   # Where to save trained models
TENSORBOARD_LOG_DIR = "runs"      # TensorBoard log directory
DATA_CACHE_DIR = "data/cache"     # Where to cache downloaded data
