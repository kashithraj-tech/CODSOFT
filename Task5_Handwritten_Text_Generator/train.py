import numpy as np
import tensorflow as tf
import pickle
import os
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Embedding
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ================================================
# SETTINGS  ← tweak these to go faster/slower
# ================================================
DATA_LIMIT       = 30000   # characters to read (was 100,000) – saves RAM & time
SEQUENCE_LENGTH  = 40      # context window (was 80) – halves input size
STEP             = 3       # skip every N chars when building sequences
                           # (was 1 – generates ~3× fewer training samples)
EMBEDDING_DIM    = 32      # small embedding instead of one-hot (saves huge RAM)
GRU_UNITS        = 128     # was 2× LSTM(256) – much lighter
DROPOUT          = 0.2
BATCH_SIZE       = 128     # larger batch = fewer steps per epoch = faster
EPOCHS           = 30      # was 100; EarlyStopping will quit even sooner
CHECKPOINT_PATH  = "checkpoints/weights.{epoch:02d}.keras"
# ================================================

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("outputs",     exist_ok=True)

# -----------------------------
# 1. Load Dataset
# -----------------------------
print("Loading dataset...")
text = open("dataset/handwriting.txt", encoding="utf-8").read()
text = text[:DATA_LIMIT]
print(f"Using {len(text):,} characters.")

# -----------------------------
# 2. Character Mapping
# -----------------------------
characters    = sorted(set(text))
vocab_size    = len(characters)
char_to_index = {c: i for i, c in enumerate(characters)}
index_to_char = {i: c for i, c in enumerate(characters)}

pickle.dump(
    (char_to_index, index_to_char),
    open("tokenizer.pkl", "wb")
)
print(f"Vocab size: {vocab_size} unique characters.")

# -----------------------------
# 3. Sequence Preparation
#    STEP > 1 drastically cuts the number of samples
# -----------------------------
print("Preparing sequences (step={})...".format(STEP))
X_seqs, Y_labels = [], []

for i in range(0, len(text) - SEQUENCE_LENGTH, STEP):
    seq    = text[i : i + SEQUENCE_LENGTH]
    target = text[i + SEQUENCE_LENGTH]
    X_seqs.append([char_to_index[c] for c in seq])
    Y_labels.append(char_to_index[target])

X = np.array(X_seqs,  dtype=np.int32)          # integer indices – tiny vs one-hot
Y = tf.keras.utils.to_categorical(Y_labels, num_classes=vocab_size)

print(f"Samples: {len(X):,}  |  X shape: {X.shape}  |  Y shape: {Y.shape}")

# -----------------------------
# 4. Build Lightweight Model
#    Embedding + single GRU replaces 2× LSTM + one-hot
# -----------------------------
print("Building model...")

model = Sequential([
    # Embedding turns integer indices into dense vectors – no giant one-hot matrix
    Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM,
              input_length=SEQUENCE_LENGTH),

    GRU(GRU_UNITS, return_sequences=False),   # GRU trains ~30 % faster than LSTM
    Dropout(DROPOUT),

    Dense(vocab_size, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# 5. Callbacks
# -----------------------------
callbacks = [
    # Save a checkpoint every epoch – if training crashes, resume from last one
    ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        save_best_only=False,
        verbose=0
    ),
    # Stop early if val_loss stops improving for 5 epochs
    EarlyStopping(
        monitor="loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    # Halve LR if loss plateaus for 3 epochs
    ReduceLROnPlateau(
        monitor="loss",
        factor=0.5,
        patience=3,
        verbose=1
    )
]

# -----------------------------
# 6. Resume from Checkpoint (optional)
# -----------------------------
latest = tf.train.latest_checkpoint("checkpoints")
if latest:
    print(f"Resuming from checkpoint: {latest}")
    model.load_weights(latest)
else:
    print("Starting fresh training.")

# -----------------------------
# 7. Train
# -----------------------------
print("Training started...")
history = model.fit(
    X, Y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

model.save("model.keras")
print("Model saved → model.keras")

# -----------------------------
# 8. Plot Loss
# -----------------------------
plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="Loss", color="royalblue")
plt.plot(history.history["accuracy"], label="Accuracy", color="seagreen")
plt.title("Training Progress")
plt.xlabel("Epoch")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/loss.png")
print("Plot saved → outputs/loss.png")
plt.show()