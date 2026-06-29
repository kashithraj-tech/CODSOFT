"""
model/train_word.py  –  Word-level model (optional / secondary model)
Run from project root:  py -3.12 model/train_word.py
"""

import numpy as np
import tensorflow as tf
import pickle
import os
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ================================================
# SETTINGS
# ================================================
DATA_LIMIT      = 50000   # was 500,000 — huge reduction
SEQUENCE_LENGTH = 10      # was 20
STEP            = 2       # skip every 2 sequences
EMBEDDING_DIM   = 64      # was 128
GRU_UNITS       = 128     # was 2×LSTM(256)
BATCH_SIZE      = 128     # was 64
EPOCHS          = 30      # was 100; EarlyStopping exits earlier
CHECKPOINT_PATH = "checkpoints/word_weights.{epoch:02d}.keras"
# ================================================

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("outputs",     exist_ok=True)

# ── Load Dataset ──────────────────────────────────────────────────────────────
print("Loading dataset...")
text = open("dataset/handwriting.txt", encoding="utf-8").read()[:DATA_LIMIT]
print(f"Using {len(text):,} characters.")

# ── Tokenizer ─────────────────────────────────────────────────────────────────
print("Building tokenizer...")
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1
print(f"Vocabulary: {total_words:,} words")

pickle.dump(tokenizer, open("model/tokenizer_word.pkl", "wb"))

# ── Sequences ─────────────────────────────────────────────────────────────────
print("Preparing sequences...")
sequence = tokenizer.texts_to_sequences([text])[0]

X, Y = [], []
for i in range(SEQUENCE_LENGTH, len(sequence), STEP):
    X.append(sequence[i - SEQUENCE_LENGTH:i])
    Y.append(sequence[i])

X = np.array(X, dtype=np.int32)
Y = tf.keras.utils.to_categorical(Y, num_classes=total_words)
print(f"Samples: {len(X):,}  |  X: {X.shape}  |  Y: {Y.shape}")

# ── Model ─────────────────────────────────────────────────────────────────────
print("Building model...")
model = Sequential([
    Embedding(input_dim=total_words, output_dim=EMBEDDING_DIM,
              input_length=SEQUENCE_LENGTH),
    GRU(GRU_UNITS, return_sequences=False),
    Dropout(0.2),
    Dense(total_words, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
    ModelCheckpoint(CHECKPOINT_PATH, save_best_only=False, verbose=0),
    EarlyStopping(monitor="loss", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3, verbose=1),
]

# Resume from checkpoint if available
latest = tf.train.latest_checkpoint("checkpoints")
if latest and "word" in latest:
    print(f"Resuming from: {latest}")
    model.load_weights(latest)

# ── Train ─────────────────────────────────────────────────────────────────────
print("Training...")
history = model.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks)

model.save("model/model_word.keras")
print("Saved → model/model_word.keras")

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"],     label="Loss",     color="royalblue")
plt.plot(history.history["accuracy"], label="Accuracy", color="seagreen")
plt.title("Word-Level GRU Training")
plt.xlabel("Epoch"); plt.legend(); plt.tight_layout()
plt.savefig("outputs/loss_word.png")
print("Plot → outputs/loss_word.png")
plt.show()