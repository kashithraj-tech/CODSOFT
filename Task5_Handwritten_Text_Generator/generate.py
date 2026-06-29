import numpy as np
import tensorflow as tf
import pickle
import os

# -----------------------------
# Load Model & Tokenizer
# -----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))

model = tf.keras.models.load_model(
    os.path.join(current_dir, "model.keras")
)

char_to_index, index_to_char = pickle.load(
    open(os.path.join(current_dir, "tokenizer.pkl"), "rb")
)

characters      = list(char_to_index.keys())
SEQUENCE_LENGTH = 40   # must match train.py


# -----------------------------
# Text Generation
# -----------------------------
def generate_text(seed_text: str,
                  num_characters: int = 300,
                  temperature: float = 0.5) -> str:
    """
    Generate text character by character.

    temperature:
        0.2 – very repetitive / conservative
        0.5 – balanced (recommended)
        1.0 – more creative / random
        >1.0 – chaotic
    """
    generated = seed_text

    # Pad or trim seed to exactly SEQUENCE_LENGTH
    context = seed_text[-SEQUENCE_LENGTH:]
    while len(context) < SEQUENCE_LENGTH:
        context = " " + context

    for _ in range(num_characters):

        # Build integer-index input (matches Embedding layer – no one-hot needed)
        x = np.array(
            [[char_to_index.get(ch, 0) for ch in context]],
            dtype=np.int32
        )  # shape: (1, SEQUENCE_LENGTH)

        # Predict next character
        prediction = model.predict(x, verbose=0)[0]          # shape: (vocab,)

        # Apply temperature sampling
        prediction = np.log(prediction.astype("float64") + 1e-8) / temperature
        prediction = np.exp(prediction)
        prediction /= prediction.sum()

        next_index = np.random.choice(len(characters), p=prediction)
        next_char  = index_to_char[next_index]

        generated += next_char
        context    = context[1:] + next_char   # slide the window forward

    return generated


# -----------------------------
# Quick Test
# -----------------------------
if __name__ == "__main__":
    seed = "Artificial Intelligence"

    print("\n--- Generated Text (temperature=0.5) ---\n")
    print(generate_text(seed, num_characters=300, temperature=0.5))

    print("\n--- Generated Text (temperature=0.8) ---\n")
    print(generate_text(seed, num_characters=300, temperature=0.8))