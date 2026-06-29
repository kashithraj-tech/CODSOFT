"""
model/generate_word.py  –  Word-level text generation
"""

import numpy as np
import tensorflow as tf
import pickle
import os

# ── Load ──────────────────────────────────────────────────────────────────────
_base = os.path.dirname(os.path.abspath(__file__))

model = tf.keras.models.load_model(
    os.path.join(_base, "model_word.keras")
)

tokenizer = pickle.load(
    open(os.path.join(_base, "tokenizer_word.pkl"), "rb")
)

SEQUENCE_LENGTH = 10   # must match train_word.py


def generate_text(seed_text: str,
                  next_words: int = 80,
                  temperature: float = 0.7) -> str:
    """Generate next_words words after seed_text."""
    result = seed_text

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([result])[0]
        token_list = token_list[-SEQUENCE_LENGTH:]
        token_list = tf.keras.preprocessing.sequence.pad_sequences(
            [token_list], maxlen=SEQUENCE_LENGTH, padding="pre"
        )

        prediction = model.predict(token_list, verbose=0)[0].astype("float64")

        # Temperature sampling
        prediction = np.log(prediction + 1e-8) / temperature
        prediction = np.exp(prediction)
        prediction /= prediction.sum()

        predicted_idx = np.random.choice(len(prediction), p=prediction)

        # Index → word
        output_word = ""
        for word, idx in tokenizer.word_index.items():
            if idx == predicted_idx:
                output_word = word
                break

        result += " " + output_word

    return result


if __name__ == "__main__":
    print(generate_text("Artificial Intelligence", next_words=50, temperature=0.7))