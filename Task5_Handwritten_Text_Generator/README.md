# ✍️ Handwritten Text Generator (Character-Level RNN)

An interactive, offline character-level Recurrent Neural Network (RNN) designed to learn and mimic human handwriting textual patterns. Built using **TensorFlow/Keras** and served through a lightweight **Streamlit** web interface, this model generates text character-by-character based on a seed sentence.

---

## 🚀 Features

* **Deep Character-Level GRU Network:** Utilizes a stacked Gated Recurrent Unit (GRU) architecture to learn local syllables, punctuation habits, and word structures.
* **Dense Word Embeddings:** Swaps heavy one-hot encoding layouts for dense vector embedding layers to minimize RAM and storage requirements.
* **Temperature-Based Sampling:** Allows users to adjust generation variance (lower temperature for realistic text, higher for creative outputs).
* **Streamlit Interactive UI:** A clean interface complete with sliding adjustments for context configuration.
* **100% Offline-Capable:** No mobile data or internet bandwidth is consumed during training or inference once packages are installed locally.

---

## 📁 Project Structure

```text
├── dataset/
│   └── handwriting.txt       # Input raw text corpus
├── checkpoints/              # Auto-generated model training weights per epoch
├── outputs/                  # Holds loss/accuracy progress tracking charts
├── ui/
│   └── app.py                # Main Streamlit web application interface
├── train.py                  # Character-level training script (Upgraded Architecture)
├── train_word.py             # Optional secondary word-level training script
├── generate.py               # Standalone character text generation snippet
├── generate_word.py          # Standalone word text generation snippet
├── tokenizer.pkl             # Character mapping binaries (Auto-generated)
└── model.keras               # Fully trained neural network file (Auto-generated)
