# 🎬 Movie Genre Classification using NLP (TF-IDF)

An interactive Natural Language Processing (NLP) pipeline that analyzes text features (movie titles/vocabulary patterns) to predict the genre of a film. This project transforms textual information using **TF-IDF Vectorization** and runs it through a finely tuned, class-balanced **Logistic Regression Classifier** to achieve high classification accuracy.

---

## 📌 Project Overview & Specifications

This project satisfies all four core pipeline specifications for advanced text classification:
1. **Text Vectorization Layer:** Implements `TfidfVectorizer` to extract statistical word weights while filtering out standard English stop words.
2. **Text Classification Engine:** Employs a robust `LogisticRegression` model with balanced class weight adjustments to eliminate majority class bias.
3. **Performance Tracking:** Evaluates metrics using test split subsets, returning comprehensive Accuracy and Classification Reports (Precision, Recall, F1-Score).
4. **Interactive Prediction Interface:** Features a live user loop capable of categorizing input strings on demand.

### 🚀 Key Enhancements Implemented
* **Sublinear Term Frequency (TF) Scaling:** Smooths out dominant word frequencies logarithmically so specific recurring phrases don't drown out rare, highly predictive vocabulary tokens.
* **Algorithmic Class Balancing:** Configured with `class_weight='balanced'` to prevent majority class bias (such as defaulting every input to "Comedy") when working on small or skewed datasets.
* **Advanced Feature Importance Visualization:** Renders a sorted, horizontal Seaborn bar dashboard tracking relative vocabulary weights.

---

## 🛠️ Requirements & Installation

Before running the project, ensure you have Python installed along with the required libraries.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn