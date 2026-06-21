# Intelligent Credit Card Fraud Detection System

## Project Overview

The Intelligent Credit Card Fraud Detection System is a machine learning project designed to identify fraudulent credit card transactions. The system analyzes transaction details and predicts whether a transaction is legitimate or fraudulent using classification algorithms.

The project also includes a Streamlit-based web interface that allows users to enter transaction information and receive instant fraud risk predictions.

---

## Objectives

* Detect fraudulent credit card transactions.
* Compare multiple machine learning algorithms.
* Improve fraud detection accuracy using feature engineering.
* Provide a user-friendly interface for prediction.
* Generate fraud risk scores for decision-making.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit

---

## Dataset

The project uses a credit card transaction dataset containing:

* Customer Information
* Merchant Information
* Transaction Amount
* Geographic Location
* Transaction Time
* Fraud Labels

Files Used:

* fraudTrain.csv
* fraudTest.csv

---

## Feature Engineering

A new feature called **distance** was created using customer and merchant coordinates:

Distance = √((lat - merch_lat)² + (long - merch_long)²)

This helps improve fraud detection performance.

---

## Machine Learning Models

### 1. Logistic Regression

A baseline classification model used for fraud prediction.

### 2. Decision Tree

A tree-based model used to identify fraud patterns.

### 3. Random Forest

An ensemble learning model that combines multiple decision trees and provides the best performance.

---

## Model Evaluation Metrics

The following metrics were used:

* Accuracy Score
* Classification Report
* Confusion Matrix
* ROC-AUC Score
* Fraud Risk Score

---

## Best Model

Random Forest achieved the highest performance and was selected as the final model.

The trained model is saved as:

fraud_detection_model.pkl

---

## Streamlit User Interface

The project includes a Streamlit-based web application where users can:

* Enter transaction details
* Predict fraud risk
* View fraud probability
* View risk category (Low, Medium, High)

---

## Project Structure

Task2_Credit_Card_Fraud/

├── fraud_detection.py

├── fraud_detection_model.pkl

├── fraudTrain.csv

├── fraudTest.csv

├── README.md

└── ui/

    └── app.py

---

## How to Run

### Train Model

Run:

python fraud_detection.py

### Launch Streamlit UI

Open terminal inside ui folder:

cd ui

python -m streamlit run app.py

---

## Output

The system predicts:

* Legitimate Transaction
* Fraudulent Transaction

It also provides:

* Fraud Probability
* Risk Score
* Risk Category

---

## Conclusion

This project successfully detects fraudulent credit card transactions using machine learning techniques. Random Forest provided the best accuracy among the tested models. The Streamlit interface enables easy interaction and real-time fraud prediction.
