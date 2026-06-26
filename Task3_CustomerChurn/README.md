# 🏦 Customer Churn Prediction System

## CodSoft Internship - Task 3

A Machine Learning based Customer Churn Prediction System developed using Python, Scikit-learn and Flask.

---

## 📌 Project Objective

The objective of this project is to predict whether a customer will leave (churn) or stay with the bank using historical customer information.

The system helps businesses identify customers at high risk of leaving so that retention strategies can be applied.

---

## 🚀 Features

- Customer Churn Prediction
- Logistic Regression Model
- Random Forest Model
- Automatic Best Model Selection
- Feature Scaling
- Data Preprocessing
- Label Encoding
- Feature Importance Graph
- Model Comparison Graph
- Confusion Matrix Visualization
- Churn Probability Prediction
- Risk Level Classification
- Customer Retention Recommendation
- Prediction History Storage
- Professional Flask Web Interface

---

## 🛠 Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- HTML
- CSS

---

## 📂 Dataset

Dataset Used:

Churn_Modelling.csv

Dataset contains customer information such as:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Credit Card Status
- Active Member Status
- Estimated Salary
- Exited (Target)

---

## 📊 Machine Learning Models

- Logistic Regression
- Random Forest Classifier

The project automatically selects the model with the highest accuracy.

---

## 📈 Performance Metrics

The project evaluates the model using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Classification Report
- Confusion Matrix

---

## 📷 Output Visualizations

- Feature Importance Graph
- Model Accuracy Comparison
- Confusion Matrix

---

## 💻 User Interface

The Flask dashboard allows users to:

- Enter customer details
- Predict customer churn
- View churn probability
- View risk level
- Receive retention recommendations

---

## ▶️ How to Run

Clone the repository

```bash
git clone https://github.com/yourusername/Task3_CustomerChurn.git
```

Move into the project folder

```bash
cd Task3_CustomerChurn
```

Install dependencies

```bash
pip install pandas numpy scikit-learn flask matplotlib joblib
```

Train the model

```bash
python train_model.py
```

Run the Flask application

```bash
python app.py
```

Open your browser

```
http://127.0.0.1:5000
```

---

## 📁 Project Structure

```
Task3_CustomerChurn
│
├── app.py
├── train_model.py
├── Churn_Modelling.csv
├── churn_model.pkl
├── scaler.pkl
├── geo_encoder.pkl
├── gender_encoder.pkl
│
├── templates
│   └── index.html
│
├── static
│   ├── style.css
│   ├── feature_importance.png
│   ├── model_comparison.png
│   └── confusion_matrix.png
│
└── README.md
```

---

## 🎯 Future Improvements

- XGBoost Integration
- LightGBM Integration
- SHAP Explainability
- Interactive Dashboard
- PDF Prediction Reports
- Cloud Deployment

---

## 👩‍💻 Author

**Kashithra Janarthanam**

CodSoft Machine Learning Internship

---
