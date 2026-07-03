# Loan Project
# Credit Risk Prediction

A production-ready Machine Learning project for predicting whether a loan applicant is likely to become a credit risk based on customer financial and behavioral information.

The project follows a modular architecture inspired by real-world Machine Learning systems, including data validation, preprocessing, model training, evaluation, model persistence, logging, and prediction pipelines.

---

# Features

* Data Loading
* Data Validation
* Outlier Handling (IQR Capping)
* Feature Engineering & Preprocessing
* One-Hot Encoding
* Feature Scaling
* Train/Test Split
* Random Forest Classifier
* Model Evaluation
* Model Persistence
* Prediction Pipeline
* Custom Exception Handling
* Logging System
* Modular Project Structure
* Production-Oriented Codebase

---

# Project Structure

```text
Credit_Risk_Project/
│
├── artifacts/
│   ├── random_forest.pkl
│   └── preprocessor.pkl
│
├── logs/
│
├── notebooks/
│
├── src/
│
│   ├── components/
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   ├── model_pusher.py
│   │   ├── prediction_pipeline.py
│   │   ├── logger.py
│   │   └── constants.py
│   │
│   ├── pipeline/
│   │   └── training_pipeline.py
│   │
│   ├── config/
│   │   └── configuration.py
│   │
│   ├── utils.py
│   └── exception.py
│
├── requirements.txt
├── Dockerfile
├── README.md
└── app.py
```

---

# Dataset

The dataset contains customer financial information such as:

* Age
* Gender
* Employment Status
* Annual Income
* Account Age
* Average Monthly Balance
* Debit Card Usage
* Online Banking Activity
* Credit Score
* Outstanding Debt
* Previous Loan Defaults
* Fraud Flag
* Loan Application Amount

Target:

```text
credit_risk
```

* 0 → Low Risk
* 1 → High Risk

---

# Machine Learning Pipeline

```text
Dataset
      │
      ▼
Data Validation
      │
      ▼
Outlier Treatment
      │
      ▼
Feature Encoding
      │
      ▼
Feature Scaling
      │
      ▼
Train/Test Split
      │
      ▼
Random Forest Training
      │
      ▼
Model Evaluation
      │
      ▼
Model Saving
      │
      ▼
Prediction Pipeline
```

---

# Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Logging
* Object-Oriented Programming (OOP)

---

# Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix
* Classification Report

---

# Error Handling

The project includes a custom exception system that provides:

* File Name
* Line Number
* Exception Type
* Error Message

This makes debugging easier and improves maintainability.

---

# Logging

A centralized logging system records:

* Data loading
* Data preprocessing
* Model training
* Evaluation
* Prediction
* Errors

---

# Saved Artifacts

After training, the following files are generated:

```text
artifacts/
│
├── random_forest.pkl
└── preprocessor.pkl
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/credit-risk-prediction.git
```

Move into the project directory:

```bash
cd credit-risk-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Train the Model

```bash
python -m src.pipeline.training_pipeline
```

---

# Make Predictions

```python
from src.components.prediction_pipeline import PredictionPipeline

pipeline = PredictionPipeline(
    model_path="artifacts/random_forest.pkl",
    preprocessor_path="artifacts/preprocessor.pkl"
)

sample = {
    "age": 30,
    "gender": "Male",
    "employment_status": "Employed",
    "annual_income": 65000,
    "account_age_months": 24,
    "avg_monthly_balance": 4500,
    "num_deposits_per_month": 8,
    "avg_deposit_amount": 700,
    "debit_card_usage_frequency": 20,
    "debit_card_spending": 1200,
    "mobile_banking_logins": 35,
    "online_transfer_frequency": 6,
    "atm_withdrawal_frequency": 3,
    "credit_score": 720,
    "num_open_loans": 1,
    "total_outstanding_debt": 15000,
    "late_payment_count": 0,
    "loan_default_history": 0,
    "fraud_flag": 0,
    "loan_application_amount": 25000
}

result = pipeline.predict(sample)

print(result)
```

---

# Future Improvements

* Hyperparameter Tuning
* Cross Validation
* Feature Selection
* Experiment Tracking (MLflow)
* FastAPI Deployment
* Docker
* CI/CD
* Kubernetes Deployment
* Cloud Deployment (AWS/Azure/GCP)

---

# Author

Ahmed Hany

Faculty of Artificial Intelligence

Machine Learning Engineer (Aspiring)
