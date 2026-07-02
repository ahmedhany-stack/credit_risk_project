from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predict customer credit risk using Machine Learning",
    version="1.0.0"
)

pipeline = PredictionPipeline(
    model_path="artifacts/random_forest.pkl",
    preprocessor_path="artifacts/preprocessor.pkl"
)


class InputData(BaseModel):
    age: float
    gender: str
    employment_status: str
    annual_income: float
    account_age_months: float
    avg_monthly_balance: float
    num_deposits_per_month: float
    avg_deposit_amount: float
    debit_card_usage_frequency: float
    debit_card_spending: float
    mobile_banking_logins: float
    online_transfer_frequency: float
    atm_withdrawal_frequency: float
    credit_score: float
    num_open_loans: float
    total_outstanding_debt: float
    late_payment_count: float
    loan_default_history: float
    fraud_flag: float
    loan_application_amount: float


@app.get("/")
def root():
    return {
        "message": "Credit Risk Prediction API 🚀",
        "version": "1.0.0",
        "status": "Running"
    }


@app.get("/health")
def health():
    return {
        "status": "Healthy"
    }


@app.post("/predict")
def predict(data: InputData):

    try:

        result = pipeline.predict(data.model_dump())

        prediction = result["prediction"][0]

        probability = result["probability"][0]

        confidence = max(probability) * 100

        return {
            "success": True,
            "prediction": prediction,
            "risk": "High Risk" if prediction == 1 else "Low Risk",
            "confidence": round(confidence, 2),
            "probabilities": {
                "Low Risk": round(probability[0] * 100, 2),
                "High Risk": round(probability[1] * 100, 2)
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )