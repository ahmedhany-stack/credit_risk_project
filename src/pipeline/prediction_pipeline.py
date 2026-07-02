import pandas as pd
import joblib
import warnings

from src.components.logger import logger


class PredictionPipeline:

    def __init__(self, model_path, preprocessor_path):

        # 🔥 load artifacts
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

        # 🔥 full schema (must match training exactly)
        self.expected_features = [
            'age', 'gender', 'employment_status', 'annual_income',
            'account_age_months', 'avg_monthly_balance',
            'num_deposits_per_month', 'avg_deposit_amount',
            'debit_card_usage_frequency', 'debit_card_spending',
            'mobile_banking_logins', 'online_transfer_frequency',
            'atm_withdrawal_frequency', 'credit_score',
            'num_open_loans', 'total_outstanding_debt',
            'late_payment_count', 'loan_default_history',
            'fraud_flag', 'loan_application_amount'
        ]

    def predict(self, input_data: dict):

        try:
            logger.info("Prediction started")

            # =========================
            # dict -> DataFrame
            # =========================
            df = pd.DataFrame([input_data])

            # =========================
            # FIX 1: add missing columns safely
            # =========================
            for col in self.expected_features:
                if col not in df.columns:
                    df[col] = 0

            # =========================
            # FIX 2: enforce exact order (VERY IMPORTANT)
            # =========================
            df = df[self.expected_features]

            # =========================
            # optional: silence sklearn warning (clean logs)
            # =========================
            warnings.filterwarnings("ignore", category=UserWarning)

            logger.info("Input prepared successfully")

            # =========================
            # preprocessing
            # =========================
            processed = self.preprocessor.transform(df)

            # =========================
            # prediction
            # =========================
            prediction = self.model.predict(processed)

            result = {
                "prediction": prediction.tolist()
            }

            # =========================
            # probability (if available)
            # =========================
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(processed)
                result["probability"] = proba.tolist()

            logger.info("Prediction completed successfully")

            return result

        except Exception as e:
            logger.exception("Prediction failed")
            raise