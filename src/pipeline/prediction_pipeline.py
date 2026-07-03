import sys
import warnings
import joblib
import pandas as pd

from src.components.logger import logger
from src.exception import CustomException


class PredictionPipeline:

    def __init__(self, model_path: str, preprocessor_path: str):

        try:

            logger.info("Loading model and preprocessor...")

            self.model = joblib.load(model_path)
            self.preprocessor = joblib.load(preprocessor_path)

            self.expected_features = [
                "age",
                "gender",
                "employment_status",
                "annual_income",
                "account_age_months",
                "avg_monthly_balance",
                "num_deposits_per_month",
                "avg_deposit_amount",
                "debit_card_usage_frequency",
                "debit_card_spending",
                "mobile_banking_logins",
                "online_transfer_frequency",
                "atm_withdrawal_frequency",
                "credit_score",
                "num_open_loans",
                "total_outstanding_debt",
                "late_payment_count",
                "loan_default_history",
                "fraud_flag",
                "loan_application_amount",
            ]

            logger.info("Artifacts loaded successfully.")

        except Exception as e:
            logger.exception("Failed to load prediction artifacts.")
            raise CustomException(e, sys)

    def predict(self, input_data: dict):

        try:

            logger.info("Starting prediction...")

            # dict -> DataFrame
            df = pd.DataFrame([input_data])

            # Add missing columns
            for col in self.expected_features:
                if col not in df.columns:
                    df[col] = 0

            # Keep the exact training order
            df = df[self.expected_features]

            warnings.filterwarnings(
                "ignore",
                category=UserWarning
            )

            logger.info("Input data prepared successfully.")

            processed = self.preprocessor.transform(df)

            prediction = self.model.predict(processed)

            result = {
                "prediction": prediction.tolist()
            }

            if hasattr(self.model, "predict_proba"):

                result["probability"] = (
                    self.model.predict_proba(processed).tolist()
                )

            logger.info("Prediction completed successfully.")

            return result

        except Exception as e:

            logger.exception("Prediction pipeline failed.")
            raise CustomException(e, sys)