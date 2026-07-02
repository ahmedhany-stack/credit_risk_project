import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.components.logger import logger


class DataTransformation:

    def __init__(self):

        os.makedirs("artifacts", exist_ok=True)
        self.preprocessor_path = "artifacts/preprocessor.pkl"

        # ✅ FIXED: remove customer_id (important bug fix)
        self.categorical_columns = [
            "gender",
            "employment_status"
        ]

        self.numerical_columns = [
            'age', 'annual_income', 'account_age_months',
            'avg_monthly_balance', 'num_deposits_per_month',
            'avg_deposit_amount', 'debit_card_usage_frequency',
            'debit_card_spending', 'mobile_banking_logins',
            'online_transfer_frequency', 'atm_withdrawal_frequency',
            'credit_score', 'num_open_loans',
            'total_outstanding_debt', 'late_payment_count',
            'loan_default_history', 'fraud_flag',
            'loan_application_amount'
        ]

    def build_preprocessor(self):

        try:

            logger.info("Building preprocessing pipeline...")

            categorical_pipeline = Pipeline([
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])

            numerical_pipeline = Pipeline([
                ("scaler", StandardScaler())
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", categorical_pipeline, self.categorical_columns),
                    ("num", numerical_pipeline, self.numerical_columns)
                ],
                remainder="drop"
            )

            return preprocessor

        except Exception as e:
            logger.exception(e)
            raise

    # =========================
    # TRAINING
    # =========================
    def fit_transform(self, df: pd.DataFrame):

        try:

            logger.info("Starting fit_transform...")

            preprocessor = self.build_preprocessor()

            transformed = preprocessor.fit_transform(df)

            feature_names = preprocessor.get_feature_names_out()

            transformed = pd.DataFrame(
                transformed,
                columns=feature_names,
                index=df.index
            )

            joblib.dump(preprocessor, self.preprocessor_path)

            logger.info(f"Preprocessor saved at {self.preprocessor_path}")

            return transformed

        except Exception as e:
            logger.exception(e)
            raise

    # =========================
    # INFERENCE
    # =========================
    def transform(self, df: pd.DataFrame):

        try:

            logger.info("Loading preprocessor...")

            preprocessor = joblib.load(self.preprocessor_path)

            transformed = preprocessor.transform(df)

            transformed = pd.DataFrame(
                transformed,
                columns=preprocessor.get_feature_names_out(),
                index=df.index
            )

            return transformed

        except Exception as e:
            logger.exception(e)
            raise