import os
import sys
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.components.constants import DATA_PATH
from src.components.data_validation import load_data
from src.components.data_transformation import DataTransformation
from src.components.logger import logger
from src.exception import CustomException


class ModelTrainer:

    def __init__(self):

        os.makedirs("artifacts", exist_ok=True)

        self.model_path = os.path.join(
            "artifacts",
            "random_forest.pkl"
        )

    def train(self):

        try:

            logger.info("Loading dataset...")

            df = load_data(DATA_PATH)

            X = df.drop("credit_risk", axis=1)
            y = df["credit_risk"]

            logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            logger.info(f"Train shape: X={X_train.shape}, y={y_train.shape}")
            logger.info(f"Test shape: X={X_test.shape}, y={y_test.shape}")

            transformer = DataTransformation()

            logger.info("Fitting data transformer...")
            X_train = transformer.fit_transform(X_train)

            logger.info("Transforming test dataset...")
            X_test = transformer.transform(X_test)

            assert X_train.shape[0] > 0, "X_train is empty."
            assert y_train.shape[0] > 0, "y_train is empty."
            assert X_train.shape[0] == y_train.shape[0], "Mismatch between X_train and y_train."

            logger.info(f"Final training shape: {X_train.shape}")

            model = RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1
            )

            logger.info("Training Random Forest model...")

            model.fit(
                X_train,
                y_train
            )

            logger.info("Model training completed successfully.")

            joblib.dump(
                model,
                self.model_path
            )

            logger.info(
                f"Model saved successfully at: {self.model_path}"
            )

            model_size = os.path.getsize(
                self.model_path
            )

            logger.info(
                f"Model file size: {model_size} bytes"
            )

            if model_size < 1000:
                raise ValueError(
                    "Model file size is suspiciously small."
                )

            return model, X_test, y_test

        except Exception as e:

            logger.exception("Failed during model training.")
            raise CustomException(e, sys)