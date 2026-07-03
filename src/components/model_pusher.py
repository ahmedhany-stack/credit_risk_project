import os
import sys
import joblib

from src.components.logger import logger
from src.exception import CustomException


class ModelPusher:

    def __init__(self):

        os.makedirs("artifacts", exist_ok=True)

        self.model_path = os.path.join(
            "artifacts",
            "random_forest.pkl"
        )

    def push(self, model):

        try:

            logger.info("Saving trained model...")

            joblib.dump(
                model,
                self.model_path
            )

            logger.info(
                f"Model saved successfully at: {self.model_path}"
            )

        except Exception as e:

            logger.exception("Failed to save trained model.")
            raise CustomException(e, sys)