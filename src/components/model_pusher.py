import joblib
import os
from src.components.logger import logger


class ModelPusher:

    def __init__(self):

        os.makedirs("artifacts", exist_ok=True)

        self.model_path = "artifacts/random_forest.pkl"

    def push(self, model):

        try:

            logger.info("Saving model...")

            joblib.dump(model, self.model_path)

            logger.info(f"Model saved at {self.model_path}")

        except Exception as e:

            logger.exception(e)
            raise