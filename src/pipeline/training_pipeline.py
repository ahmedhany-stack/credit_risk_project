import sys

from src.components.logger import logger
from src.exception import CustomException

from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher


class TrainingPipeline:

    def __init__(self):

        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluation()
        self.pusher = ModelPusher()

    def run_pipeline(self):

        try:

            logger.info("=" * 80)
            logger.info("Training Pipeline Started")
            logger.info("=" * 80)

            # =======================
            # Train Model
            # =======================

            model, X_test, y_test = self.trainer.train()

            logger.info("Model training completed.")

            # =======================
            # Evaluate Model
            # =======================

            metrics = self.evaluator.evaluate(
                model=model,
                X_test=X_test,
                y_test=y_test
            )

            logger.info("Model evaluation completed.")

            # =======================
            # Save Model
            # =======================

            self.pusher.push(model)

            logger.info("Model saved successfully.")

            logger.info("=" * 80)
            logger.info("Training Pipeline Finished Successfully")
            logger.info("=" * 80)

            return metrics

        except Exception as e:

            logger.exception("Training pipeline failed.")
            raise CustomException(e, sys)


if __name__ == "__main__":

    pipeline = TrainingPipeline()

    metrics = pipeline.run_pipeline()

    print(metrics)