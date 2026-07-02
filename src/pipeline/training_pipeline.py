from src.components.logger import logger

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

            logger.info("Model Training Completed")

            # =======================
            # Evaluate Model
            # =======================

            metrics = self.evaluator.evaluate(
                model=model,
                X_test=X_test,
                y_test=y_test
            )

            logger.info("Model Evaluation Completed")

            # =======================
            # Push Model
            # =======================

            self.pusher.push(model)

            logger.info("Model Pushed Successfully")

            logger.info("=" * 80)
            logger.info("Training Pipeline Finished Successfully")
            logger.info("=" * 80)

            return metrics

        except Exception as e:

            logger.exception(e)
            raise