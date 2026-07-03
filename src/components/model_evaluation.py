import sys

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

from src.components.logger import logger
from src.exception import CustomException


class ModelEvaluation:

    def evaluate(self, model, X_test, y_test):

        try:

            logger.info("Starting model evaluation...")

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics = {

                "Accuracy": accuracy_score(
                    y_test,
                    y_pred
                ),

                "Precision": precision_score(
                    y_test,
                    y_pred
                ),

                "Recall": recall_score(
                    y_test,
                    y_pred
                ),

                "F1 Score": f1_score(
                    y_test,
                    y_pred
                ),

                "ROC AUC": roc_auc_score(
                    y_test,
                    y_prob
                ),

                "Confusion Matrix": confusion_matrix(
                    y_test,
                    y_pred
                ),

                "Classification Report": classification_report(
                    y_test,
                    y_pred
                )

            }

            logger.info("Model evaluation completed successfully.")

            return metrics

        except Exception as e:

            logger.exception("Failed during model evaluation.")
            raise CustomException(e, sys)