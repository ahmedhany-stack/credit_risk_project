from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from src.components.logger import logger


class ModelEvaluation:

    def evaluate(self, model, X_test, y_test):

        try:

            logger.info("Evaluating model...")

            y_pred = model.predict(X_test)

            y_prob = model.predict_proba(X_test)[:, 1]

            metrics = {

                "Accuracy":
                    accuracy_score(y_test, y_pred),

                "Precision":
                    precision_score(
                        y_test,
                        y_pred
                    ),

                "Recall":
                    recall_score(
                        y_test,
                        y_pred
                    ),

                "F1":
                    f1_score(
                        y_test,
                        y_pred
                    ),

                "ROC_AUC":
                    roc_auc_score(
                        y_test,
                        y_prob
                    ),

                "Confusion Matrix":
                    confusion_matrix(
                        y_test,
                        y_pred
                    ),

                "Classification Report":
                    classification_report(
                        y_test,
                        y_pred
                    )

            }

            logger.info("Evaluation Finished.")

            return metrics

        except Exception as e:

            logger.exception(e)
            raise