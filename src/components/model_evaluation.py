from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import sys
from src.utility.utility import load_object
from src.logger.logging import logging
from src.exceptions.exceptions import customexception

class ModelEvaluation:
    def __init__(self):
        logging.info("Model evaluation started")

    def eval_metrics(self, actual, pred):
        """
        Calculate evaluation metrics for classification.
        """
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred, average='binary')  # For binary classification
        recall = recall_score(actual, pred, average='binary')
        f1 = f1_score(actual, pred, average='binary')
        logging.info("Evaluation metrics calculated: Accuracy, Precision, Recall, F1-Score")
        return accuracy, precision, recall, f1

    def initiate_model_evaluation(self, train_array, test_array):
        """
        Evaluate the model on the test data.
        """
        try:
            # Extract input features and target from test dataset
            x_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Load the trained model from the specified path
            model_path = os.path.join('artifacts', 'xgboost_model.pkl')
            model = load_object(model_path)

            logging.info("Model loaded successfully")

            # Generate predictions
            predictions = model.predict(x_test)

            # Compute evaluation metrics
            accuracy, precision, recall, f1 = self.eval_metrics(y_test, predictions)

            # Log metrics for debugging
            logging.info(f"Accuracy: {accuracy}")
            logging.info(f"Precision: {precision}")
            logging.info(f"Recall: {recall}")
            logging.info(f"F1 Score: {f1}")

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }

        except Exception as e:
            logging.error("Exception occurred during model evaluation")
            raise customexception(e, sys)