from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import os
import sys
from dataclasses import dataclass

from src.logger.logging import logging
from src.exceptions.exceptions import customexception
from src.utility.utility import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'xgboost_model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting dependent and independent variables from train and test data')

            # Split the features and target variable (last column) from the train and test arrays
            x_train, y_train = train_array[:, :-1], train_array[:, -1]
            x_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Initialize the XGBoost classifier
            model = XGBClassifier(
                objective='binary:logistic',  # For binary classification
                eval_metric='logloss',         # Logistic loss function
                use_label_encoder=False,       # Suppress warning on label encoding
                random_state=42                # For reproducibility
            )

            # Train the model
            logging.info('Training the XGBoost model')
            model.fit(x_train, y_train)

            # Make predictions on the test data
            logging.info('Making predictions on the test data')
            predictions = model.predict(x_test)

            # Evaluate the model using classification metrics
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions)
            recall = recall_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)

            logging.info(f"Model Evaluation Metrics: Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1-Score={f1}")

            # Save the trained model using utility function
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            logging.info(f"Trained model saved at {self.model_trainer_config.trained_model_file_path}")

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }

        except Exception as e:
            logging.error("Exception occurred during model training")
            raise customexception(e, sys)