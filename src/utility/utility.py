from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import os
import sys
import pickle

from src.logger.logging import logging
from src.exceptions.exceptions import customexception

# Function to save any object (model, preprocessor, etc.) to disk
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved successfully at {file_path}")
    
    except Exception as e:
        logging.error("Error while saving object")
        raise customexception(e, sys)

# Function to evaluate model performance on test data
def evaluate_model(x_train, y_train, x_test, y_test, model):
    try:
        # Train the model
        model.fit(x_train, y_train)
        logging.info(f"Model {model} trained successfully")

        # Predict on test data
        y_test_pred = model.predict(x_test)
        logging.info("Predictions made on test data")

        # Classification metrics
        accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)

        # Logging metrics
        logging.info(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

        # Return a dictionary of the metrics
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    except Exception as e:
        logging.error("Error during model evaluation")
        raise customexception(e, sys)

# Function to load any object (model, preprocessor, etc.) from disk
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            obj = pickle.load(file_obj)
        logging.info(f"Object loaded successfully from {file_path}")
        return obj
    except Exception as e:
        logging.error("Error while loading object")
        raise customexception(e, sys)